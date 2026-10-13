import Foundation
import OSLog

/// Events emitted while answering one question.
public enum ChatEvent: Sendable {
    /// The user's message was persisted.
    case userMessage(StoredMessage)
    /// Retrieved sources (S1…Sn) and the memories injected into the prompt.
    case sources([RetrievedSource], memories: [MemoryRecord])
    /// A streamed piece of the raw answer.
    case token(String)
    /// Final validated answer (invalid citations removed), its cited sources and the persisted message.
    case completed(answer: String, citations: [CitedSource], message: StoredMessage)
}

/// Runtime switches the user controls.
public struct OrchestratorSettings: Sendable, Hashable {
    /// Long-term memory on/off (retrieval and extraction).
    public var memoryEnabled = true
    public init(memoryEnabled: Bool = true) { self.memoryEnabled = memoryEnabled }
}

/// Answers questions: retrieval + memory → prompt → streamed generation → citation validation → persistence,
/// then memory extraction and rolling summarisation in the background.
public actor AIOrchestrator {
    public let llm: any LLMEngine
    public let embedder: (any EmbeddingEngine)?
    public let rag: RAGEngine?
    public let conversations: ConversationStore
    public let memories: MemoryStore
    public private(set) var settings: OrchestratorSettings

    private let budget: TokenBudget
    private let promptBuilder: PromptBuilder
    private let memoryManager: MemoryManager
    private let memoryRetriever: MemoryRetriever
    private let summarizer: ConversationSummarizer
    private var background: Task<Void, Never>?
    private static let log = Logger(subsystem: "BibleAICore", category: "Orchestrator")

    public init(llm: any LLMEngine, embedder: (any EmbeddingEngine)?, rag: RAGEngine?, userDatabase: UserDatabase,
                settings: OrchestratorSettings = OrchestratorSettings(), budget: TokenBudget = .default,
                promptBuilder: PromptBuilder = PromptBuilder()) {
        self.llm = llm
        self.embedder = embedder
        self.rag = rag
        self.settings = settings
        self.budget = budget
        self.promptBuilder = promptBuilder
        conversations = ConversationStore(database: userDatabase)
        memories = MemoryStore(database: userDatabase)
        memoryManager = MemoryManager(store: memories, embedder: embedder)
        memoryRetriever = MemoryRetriever(store: memories)
        summarizer = ConversationSummarizer(llm: llm, conversations: conversations, budget: budget)
    }

    public func updateSettings(_ settings: OrchestratorSettings) {
        self.settings = settings
    }

    /// Answers `question` in `conversationID`, streaming events. Cancelling the consumer stops generation;
    /// any partial answer is still validated and saved.
    public nonisolated func ask(_ question: String, conversationID: String) -> AsyncThrowingStream<ChatEvent, Error> {
        let (stream, continuation) = AsyncThrowingStream.makeStream(of: ChatEvent.self)
        let task = Task {
            do {
                try await self.answer(question, conversationID: conversationID, continuation: continuation)
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { _ in task.cancel() }
        return stream
    }

    /// Waits for pending memory extraction / summarisation (tests, app shutdown).
    public func waitForBackgroundWork() async {
        await background?.value
    }

    /// Builds the prompt messages for `question` without generating (debugging / previews).
    public func preparePrompt(_ question: String, conversationID: String) async throws -> (messages: [ChatMessage], sources: [RetrievedSource], memories: [MemoryRecord]) {
        let history = try conversations.messages(in: conversationID)
        return try await buildPrompt(question, history: history, conversation: try conversations.conversation(id: conversationID))
    }

    // MARK: - Pipeline

    private func answer(_ rawQuestion: String, conversationID: String,
                        continuation: AsyncThrowingStream<ChatEvent, Error>.Continuation) async throws {
        // Let the previous turn's memory extraction land first, so "I prefer …" applies to the next question.
        await background?.value

        let question = rawQuestion.trimmingCharacters(in: .whitespacesAndNewlines)
        let history = try conversations.messages(in: conversationID)
        let conversation = try conversations.conversation(id: conversationID)
        let userMessage = try conversations.append(StoredMessage(conversationID: conversationID, role: .user, content: question))
        continuation.yield(.userMessage(userMessage))

        let prompt = try await buildPrompt(question, history: history, conversation: conversation)
        continuation.yield(.sources(prompt.sources, memories: prompt.memories))

        var raw = ""
        do {
            for try await piece in llm.generate(messages: prompt.messages, config: .answer) {
                raw += piece
                continuation.yield(.token(piece))
            }
        } catch is CancellationError {
            Self.log.info("generation cancelled")
        }

        let validated = CitationValidator.validate(raw.trimmingCharacters(in: .whitespacesAndNewlines),
                                                   sourceCount: prompt.sources.count)
        if !validated.removed.isEmpty {
            Self.log.notice("removed invalid citations: \(validated.removed.joined(separator: ","), privacy: .public)")
        }
        let byID = Dictionary(uniqueKeysWithValues: prompt.sources.map { ($0.id, $0) })
        let citations = validated.cited.compactMap { byID[$0]?.cited }
        let answerText = validated.text.trimmingCharacters(in: .whitespacesAndNewlines)

        if !answerText.isEmpty {
            let assistant = try conversations.append(StoredMessage(conversationID: conversationID, role: .assistant,
                                                                   content: answerText, sources: citations))
            continuation.yield(.completed(answer: answerText, citations: citations, message: assistant))
        }
        scheduleBackgroundWork(question: question, userMessageID: userMessage.id, conversationID: conversationID)
    }

    private func buildPrompt(_ question: String, history: [StoredMessage], conversation: Conversation?) async throws
        -> (messages: [ChatMessage], sources: [RetrievedSource], memories: [MemoryRecord]) {
        let memoryEnabled = settings.memoryEnabled
        let queryVector: [Float]? = if let embedder {
            try? await embedder.embed(question, kind: .query)
        } else {
            nil
        }

        async let sources = retrieveSources(question, queryVector: queryVector)
        async let selected = retrieveMemories(queryVector: queryVector, enabled: memoryEnabled)
        let (retrieved, memoryRecords) = try await (sources, selected)

        // Replay only what the rolling summary does not already cover (CONTRACTS §5).
        let uncovered = history.dropFirst(min(conversation?.summaryMessageCount ?? 0, history.count))
        let recent = budget.trimRecent(uncovered.map(\.chatMessage))
        let input = PromptInput(
            memories: budget.trimMemories(memoryRecords.map(\.content)),
            summary: budget.trimSummary(conversation?.summary),
            sources: retrieved.map(\.promptSource),
            recentMessages: recent,
            question: question
        )
        return (promptBuilder.messages(for: input), retrieved, memoryRecords)
    }

    private nonisolated func retrieveSources(_ question: String, queryVector: [Float]?) async throws -> [RetrievedSource] {
        guard let rag else { return [] }
        return try await rag.retrieve(question, queryVector: queryVector)
    }

    private nonisolated func retrieveMemories(queryVector: [Float]?, enabled: Bool) async throws -> [MemoryRecord] {
        guard enabled else { return [] }
        return try memoryRetriever.retrieve(queryVector: queryVector).map(\.memory)
    }

    private func scheduleBackgroundWork(question: String, userMessageID: String, conversationID: String) {
        let previous = background
        let memoryEnabled = settings.memoryEnabled
        let llm = llm, manager = memoryManager, summarizer = summarizer
        background = Task.detached(priority: .utility) {
            await previous?.value
            if memoryEnabled {
                do {
                    let raw = try await llm.complete(messages: MemoryExtractor.messages(for: question), config: .deterministic)
                    let extracted = MemoryExtractor.parse(raw)
                    try await manager.consolidate(extracted, sourceMessageID: userMessageID)
                } catch {
                    Self.log.error("memory extraction failed: \(String(describing: error), privacy: .public)")
                }
            }
            do {
                try await summarizer.summarizeIfNeeded(conversationID: conversationID)
            } catch {
                Self.log.error("summarisation failed: \(String(describing: error), privacy: .public)")
            }
        }
    }
}
