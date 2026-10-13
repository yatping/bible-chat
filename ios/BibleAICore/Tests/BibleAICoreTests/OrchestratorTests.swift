import Foundation
import Testing
@testable import BibleAICore

/// A mock model that routes by request kind: memory extraction, summarisation or answer.
func scriptedEngine(answer: @escaping @Sendable (_ contextBlock: String) -> String = { _ in "OK [S1]." }) -> MockLLMEngine {
    MockLLMEngine { messages, _ in
        let last = messages.last?.content ?? ""
        if last.hasPrefix("You are the memory extraction component") {
            let message = last.components(separatedBy: "USER MESSAGE:\n").last ?? ""
            return extractionReply(for: message)
        }
        if last.hasPrefix("You maintain a running summary") {
            return "The user asked many questions about scripture."
        }
        return answer(last)
    }
}

/// What a well-behaved extraction model would answer for the scenario's messages.
func extractionReply(for message: String) -> String {
    let lower = message.lowercased()
    if lower.contains("prefer short answers") {
        return #"{"memories":[{"type":"preference","key":"answer_length","value":"User prefers short answers.","importance":0.8,"confidence":1.0}]}"#
    }
    if lower.contains("prefer detailed explanations") {
        return #"Sure: ```json {"memories":[{"type":"preference","key":"answer_length","value":"User prefers detailed explanations.","importance":0.8}]} ```"#
    }
    if lower.contains("web translation") {
        return #"{"memories":[{"type":"preference","key":"bible_translation","value":"Preferred Bible translation is WEB.","importance":0.8,"confidence":1.0}]}"#
    }
    return #"{"memories":[]}"#
}

func collect(_ stream: AsyncThrowingStream<ChatEvent, Error>) async throws -> (tokens: String, sources: [RetrievedSource], completed: (String, [CitedSource])?) {
    var tokens = ""
    var sources: [RetrievedSource] = []
    var completed: (String, [CitedSource])?
    for try await event in stream {
        switch event {
        case .userMessage: break
        case .sources(let s, _): sources = s
        case .token(let t): tokens += t
        case .completed(let answer, let citations, _): completed = (answer, citations)
        }
    }
    return (tokens, sources, completed)
}

@Suite("AI orchestrator end-to-end")
struct OrchestratorTests {
    @Test func answersValidatesCitationsPersistsAndExtractsMemory() async throws {
        let content = try TestContent.make()
        let userDB = try UserDatabase.inMemory()
        let llm = scriptedEngine { _ in
            "Peter suggested seven times, but Jesus said seventy times seven [S1]. The commentary notes this means without limit [S2][S9]. As in [S9] the king forgave the debt."
        }
        let orchestrator = AIOrchestrator(llm: llm, embedder: FakeEmbedder(), rag: try content.ragEngine(), userDatabase: userDB)
        let conversation = try ConversationStore(database: userDB).createConversation()

        let result = try await collect(orchestrator.ask("I use the WEB translation. Why did Jesus tell Peter to forgive 70x7?",
                                                        conversationID: conversation.id))
        await orchestrator.waitForBackgroundWork()

        #expect(result.tokens.contains("[S9]")) // raw stream is unvalidated
        let (answer, citations) = try #require(result.completed)
        #expect(!answer.contains("[S9]"))
        #expect(answer.hasSuffix("without limit [S2]. As in the king forgave the debt."))
        #expect(citations.map(\.id) == ["S1", "S2"])
        #expect(citations.first?.reference == result.sources.first?.reference)

        // Persisted: user question (raw) + validated answer with sources_json.
        let messages = try ConversationStore(database: userDB).messages(in: conversation.id)
        #expect(messages.map(\.role) == [.user, .assistant])
        #expect(messages[0].content == "I use the WEB translation. Why did Jesus tell Peter to forgive 70x7?")
        #expect(messages[1].content == answer)
        #expect(messages[1].sources == citations)
        let json = try userDB.db.queryOne("SELECT sources_json FROM messages WHERE role = 'assistant'")?.string("sources_json")
        #expect(json?.contains("\"chunk_id\":") == true)
        #expect(try ConversationStore(database: userDB).conversation(id: conversation.id)?.title?.hasPrefix("I use the WEB") == true)

        // Prompt: system + context block with sources, no memory yet.
        let answerPrompt = try #require(llm.requests.first)
        #expect(answerPrompt.first?.role == .system)
        #expect(answerPrompt.last?.content.contains("SOURCES\n[S1] type: ") == true)
        #expect(answerPrompt.last?.content.contains("| reference: Matthew 18:21-35\n") == true)
        #expect(answerPrompt.last?.content.hasPrefix("MEMORY") == false)

        // Memory extraction ran after the answer and stored the translation preference.
        let memories = try MemoryStore(database: userDB).activeMemories()
        #expect(memories.map(\.content) == ["Preferred Bible translation is WEB."])
        #expect(memories.first?.sourceMessageID == messages[0].id)

        // Next turn: memory injected, history replayed raw.
        _ = try await collect(orchestrator.ask("And Genesis 1?", conversationID: conversation.id))
        let second = try #require(llm.requests.last { $0.first?.role == .system })
        #expect(second.count == 4)
        #expect(second[1] == .user("I use the WEB translation. Why did Jesus tell Peter to forgive 70x7?"))
        #expect(second[2] == .assistant(answer))
        #expect(second[3].content.hasPrefix("MEMORY\n[M1] Preferred Bible translation is WEB.\n\nSOURCES\n[S1] type: scripture | reference: Genesis 1:1-2:3"))
    }

    @Test func memoryDisabledSkipsRetrievalAndExtraction() async throws {
        let userDB = try UserDatabase.inMemory()
        let llm = scriptedEngine { _ in "No sources [S1]." }
        let orchestrator = AIOrchestrator(llm: llm, embedder: nil, rag: nil, userDatabase: userDB,
                                          settings: OrchestratorSettings(memoryEnabled: false))
        let conversation = try ConversationStore(database: userDB).createConversation()
        let result = try await collect(orchestrator.ask("I use the WEB translation.", conversationID: conversation.id))
        await orchestrator.waitForBackgroundWork()
        #expect(result.completed?.0 == "No sources.")
        #expect(result.completed?.1.isEmpty == true)
        #expect(llm.requests.count == 1)
        #expect(llm.requests[0].last?.content == "SOURCES\n(no sources retrieved)\n\nQUESTION\nI use the WEB translation.")
        #expect(try MemoryStore(database: userDB).activeMemories().isEmpty)
    }

    @Test func cancellingStopsStreamingAndKeepsPartialAnswer() async throws {
        let userDB = try UserDatabase.inMemory()
        let llm = MockLLMEngine(tokenDelay: .milliseconds(20)) { messages, _ in
            messages.last!.content.hasPrefix("You are") ? #"{"memories":[]}"# : String(repeating: "word ", count: 200)
        }
        let orchestrator = AIOrchestrator(llm: llm, embedder: nil, rag: nil, userDatabase: userDB)
        let conversation = try ConversationStore(database: userDB).createConversation()
        let consumer = Task {
            var count = 0
            for try await event in orchestrator.ask("Tell me a long story", conversationID: conversation.id) {
                if case .token = event { count += 1 }
                if count == 5 { break }
            }
        }
        try await consumer.value
        try await Task.sleep(for: .milliseconds(300))
        await orchestrator.waitForBackgroundWork()
        let messages = try ConversationStore(database: userDB).messages(in: conversation.id)
        #expect(messages.first?.role == .user)
        if messages.count == 2 {
            #expect(messages[1].content.split(separator: " ").count < 200)
        }
    }
}

/// Design scenario: a stated preference survives 20 unrelated turns (well outside the recent window)
/// and a later contradicting preference supersedes it.
@Suite("Memory evaluation scenario")
struct MemoryEvaluationTests {
    @Test func shortAnswerPreferenceSurvivesAndIsSuperseded() async throws {
        let content = try TestContent.make()
        let userDB = try UserDatabase.inMemory()
        let llm = scriptedEngine { _ in "Here is an answer [S1]." }
        let orchestrator = AIOrchestrator(llm: llm, embedder: FakeEmbedder(), rag: try content.ragEngine(), userDatabase: userDB)
        let conversation = try ConversationStore(database: userDB).createConversation()

        _ = try await collect(orchestrator.ask("I prefer short answers.", conversationID: conversation.id))
        let unrelated = ["What happened on the first day of creation?", "Who was Nicodemus?", "What does John 3:16 say?",
                         "What is the unforgiving servant parable about?", "What does Romans 8:28 promise?"]
        for i in 0..<20 {
            _ = try await collect(orchestrator.ask(unrelated[i % unrelated.count], conversationID: conversation.id))
        }

        _ = try await collect(orchestrator.ask("Explain Romans 5.", conversationID: conversation.id))
        let romans = try #require(llm.requests.last { $0.first?.role == .system })
        let context = try #require(romans.last?.content)
        #expect(context.hasPrefix("MEMORY\n[M1] User prefers short answers.\n\n"))
        #expect(context.contains("reference: Romans 5:1-11"))
        // The preference came from memory, not from replayed history (only messages the summary doesn't cover).
        #expect(romans.count > 2 && romans.count <= 1 + 16 + 1)
        #expect(!romans.dropLast().contains { $0.content == "I prefer short answers." })
        // 42 messages → the rolling summary has kicked in.
        await orchestrator.waitForBackgroundWork()
        #expect(context.contains("CONVERSATION SUMMARY\nThe user asked many questions about scripture."))

        _ = try await collect(orchestrator.ask("I now prefer detailed explanations.", conversationID: conversation.id))
        await orchestrator.waitForBackgroundWork()
        let store = MemoryStore(database: userDB)
        #expect(try store.activeMemory(key: .answerLength)?.content == "User prefers detailed explanations.")
        #expect(try store.supersededMemories().map(\.content) == ["User prefers short answers."])

        _ = try await collect(orchestrator.ask("Explain Romans 5 again.", conversationID: conversation.id))
        let after = try #require(llm.requests.last { $0.first?.role == .system }?.last?.content)
        #expect(after.contains("[M1] User prefers detailed explanations."))
        #expect(!after.contains("short answers"))
    }
}
