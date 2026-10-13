import BibleAICore
import Foundation
import Observation

/// State of one chat screen: persisted messages plus the answer currently streaming.
@MainActor
@Observable
final class ChatViewModel {
    struct Item: Identifiable, Equatable {
        let id: String
        let role: StoredMessage.Role
        var text: String
        var citations: [CitedSource]
        var isStreaming = false
    }

    let conversationID: String
    private(set) var items: [Item] = []
    /// Sources retrieved for the answer currently streaming (S1…Sn).
    private(set) var liveSources: [RetrievedSource] = []
    private(set) var isGenerating = false
    var errorMessage: String?

    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private let env: AppEnvironment

    init(conversationID: String, env: AppEnvironment) {
        self.conversationID = conversationID
        self.env = env
        reload()
    }

    func reload() {
        let messages = (try? env.conversations?.messages(in: conversationID)) ?? []
        items = messages.map { Item(id: $0.id, role: $0.role, text: $0.content, citations: $0.sources ?? []) }
    }

    func send(_ question: String) {
        let trimmed = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isGenerating, let orchestrator = env.orchestrator else { return }
        isGenerating = true
        errorMessage = nil
        liveSources = []
        let streamingID = UUID().uuidString
        task = Task {
            do {
                for try await event in orchestrator.ask(trimmed, conversationID: conversationID) {
                    switch event {
                    case .userMessage(let message):
                        items.append(Item(id: message.id, role: .user, text: message.content, citations: []))
                        items.append(Item(id: streamingID, role: .assistant, text: "", citations: [], isStreaming: true))
                    case .sources(let sources, _):
                        liveSources = sources
                    case .token(let piece):
                        if let index = items.lastIndex(where: { $0.id == streamingID }) { items[index].text += piece }
                    case .completed(let answer, let citations, let message):
                        if let index = items.lastIndex(where: { $0.id == streamingID }) {
                            items[index] = Item(id: message.id, role: .assistant, text: answer, citations: citations)
                        }
                    }
                }
            } catch {
                errorMessage = String(describing: error)
            }
            finish()
        }
    }

    func stop() {
        task?.cancel()
    }

    private func finish() {
        isGenerating = false
        task = nil
        // Replace the streamed (unvalidated) text with what was persisted.
        reload()
    }

    /// Looks up the text of a cited chunk for the source sheet.
    func chunk(for citation: CitedSource) -> Chunk? {
        if let live = liveSources.first(where: { $0.chunk.id == citation.chunkID }) { return live.chunk }
        return try? env.content?.chunks(ids: [citation.chunkID]).first
    }
}
