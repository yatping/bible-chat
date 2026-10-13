import Foundation

/// Maintains the rolling conversation summary (CONTRACTS §5).
///
/// Messages not covered by the summary are replayed verbatim. Once more than `threshold` are uncovered, all but the
/// last `recentWindow` are folded into the summary with `shared/summarization_prompt.txt`, so every message is always
/// either summarised or replayed.
public struct ConversationSummarizer: Sendable {
    public var recentWindow = TokenBudget.default.recentMessageCount
    public var threshold = TokenBudget.default.maxReplayedMessages
    /// Per-message character cap when building the summarisation input (keeps the request inside n_ctx).
    public var maxCharsPerMessage = 600

    public let llm: any LLMEngine
    public let conversations: ConversationStore
    public let budget: TokenBudget

    public init(llm: any LLMEngine, conversations: ConversationStore, budget: TokenBudget = .default) {
        self.llm = llm
        self.conversations = conversations
        self.budget = budget
    }

    /// Message indices to fold into the summary, or nil if no summary update is due.
    public static func pendingRange(totalMessages: Int, covered: Int, recentWindow: Int = 8, threshold: Int = 16) -> Range<Int>? {
        guard covered >= 0, totalMessages - covered > threshold else { return nil }
        return covered..<(totalMessages - recentWindow)
    }

    /// Updates the summary if due. Returns true when a new summary was stored.
    @discardableResult
    public func summarizeIfNeeded(conversationID: String) async throws -> Bool {
        guard let conversation = try conversations.conversation(id: conversationID) else { return false }
        let messages = try conversations.messages(in: conversationID)
        guard let range = Self.pendingRange(totalMessages: messages.count, covered: conversation.summaryMessageCount,
                                            recentWindow: recentWindow, threshold: threshold) else { return false }
        let prompt = prompt(previousSummary: conversation.summary, messages: Array(messages[range]))
        let raw = try await llm.complete(messages: [.user(prompt)], config: .deterministic)
        let summary = TokenBudget.truncate(raw.trimmingCharacters(in: .whitespacesAndNewlines), toTokens: budget.summary)
        guard !summary.isEmpty else { return false }
        try conversations.updateSummary(conversationID, summary: summary, coveredMessageCount: range.upperBound)
        return true
    }

    /// `shared/summarization_prompt.txt` with placeholders filled.
    public func prompt(previousSummary: String?, messages: [StoredMessage]) -> String {
        var template = SharedTexts.summarizationPrompt
        while template.hasSuffix("\n") { template.removeLast() }
        let previous = previousSummary?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let lines = messages.map { message in
            let speaker = message.role == .user ? "User" : "Assistant"
            var content = message.content.trimmingCharacters(in: .whitespacesAndNewlines)
            if content.count > maxCharsPerMessage { content = String(content.prefix(maxCharsPerMessage)) + "…" }
            return "\(speaker): \(content)"
        }
        return template
            .replacingOccurrences(of: "{{previous_summary}}", with: previous.isEmpty ? "(none)" : previous)
            .replacingOccurrences(of: "{{messages}}", with: lines.joined(separator: "\n"))
    }
}
