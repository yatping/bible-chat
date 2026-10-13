/// Token budgets for the on-device context window (CONTRACTS §5).
/// Token counts are estimated as `ceil(chars / 4)`, with chars counted as Unicode scalars
/// (matching Python's `len`, which the content builder uses for `token_estimate`).
public struct TokenBudget: Sendable, Hashable {
    public var contextLength = 6144
    public var system = 300
    public var memory = 300
    public var summary = 300
    public var recentMessages = 1000
    /// Messages kept verbatim after a summary update.
    public var recentMessageCount = 8
    /// Upper bound on replayed (not yet summarised) messages; also the summarisation trigger.
    public var maxReplayedMessages = 16
    public var sources = 2500
    public var answer = 1200

    public init() {}

    public static let `default` = TokenBudget()

    /// `ceil(chars / 4)`.
    public static func estimate(_ text: String) -> Int {
        (text.unicodeScalars.count + 3) / 4
    }

    /// The last `maxReplayedMessages` messages, dropping the oldest until they fit `recentMessages` tokens.
    public func trimRecent(_ messages: [ChatMessage]) -> [ChatMessage] {
        var window = Array(messages.suffix(maxReplayedMessages))
        while !window.isEmpty, window.reduce(0, { $0 + Self.estimate($1.content) }) > recentMessages {
            window.removeFirst()
        }
        return window
    }

    /// Memories in order, stopping before the first one that would exceed the memory budget.
    public func trimMemories(_ memories: [String]) -> [String] {
        var used = 0
        var kept: [String] = []
        for memory in memories {
            let cost = Self.estimate(memory)
            if used + cost > self.memory { break }
            used += cost
            kept.append(memory)
        }
        return kept
    }

    /// Truncates the conversation summary to its budget (on a word boundary where possible).
    public func trimSummary(_ summary: String?) -> String? {
        guard let summary else { return nil }
        return Self.truncate(summary, toTokens: self.summary)
    }

    /// Truncates `text` to at most `tokens` estimated tokens.
    public static func truncate(_ text: String, toTokens tokens: Int) -> String {
        let maxScalars = tokens * 4
        guard text.unicodeScalars.count > maxScalars else { return text }
        var scalars = String.UnicodeScalarView(text.unicodeScalars.prefix(max(0, maxScalars - 1)))
        if let lastSpace = scalars.lastIndex(where: { $0 == " " || $0 == "\n" }),
           scalars.distance(from: scalars.startIndex, to: lastSpace) > maxScalars / 2 {
            scalars = String.UnicodeScalarView(scalars[..<lastSpace])
        }
        return String(scalars) + "…"
    }
}
