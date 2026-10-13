import Foundation

/// A retrieved source as shown to the model in the SOURCES section.
public struct PromptSource: Sendable, Hashable {
    public var type: String
    public var reference: String
    public var text: String

    public init(type: String, reference: String, text: String) {
        self.type = type
        self.reference = reference
        self.text = text
    }
}

/// Everything that goes into one answer prompt.
public struct PromptInput: Sendable, Hashable {
    public var memories: [String]
    public var summary: String?
    public var sources: [PromptSource]
    /// Recent conversation turns, oldest first: raw questions and validated answers.
    public var recentMessages: [ChatMessage]
    public var question: String

    public init(memories: [String] = [], summary: String? = nil, sources: [PromptSource] = [],
                recentMessages: [ChatMessage] = [], question: String) {
        self.memories = memories
        self.summary = summary
        self.sources = sources
        self.recentMessages = recentMessages
        self.question = question
    }
}

/// Builds the chat messages of CONTRACTS §1. Golden-tested against `shared/fixtures/prompt_golden.json`;
/// the training pipeline renders exactly the same messages, so any change here is a contract change.
public struct PromptBuilder: Sendable {
    public let systemPrompt: String

    /// - Parameter systemPrompt: defaults to `shared/system_prompt.txt`; the trailing newline is stripped.
    public init(systemPrompt: String = SharedTexts.systemPrompt) {
        var prompt = systemPrompt
        while prompt.hasSuffix("\n") { prompt.removeLast() }
        self.systemPrompt = prompt
    }

    /// `[system, ...recent, user(context block)]`.
    public func messages(for input: PromptInput) -> [ChatMessage] {
        [.system(systemPrompt)] + input.recentMessages + [.user(contextBlock(for: input))]
    }

    /// The final user turn: MEMORY, CONVERSATION SUMMARY, SOURCES, QUESTION.
    public func contextBlock(for input: PromptInput) -> String {
        var sections: [String] = []

        let memories = input.memories.map(Self.trim)
        if !memories.isEmpty {
            let lines = memories.enumerated().map { "[M\($0.offset + 1)] \($0.element)" }
            sections.append((["MEMORY"] + lines).joined(separator: "\n"))
        }

        let summary = Self.trim(input.summary ?? "")
        if !summary.isEmpty {
            sections.append("CONVERSATION SUMMARY\n\(summary)")
        }

        if input.sources.isEmpty {
            sections.append("SOURCES\n(no sources retrieved)")
        } else {
            let blocks = input.sources.enumerated().map { offset, source in
                "[S\(offset + 1)] type: \(Self.trim(source.type)) | reference: \(Self.trim(source.reference))\n\(Self.trim(source.text))"
            }
            sections.append("SOURCES\n" + blocks.joined(separator: "\n\n"))
        }

        sections.append("QUESTION\n\(Self.trim(input.question))")
        return sections.joined(separator: "\n\n")
    }

    static func trim(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
