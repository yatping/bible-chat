import Foundation

/// A scripted `LLMEngine` for tests, previews and the DEBUG "mock model" toggle.
///
/// The responder receives the request and returns the full reply, which is streamed in small pieces.
public final class MockLLMEngine: LLMEngine, @unchecked Sendable {
    public typealias Responder = @Sendable ([ChatMessage], GenerationConfig) -> String

    private let responder: Responder
    private let tokenDelay: Duration
    private let lock = NSLock()
    private var _requests: [[ChatMessage]] = []

    /// - Parameters:
    ///   - tokenDelay: pause between streamed pieces (simulates generation speed).
    ///   - responder: produces the reply for a request.
    public init(tokenDelay: Duration = .zero, responder: @escaping Responder) {
        self.tokenDelay = tokenDelay
        self.responder = responder
    }

    /// Every request received so far.
    public var requests: [[ChatMessage]] {
        lock.withLock { _requests }
    }

    public func load() async throws {}

    public func generate(messages: [ChatMessage], config: GenerationConfig) -> AsyncThrowingStream<String, Error> {
        lock.withLock { _requests.append(messages) }
        let reply = responder(messages, config)
        let delay = tokenDelay
        return AsyncThrowingStream { continuation in
            let task = Task {
                for piece in Self.pieces(of: reply) {
                    if Task.isCancelled { break }
                    if delay > .zero { try? await Task.sleep(for: delay) }
                    continuation.yield(piece)
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Splits text into word-ish pieces, keeping whitespace attached (like real token pieces).
    static func pieces(of text: String) -> [String] {
        var pieces: [String] = []
        var current = ""
        for character in text {
            if character.isWhitespace, !current.isEmpty, !(current.last?.isWhitespace ?? false) {
                pieces.append(current)
                current = ""
            }
            current.append(character)
        }
        if !current.isEmpty { pieces.append(current) }
        return pieces
    }

    /// A demo responder: answers from the prompt's sources, extracts simple preferences, summarises tersely.
    public static func demo(tokenDelay: Duration = .milliseconds(25)) -> MockLLMEngine {
        MockLLMEngine(tokenDelay: tokenDelay) { messages, _ in
            let last = messages.last?.content ?? ""
            if last.hasPrefix("You are the memory extraction component") { return #"{"memories":[]}"# }
            if last.hasPrefix("You maintain a running summary") { return "The user has been studying several passages." }
            let sources = last.components(separatedBy: "\n").filter { $0.hasPrefix("[S") }
            guard let first = sources.first else {
                return "The supplied sources do not cover this question, so I can't answer it reliably."
            }
            let reference = first.components(separatedBy: "reference: ").last ?? "the passage"
            return "This is the mock model (DEBUG). The most relevant passage is \(reference) [S1]."
                + (sources.count > 1 ? " See also [S2]." : "")
        }
    }
}
