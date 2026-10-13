/// A local text generator. Implementations: `GemmaEngine` (llama.cpp, app target), `MockLLMEngine` (tests / DEBUG).
///
/// Engines must serialise requests internally; callers may issue overlapping requests.
public protocol LLMEngine: Sendable {
    /// Loads weights. Idempotent; called before the first `generate`.
    func load() async throws

    /// Streams generated text pieces for `messages` (rendered with the model's chat template).
    /// Cancelling the consuming task stops generation.
    func generate(messages: [ChatMessage], config: GenerationConfig) -> AsyncThrowingStream<String, Error>
}

extension LLMEngine {
    /// Runs a full generation and returns the concatenated text.
    public func complete(messages: [ChatMessage], config: GenerationConfig) async throws -> String {
        var text = ""
        for try await piece in generate(messages: messages, config: config) {
            text += piece
        }
        return text
    }
}

/// Errors shared by engine implementations.
public enum LLMEngineError: Error, Sendable, Equatable, CustomStringConvertible {
    case modelNotFound(String)
    case loadFailed(String)
    case promptTooLong(tokens: Int, limit: Int)
    case decodeFailed(Int32)
    case tokenizationFailed

    public var description: String {
        switch self {
        case .modelNotFound(let path): "Model file not found at \(path)."
        case .loadFailed(let reason): "Could not load model: \(reason)"
        case .promptTooLong(let tokens, let limit): "Prompt is too long (\(tokens) tokens, limit \(limit))."
        case .decodeFailed(let code): "Inference failed (llama_decode returned \(code))."
        case .tokenizationFailed: "Could not tokenize the prompt."
        }
    }
}
