#if canImport(llama)
import BibleAICore
import Foundation
import OSLog
import llama

/// `LLMEngine` running the fine-tuned Gemma 4 E2B GGUF through llama.cpp's C API (Metal offload).
///
/// The actor runs on its own serial dispatch queue, so blocking `llama_decode` calls never occupy the
/// Swift concurrency thread pool or the main thread. Requests are serialised by the actor; each request
/// clears the KV cache, so there is no state carried between requests.
actor GemmaEngine: LLMEngine {
    struct Options: Sendable {
        var contextLength: UInt32 = UInt32(TokenBudget.default.contextLength)
        var batchSize: UInt32 = 512
        /// All layers on the GPU on device; the simulator has no usable Metal backend for this.
        var gpuLayers: Int32 = {
            #if targetEnvironment(simulator)
            0
            #else
            999
            #endif
        }()
        var threads: Int32 = Int32(max(1, min(4, ProcessInfo.processInfo.activeProcessorCount - 2)))
    }

    private let modelURL: URL
    private let options: Options
    private let queue = DispatchSerialQueue(label: "bibleai.llm", qos: .userInitiated)
    private var handle: LlamaHandle?
    private static let log = Logger(subsystem: "BibleAI", category: "GemmaEngine")

    nonisolated var unownedExecutor: UnownedSerialExecutor { queue.asUnownedSerialExecutor() }

    init(modelURL: URL, options: Options = Options()) {
        self.modelURL = modelURL
        self.options = options
    }

    func load() async throws {
        guard handle == nil else { return }
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw LLMEngineError.modelNotFound(modelURL.path)
        }
        LlamaBackend.initialize()

        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = options.gpuLayers
        guard let model = llama_model_load_from_file(modelURL.path, modelParams) else {
            throw LLMEngineError.loadFailed("llama_model_load_from_file returned NULL")
        }

        var contextParams = llama_context_default_params()
        contextParams.n_ctx = options.contextLength
        contextParams.n_batch = options.batchSize
        contextParams.n_ubatch = options.batchSize
        contextParams.n_threads = options.threads
        contextParams.n_threads_batch = options.threads
        contextParams.n_seq_max = 1
        guard let context = llama_init_from_model(model, contextParams) else {
            llama_model_free(model)
            throw LLMEngineError.loadFailed("llama_init_from_model returned NULL")
        }
        handle = LlamaHandle(model: model, context: context)
        Self.log.info("loaded \(self.modelURL.lastPathComponent, privacy: .public), n_ctx \(llama_n_ctx(context))")
    }

    nonisolated func generate(messages: [ChatMessage], config: GenerationConfig) -> AsyncThrowingStream<String, Error> {
        let (stream, continuation) = AsyncThrowingStream.makeStream(of: String.self)
        let task = Task {
            do {
                try await self.run(messages: messages, config: config) { continuation.yield($0) }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { _ in task.cancel() }
        return stream
    }

    // MARK: - Generation

    private func run(messages: [ChatMessage], config: GenerationConfig, emit: @Sendable (String) -> Void) async throws {
        try await load()
        guard let handle else { throw LLMEngineError.loadFailed("model not loaded") }
        let model = handle.model, context = handle.context
        guard let vocab = llama_model_get_vocab(model) else { throw LLMEngineError.loadFailed("model has no vocabulary") }

        let prompt = render(messages, model: model)
        var tokens = try tokenize(prompt, vocab: vocab, addSpecial: !prompt.hasPrefix(ChatTemplate.bosToken))
        let contextLength = Int(llama_n_ctx(context))
        let reserve = min(config.maxTokens, 256)
        guard tokens.count + reserve <= contextLength else {
            throw LLMEngineError.promptTooLong(tokens: tokens.count, limit: contextLength - reserve)
        }
        let maxNewTokens = min(config.maxTokens, contextLength - tokens.count)

        // Fresh state for every request.
        llama_memory_clear(llama_get_memory(context), true)

        // Prompt prefill in n_batch chunks; logits are produced for the last token only.
        let batchSize = Int(llama_n_batch(context))
        var start = 0
        while start < tokens.count {
            try Task.checkCancellation()
            let count = min(batchSize, tokens.count - start)
            let rc = tokens.withUnsafeMutableBufferPointer { buffer in
                llama_decode(context, llama_batch_get_one(buffer.baseAddress! + start, Int32(count)))
            }
            guard rc == 0 else { throw LLMEngineError.decodeFailed(rc) }
            start += count
        }

        let sampler = makeSampler(config: config, vocab: vocab)
        defer { llama_sampler_free(sampler) }

        var decoder = UTF8StreamDecoder()
        var thoughts = ThoughtChannelFilter()
        var text = ""
        var pieceBuffer = [CChar](repeating: 0, count: 256)
        for _ in 0..<maxNewTokens {
            if Task.isCancelled { break }
            var token = llama_sampler_sample(sampler, context, -1) // also accepts the token
            if llama_vocab_is_eog(vocab, token) { break }

            var length = llama_token_to_piece(vocab, token, &pieceBuffer, Int32(pieceBuffer.count), 0, true)
            if length < 0 {
                pieceBuffer = [CChar](repeating: 0, count: Int(-length))
                length = llama_token_to_piece(vocab, token, &pieceBuffer, Int32(pieceBuffer.count), 0, true)
            }
            // Special tokens are rendered (last argument `true`) so thought-channel markers reach the filter.
            let decoded = decoder.append(pieceBuffer.prefix(Int(max(0, length))).map { UInt8(bitPattern: $0) })
            let piece = thoughts.feed(decoded)
            if !piece.isEmpty {
                text += piece
                if ChatTemplate.stopStrings.contains(where: text.contains) { break }
                emit(piece)
            }

            let rc = llama_decode(context, llama_batch_get_one(&token, 1))
            guard rc == 0 else { throw LLMEngineError.decodeFailed(rc) }
        }
        let rest = thoughts.feed(decoder.finish()) + thoughts.finish()
        if !rest.isEmpty { emit(rest) }
    }

    /// Renders with the GGUF's embedded template when llama.cpp recognises it, else the Gemma 4 Swift fallback.
    private func render(_ messages: [ChatMessage], model: OpaquePointer) -> String {
        if let template = llama_model_chat_template(model, nil),
           let rendered = Self.applyTemplate(template, messages) {
            return rendered
        }
        return ChatTemplate.gemma4(messages, includeBOS: false, addGenerationPrompt: true)
    }

    private static func applyTemplate(_ template: UnsafePointer<CChar>, _ messages: [ChatMessage]) -> String? {
        let roles = messages.map { strdup($0.role.rawValue) }
        let contents = messages.map { strdup($0.content) }
        defer { (roles + contents).forEach { free($0) } }
        let chat = zip(roles, contents).map { llama_chat_message(role: $0, content: $1) }
        var capacity = max(1024, messages.reduce(0) { $0 + $1.content.utf8.count } * 2 + 256)
        for _ in 0..<2 {
            var buffer = [CChar](repeating: 0, count: capacity)
            let length = llama_chat_apply_template(template, chat, chat.count, true, &buffer, Int32(capacity))
            if length < 0 { return nil } // template not supported by llama.cpp's built-in list
            if Int(length) <= capacity {
                return String(decoding: buffer.prefix(Int(length)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
            }
            capacity = Int(length) + 1
        }
        return nil
    }

    private func tokenize(_ text: String, vocab: OpaquePointer, addSpecial: Bool) throws -> [llama_token] {
        let utf8Count = Int32(text.utf8.count)
        var tokens = [llama_token](repeating: 0, count: Int(utf8Count) + 16)
        var count = llama_tokenize(vocab, text, utf8Count, &tokens, Int32(tokens.count), addSpecial, true)
        if count < 0, count != Int32.min {
            tokens = [llama_token](repeating: 0, count: Int(-count))
            count = llama_tokenize(vocab, text, utf8Count, &tokens, Int32(tokens.count), addSpecial, true)
        }
        guard count > 0 else { throw LLMEngineError.tokenizationFailed }
        return Array(tokens.prefix(Int(count)))
    }

    /// penalties → top-k → top-p → temperature → dist (or greedy at temperature 0).
    private func makeSampler(config: GenerationConfig, vocab: OpaquePointer) -> UnsafeMutablePointer<llama_sampler> {
        let chain = llama_sampler_chain_init(llama_sampler_chain_default_params())!
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(
            llama_vocab_n_tokens(vocab), Int32(config.repeatLastN), config.repeatPenalty, 0, 0))
        if config.temperature <= 0 {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy())
        } else {
            llama_sampler_chain_add(chain, llama_sampler_init_top_k(Int32(config.topK)))
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(config.topP, 1))
            llama_sampler_chain_add(chain, llama_sampler_init_temp(config.temperature))
            llama_sampler_chain_add(chain, llama_sampler_init_dist(config.seed))
        }
        return chain
    }
}

/// Owns a llama model + context and frees them when released.
final class LlamaHandle: @unchecked Sendable {
    let model: OpaquePointer
    let context: OpaquePointer

    init(model: OpaquePointer, context: OpaquePointer) {
        self.model = model
        self.context = context
    }

    deinit {
        llama_free(context)
        llama_model_free(model)
    }
}

/// One-time `llama_backend_init`.
enum LlamaBackend {
    private static let once: Void = {
        llama_backend_init()
        llama_log_set({ level, text, _ in
            #if DEBUG
            if level.rawValue >= GGML_LOG_LEVEL_WARN.rawValue, let text {
                print("[llama] \(String(cString: text))", terminator: "")
            }
            #endif
        }, nil)
    }()

    static func initialize() { _ = once }
}
#endif
