#if canImport(llama)
import BibleAICore
import Foundation
import OSLog
import llama

/// `EmbeddingEngine` running EmbeddingGemma 300M (GGUF) through llama.cpp.
///
/// Pooling comes from the GGUF metadata (mean for EmbeddingGemma). The GGUF must be converted with
/// `--sentence-transformers-dense-modules` so the Dense projections after pooling are included (CONTRACTS §2).
/// This engine returns raw pooled vectors; `EmbeddingEngine.embed` adds the §2 prompts, truncation and normalisation.
actor LlamaEmbeddingEngine: EmbeddingEngine {
    private let modelURL: URL
    private let contextLength: UInt32
    private let queue = DispatchSerialQueue(label: "bibleai.embedder", qos: .userInitiated)
    private var handle: LlamaHandle?
    private static let log = Logger(subsystem: "BibleAI", category: "Embedding")

    nonisolated var unownedExecutor: UnownedSerialExecutor { queue.asUnownedSerialExecutor() }

    init(modelURL: URL, contextLength: UInt32 = 2048) {
        self.modelURL = modelURL
        self.contextLength = contextLength
    }

    func load() throws {
        guard handle == nil else { return }
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw LLMEngineError.modelNotFound(modelURL.path)
        }
        LlamaBackend.initialize()
        var modelParams = llama_model_default_params()
        #if targetEnvironment(simulator)
        modelParams.n_gpu_layers = 0
        #else
        modelParams.n_gpu_layers = 999
        #endif
        guard let model = llama_model_load_from_file(modelURL.path, modelParams) else {
            throw LLMEngineError.loadFailed("embedding model failed to load")
        }
        var params = llama_context_default_params()
        params.embeddings = true
        params.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED // use the GGUF's pooling (mean)
        params.n_ctx = contextLength
        // Non-causal attention: a whole sequence must fit in one micro-batch.
        params.n_batch = contextLength
        params.n_ubatch = contextLength
        params.n_seq_max = 1
        guard let context = llama_init_from_model(model, params) else {
            llama_model_free(model)
            throw LLMEngineError.loadFailed("embedding context failed to initialise")
        }
        handle = LlamaHandle(model: model, context: context)
    }

    func rawEmbeddings(for prompts: [String]) async throws -> [[Float]] {
        try load()
        guard let handle else { throw LLMEngineError.loadFailed("embedding model not loaded") }
        let model = handle.model, context = handle.context
        let vocab = llama_model_get_vocab(model)
        let outputDim = Int(llama_model_n_embd_out(model) > 0 ? llama_model_n_embd_out(model) : llama_model_n_embd(model))
        let useEncoder = llama_model_has_encoder(model) && !llama_model_has_decoder(model)

        var results: [[Float]] = []
        results.reserveCapacity(prompts.count)
        for prompt in prompts {
            try Task.checkCancellation()
            var tokens = tokenize(prompt, vocab: vocab)
            if tokens.count > Int(contextLength) { tokens = Array(tokens.prefix(Int(contextLength))) }
            guard !tokens.isEmpty else { throw LLMEngineError.tokenizationFailed }

            llama_memory_clear(llama_get_memory(context), true)
            var batch = llama_batch_init(Int32(tokens.count), 0, 1)
            defer { llama_batch_free(batch) }
            for (i, token) in tokens.enumerated() {
                batch.token[i] = token
                batch.pos[i] = Int32(i)
                batch.n_seq_id[i] = 1
                batch.seq_id[i]![0] = 0
                batch.logits[i] = 1
            }
            batch.n_tokens = Int32(tokens.count)

            let rc = useEncoder ? llama_encode(context, batch) : llama_decode(context, batch)
            guard rc == 0 else { throw LLMEngineError.decodeFailed(rc) }
            guard let pooled = llama_get_embeddings_seq(context, 0) else {
                throw LLMEngineError.loadFailed("no pooled embedding (pooling type NONE?)")
            }
            results.append(Array(UnsafeBufferPointer(start: pooled, count: outputDim)))
        }
        return results
    }

    private func tokenize(_ text: String, vocab: OpaquePointer?) -> [llama_token] {
        let utf8Count = Int32(text.utf8.count)
        var tokens = [llama_token](repeating: 0, count: Int(utf8Count) + 8)
        var count = llama_tokenize(vocab, text, utf8Count, &tokens, Int32(tokens.count), true, false)
        if count < 0, count != Int32.min {
            tokens = [llama_token](repeating: 0, count: Int(-count))
            count = llama_tokenize(vocab, text, utf8Count, &tokens, Int32(tokens.count), true, false)
        }
        return Array(tokens.prefix(Int(max(0, count))))
    }
}
#endif
