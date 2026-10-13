import BibleAICore
import Foundation

/// Creates the inference engines. Falls back to the mock engine when llama.xcframework is not linked.
enum EngineFactory {
    /// True when the app was built with llama.cpp.
    static var hasLlama: Bool {
        #if canImport(llama)
        true
        #else
        false
        #endif
    }

    static func makeLLM(modelURL: URL?) -> any LLMEngine {
        #if canImport(llama)
        if let modelURL { return GemmaEngine(modelURL: modelURL) }
        #endif
        return MockLLMEngine.demo()
    }

    static func makeEmbedder(modelURL: URL?) -> (any EmbeddingEngine)? {
        #if canImport(llama)
        if let modelURL { return LlamaEmbeddingEngine(modelURL: modelURL) }
        #endif
        return nil
    }
}
