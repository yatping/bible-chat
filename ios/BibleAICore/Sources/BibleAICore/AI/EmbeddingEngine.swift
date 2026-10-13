import Foundation
import Accelerate

/// What kind of text is being embedded; selects the EmbeddingGemma prompt (CONTRACTS §2).
public enum EmbeddingKind: Sendable, Hashable {
    /// `task: search result | query: {text}`
    case query
    /// `title: {title} | text: {text}`; a nil/empty title becomes `none` (memories use `none`).
    case document(title: String? = nil)
}

/// A text embedder. Implementations only provide raw pooled vectors for already formatted prompts;
/// `embed(_:kind:)` adds the §2 prompts and the Matryoshka truncation + L2 normalisation.
public protocol EmbeddingEngine: Sendable {
    /// Raw (pooled, untruncated) embeddings for prompts that already contain the §2 prefix.
    func rawEmbeddings(for prompts: [String]) async throws -> [[Float]]
}

extension EmbeddingEngine {
    /// Embeds `texts` and returns vectors truncated to `EmbeddingMath.dimension` and L2-normalised.
    public func embed(_ texts: [String], kind: EmbeddingKind) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }
        let prompts = texts.map { EmbeddingPrompt.format($0, kind: kind) }
        return try await rawEmbeddings(for: prompts).map { EmbeddingMath.truncateAndNormalize($0) }
    }

    /// Embeds a single text.
    public func embed(_ text: String, kind: EmbeddingKind) async throws -> [Float] {
        try await embed([text], kind: kind)[0]
    }
}

/// EmbeddingGemma prompt templates (CONTRACTS §2). llama.cpp does not add them, so the app does.
public enum EmbeddingPrompt {
    public static func query(_ text: String) -> String {
        "task: search result | query: \(text)"
    }

    public static func document(title: String?, text: String) -> String {
        let title = title?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return "title: \(title.isEmpty ? "none" : title) | text: \(text)"
    }

    public static func format(_ text: String, kind: EmbeddingKind) -> String {
        switch kind {
        case .query: query(text)
        case .document(let title): document(title: title, text: text)
        }
    }
}

/// Vector helpers shared by RAG and memory code.
public enum EmbeddingMath {
    /// Matryoshka dimension kept on device (CONTRACTS §2).
    public static let dimension = 256

    /// Keeps the first `dimension` values and L2-normalises them.
    public static func truncateAndNormalize(_ vector: [Float], dimension: Int = EmbeddingMath.dimension) -> [Float] {
        var head = Array(vector.prefix(dimension))
        if head.count < dimension { head += [Float](repeating: 0, count: dimension - head.count) }
        let norm = sqrt(vDSP.sumOfSquares(head))
        guard norm > 0, norm.isFinite else { return head }
        return vDSP.divide(head, norm)
    }

    /// Dot product (= cosine similarity for normalised vectors). Mismatched lengths compare the common prefix.
    public static func dot(_ a: [Float], _ b: [Float]) -> Float {
        let n = min(a.count, b.count)
        guard n > 0 else { return 0 }
        if a.count == b.count { return vDSP.dot(a, b) }
        return vDSP.dot(Array(a.prefix(n)), Array(b.prefix(n)))
    }

    /// Little-endian Float32 blob (memory vectors in `user.sqlite`).
    public static func blob(from vector: [Float]) -> Data {
        var data = Data(capacity: vector.count * 4)
        for value in vector {
            withUnsafeBytes(of: value.bitPattern.littleEndian) { data.append(contentsOf: $0) }
        }
        return data
    }

    /// Decodes a little-endian Float32 blob.
    public static func vector(from blob: Data) -> [Float] {
        let count = blob.count / 4
        return blob.withUnsafeBytes { raw in
            (0..<count).map { i in
                Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: i * 4, as: UInt32.self)))
            }
        }
    }
}
