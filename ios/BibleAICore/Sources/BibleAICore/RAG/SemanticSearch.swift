import Foundation

/// Dense retrieval: query vector · all document rows (CONTRACTS §3 step 2).
public struct SemanticSearch: Sendable {
    public let store: VectorStore
    /// `chunk_id` of each row of the vector store.
    public let rowChunkIDs: [Int64]

    public init(store: VectorStore, rowChunkIDs: [Int64]) {
        precondition(store.count == rowChunkIDs.count, "embedding_rows must cover every vector row")
        self.store = store
        self.rowChunkIDs = rowChunkIDs
    }

    /// Opens `rag_embeddings.f16` using the shape stored in `content_meta`.
    public static func load(database: ContentDatabase, embeddingsURL: URL) throws -> SemanticSearch {
        let shape = try database.embeddingShape()
        let store = try VectorStore(url: embeddingsURL, count: shape.count, dimension: shape.dim)
        return SemanticSearch(store: store, rowChunkIDs: try database.embeddingRowChunkIDs())
    }

    /// Chunk ids of the `limit` most similar rows for an already embedded (truncated, normalised) query.
    public func search(queryVector: [Float], limit: Int = 30) throws -> [Int64] {
        try store.topK(queryVector, k: limit).map { rowChunkIDs[$0.row] }
    }
}
