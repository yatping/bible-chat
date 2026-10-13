import Foundation
import OSLog

/// Retrieval settings (CONTRACTS §3).
public struct RAGConfiguration: Sendable, Hashable {
    public var topK = 6
    public var sourceTokenBudget = TokenBudget.default.sources
    public var semanticLimit = 30
    public var lexicalLimit = 30

    public init() {}
}

/// Hybrid retrieval: explicit references (R), semantic (V) and BM25 (L) fused with weighted RRF,
/// then trimmed to the sources token budget.
public final class RAGEngine: Sendable {
    public let database: ContentDatabase
    public let configuration: RAGConfiguration
    private let semantic: SemanticSearch?
    private let embedder: (any EmbeddingEngine)?
    private let lexical: FTSRetriever
    private let parser: ReferenceParser
    private static let log = Logger(subsystem: "BibleAICore", category: "RAG")

    /// - Parameters:
    ///   - semantic: nil disables dense retrieval (e.g. embeddings file missing).
    ///   - embedder: used when `retrieve` is not given a precomputed query vector.
    public init(database: ContentDatabase, semantic: SemanticSearch?, embedder: (any EmbeddingEngine)?,
                configuration: RAGConfiguration = RAGConfiguration()) throws {
        self.database = database
        self.semantic = semantic
        self.embedder = embedder
        self.configuration = configuration
        lexical = FTSRetriever(database: database)
        let chapterCounts = Dictionary(try database.books().map { ($0.index, $0.chapters) }, uniquingKeysWith: { a, _ in a })
        parser = ReferenceParser(aliases: try database.aliases(), chapterCounts: chapterCounts)
    }

    /// The reference parser built from the content DB's aliases.
    public var referenceParser: ReferenceParser { parser }

    /// Retrieves, fuses and numbers sources for `query`.
    /// - Parameter queryVector: the already embedded query (`EmbeddingKind.query`), to share one embedding with memory retrieval.
    public func retrieve(_ query: String, queryVector: [Float]? = nil) async throws -> [RetrievedSource] {
        let ids = try await rankedChunkIDs(query, queryVector: queryVector)
        let chunks = try database.chunks(ids: ids)
        return RetrievedSource.numbered(Self.trim(chunks, toTokens: configuration.sourceTokenBudget))
    }

    /// Fused chunk ids, best first (top `configuration.topK`).
    public func rankedChunkIDs(_ query: String, queryVector: [Float]? = nil) async throws -> [Int64] {
        async let dense = semanticIDs(query, queryVector: queryVector)
        let referenceIDs = try referenceIDs(query)
        let lexicalIDs = try lexical.search(query, limit: configuration.lexicalLimit)
        let semanticIDs = await dense
        return HybridRanker.fuse([
            .init(ids: referenceIDs, weight: HybridRanker.referenceWeight),
            .init(ids: semanticIDs, weight: HybridRanker.semanticWeight),
            .init(ids: lexicalIDs, weight: HybridRanker.lexicalWeight),
        ], topK: configuration.topK)
    }

    /// Walks chunks in rank order and stops at the first that would exceed `budget`; always keeps the first.
    public static func trim(_ chunks: [Chunk], toTokens budget: Int) -> [Chunk] {
        var used = 0
        var kept: [Chunk] = []
        for chunk in chunks {
            if !kept.isEmpty, used + chunk.tokenEstimate > budget { break }
            used += chunk.tokenEstimate
            kept.append(chunk)
        }
        return kept
    }

    private func referenceIDs(_ query: String) throws -> [Int64] {
        var ids: [Int64] = []
        for reference in parser.references(in: query) {
            for id in try database.scriptureChunkIDs(overlapping: reference)
            where !ids.contains(id) {
                ids.append(id)
            }
        }
        return ids
    }

    private func semanticIDs(_ query: String, queryVector: [Float]?) async -> [Int64] {
        guard let semantic else { return [] }
        do {
            var vector = queryVector
            if vector == nil, let embedder { vector = try await embedder.embed(query, kind: .query) }
            guard let vector else { return [] }
            return try semantic.search(queryVector: vector, limit: configuration.semanticLimit)
        } catch {
            Self.log.error("semantic search failed: \(String(describing: error), privacy: .public)")
            return []
        }
    }
}
