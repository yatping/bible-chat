import Foundation
import Testing
@testable import BibleAICore

/// Runs against `content/build/bible.sqlite` + `rag_embeddings.f16` when the content pipeline has produced them.
@Suite("Real content DB smoke test", .enabled(if: RealContent.available))
struct RealContentSmokeTests {
    @Test func opensAndRetrieves() async throws {
        let db = try ContentDatabase(url: RealContent.database)
        #expect(try db.books().count == 66)
        let semantic = try SemanticSearch.load(database: db, embeddingsURL: RealContent.embeddings)
        #expect(semantic.store.dimension == EmbeddingMath.dimension)

        let rag = try RAGEngine(database: db, semantic: semantic, embedder: nil)
        #expect(rag.referenceParser.reference(in: "Psalm 23")?.bookIndex == 19)
        let forgiveness = try await rag.retrieve("Why did Jesus tell Peter to forgive 70x7?")
        #expect(!forgiveness.isEmpty)
        let matthew = try await rag.retrieve("Matthew 18:22")
        #expect(matthew.first?.reference.hasPrefix("Matthew 18") == true)
        let love = try await rag.retrieve("1 Cor 13:4-7")
        #expect(love.first?.reference.hasPrefix("1 Corinthians 13") == true)
        #expect(try db.verses(book: 43, chapter: 3).count == 36)

    }
}

enum RealContent {
    static let directory = Repo.root.appendingPathComponent("content/build")
    static let database = directory.appendingPathComponent("bible.sqlite")
    static let embeddings = directory.appendingPathComponent("rag_embeddings.f16")
    static var available: Bool {
        FileManager.default.fileExists(atPath: database.path) && FileManager.default.fileExists(atPath: embeddings.path)
    }
}
