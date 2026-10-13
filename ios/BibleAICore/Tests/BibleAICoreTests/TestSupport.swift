import Foundation
@testable import BibleAICore

/// Paths into the repository, located relative to this file so tests read the real `shared/` contracts.
enum Repo {
    static let root: URL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent() // BibleAICoreTests
        .deletingLastPathComponent() // Tests
        .deletingLastPathComponent() // BibleAICore
        .deletingLastPathComponent() // ios
        .deletingLastPathComponent() // repo root
    static let shared = root.appendingPathComponent("shared")
    static let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures")

    static func sharedText(_ name: String) throws -> String {
        try String(contentsOf: shared.appendingPathComponent(name), encoding: .utf8)
    }

    static func sharedJSON<T: Decodable>(_ name: String, as type: T.Type) throws -> T {
        try JSONDecoder().decode(T.self, from: Data(contentsOf: shared.appendingPathComponent("fixtures/\(name)")))
    }
}

/// Deterministic bag-of-words embedder: each content word adds 1 to a hashed bucket in the first 256 dims of a
/// 768-dim raw vector. Prompt scaffolding words are ignored so only content drives similarity.
struct FakeEmbedder: EmbeddingEngine {
    static let ignored: Set<String> = ["task", "search", "result", "query", "title", "text", "none"]

    func rawEmbeddings(for prompts: [String]) async throws -> [[Float]] {
        prompts.map(Self.vector)
    }

    static func vector(_ text: String) -> [Float] {
        var v = [Float](repeating: 0, count: 768)
        for token in FTSQueryBuilder.tokens(in: text.lowercased())
        where token.count > 2 && !ignored.contains(token) && !FTSQueryBuilder.sharedStopwords.contains(token) {
            v[Int(fnv1a(stem(token)) % 256)] += 1
        }
        v[300] = 5 // beyond the Matryoshka cut: must be dropped by truncation
        return v
    }

    /// Crude stemming so "forgive"/"forgiven" collide.
    static func stem(_ token: String) -> String {
        String(token.prefix(6))
    }

    static func fnv1a(_ s: String) -> UInt64 {
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in s.utf8 { hash = (hash ^ UInt64(byte)) &* 0x100000001b3 }
        return hash
    }
}

/// A small but real content DB (shared/schema_content.sql + WEB pericopes) with a matching embeddings file.
struct TestContent {
    let database: ContentDatabase
    let semantic: SemanticSearch
    let embedder = FakeEmbedder()
    let directory: URL

    struct Fixture: Decodable {
        struct Pericope: Decodable {
            struct V: Decodable { let chapter: Int; let verse: Int; let text: String }
            let id: Int64; let title: String; let reference: String; let book_index: Int
            let chapter: Int; let verse_start: Int; let chapter_end: Int; let verse_end: Int
            let verses: [V]
        }
        let chunks: [Pericope]
    }

    static let books: [(Int, String, String, String, String, Int)] = [
        (1, "Gen", "GEN", "Genesis", "OT", 50), (19, "Ps", "PSA", "Psalms", "OT", 150),
        (40, "Matt", "MAT", "Matthew", "NT", 28), (43, "John", "JHN", "John", "NT", 21),
        (45, "Rom", "ROM", "Romans", "NT", 16), (46, "1Cor", "1CO", "1 Corinthians", "NT", 16),
        (65, "Jude", "JUD", "Jude", "NT", 1),
    ]

    static let aliases: [String: Int] = [
        "genesis": 1, "gen": 1, "gn": 1, "ge": 1, "psalms": 19, "psalm": 19, "ps": 19,
        "matthew": 40, "matt": 40, "mt": 40, "john": 43, "jn": 43, "jhn": 43,
        "romans": 45, "rom": 45, "ro": 45, "1 corinthians": 46, "1 cor": 46, "1cor": 46, "first corinthians": 46,
        "1co": 46, "jude": 65, "jud": 65,
    ]

    static let commentary = Chunk(
        id: 100, sourceType: "commentary", title: "Forgiveness without limit", reference: "Matthew 18 commentary",
        text: "Peter's offer of seven times seemed generous, but Jesus answers seventy times seven. "
            + "The expression is generally understood as an unlimited number: forgiveness is not counted.",
        tokenEstimate: 0)

    static func make() throws -> TestContent {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("BibleAICoreTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let db = try SQLiteDatabase(path: directory.appendingPathComponent("bible.sqlite").path)
        try db.execute(Repo.sharedText("schema_content.sql"))

        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: Repo.fixtures.appendingPathComponent("web_pericopes.json")))
        var chunks: [Chunk] = []
        try db.transaction {
            for b in books {
                try db.run("INSERT INTO books VALUES (?, ?, ?, ?, ?, ?)", [b.0, b.1, b.2, b.3, b.4, b.5])
            }
            for (alias, book) in aliases { try db.run("INSERT INTO book_aliases VALUES (?, ?)", [alias, book]) }
            for p in fixture.chunks {
                for v in p.verses {
                    try db.run("INSERT INTO verses VALUES ('WEB', ?, ?, ?, ?)", [p.book_index, v.chapter, v.verse, v.text])
                }
                let text = p.verses.map(\.text).joined(separator: " ")
                chunks.append(Chunk(id: p.id, sourceType: "scripture", title: p.title, reference: p.reference,
                                    translation: "WEB", bookIndex: p.book_index, chapter: p.chapter,
                                    verseStart: p.verse_start, chapterEnd: p.chapter_end, verseEnd: p.verse_end,
                                    text: text, tokenEstimate: TokenBudget.estimate(text)))
            }
            var commentary = commentary
            commentary.tokenEstimate = TokenBudget.estimate(commentary.text)
            chunks.append(commentary)
            for c in chunks {
                try db.run("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           [c.id, c.sourceType, c.title, c.reference, c.translation, c.bookIndex, c.chapter,
                            c.verseStart, c.chapterEnd, c.verseEnd, c.text, c.tokenEstimate])
            }
            try db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
            for (row, c) in chunks.enumerated() {
                try db.run("INSERT INTO embedding_rows VALUES (?, ?)", [row, c.id])
            }
        }

        // Document vectors exactly as the content builder makes them: `title: {reference} | text: {text}`.
        let vectors = chunks.map { EmbeddingMath.truncateAndNormalize(FakeEmbedder.vector(EmbeddingPrompt.document(title: $0.reference, text: $0.text))) }
        let embeddingsURL = directory.appendingPathComponent("rag_embeddings.f16")
        try VectorStore.float16Data(from: vectors).write(to: embeddingsURL)

        let meta = ["schema_version": "1", "translation": "WEB", "embedding_model": "fake", "embedding_dim": "256",
                    "embedding_count": "\(chunks.count)", "embedding_file": "rag_embeddings.f16"]
        for (k, v) in meta { try db.run("INSERT INTO content_meta VALUES (?, ?)", [k, v]) }

        let database = try ContentDatabase(database: db)
        return TestContent(database: database, semantic: try SemanticSearch.load(database: database, embeddingsURL: embeddingsURL),
                           directory: directory)
    }

    func ragEngine(configuration: RAGConfiguration = RAGConfiguration()) throws -> RAGEngine {
        try RAGEngine(database: database, semantic: semantic, embedder: embedder, configuration: configuration)
    }
}
