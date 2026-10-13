import Foundation

/// Read-only access to the bundled `bible.sqlite` (schema: `shared/schema_content.sql`).
public final class ContentDatabase: Sendable {
    public enum Error: Swift.Error, CustomStringConvertible {
        case missingMeta(String)
        case unsupportedSchema(String)

        public var description: String {
            switch self {
            case .missingMeta(let key): "bible.sqlite is missing content_meta key '\(key)'."
            case .unsupportedSchema(let version): "bible.sqlite has unsupported schema_version \(version)."
            }
        }
    }

    let db: SQLiteDatabase
    /// `content_meta` key/value pairs.
    public let meta: [String: String]

    /// Opens the bundled file read-only and immutable.
    public convenience init(url: URL) throws {
        try self.init(database: SQLiteDatabase(path: url.path, readOnly: true, immutable: true))
    }

    /// Wraps an existing connection (tests build the content DB in memory).
    public init(database: SQLiteDatabase) throws {
        db = database
        var meta: [String: String] = [:]
        for row in try db.query("SELECT key, value FROM content_meta") {
            if let key = row.string("key"), let value = row.string("value") { meta[key] = value }
        }
        if let version = meta["schema_version"], version != "1" { throw Error.unsupportedSchema(version) }
        self.meta = meta
    }

    /// Default scripture translation of this build (`content_meta.translation`).
    public var translation: String { meta["translation"] ?? "WEB" }

    /// `(count, dim)` of `rag_embeddings.f16`.
    public func embeddingShape() throws -> (count: Int, dim: Int) {
        guard let count = meta["embedding_count"].flatMap(Int.init) else { throw Error.missingMeta("embedding_count") }
        guard let dim = meta["embedding_dim"].flatMap(Int.init) else { throw Error.missingMeta("embedding_dim") }
        return (count, dim)
    }

    // MARK: Books & verses

    public func books() throws -> [Book] {
        try db.query("SELECT book_index, osis, name, testament, chapters FROM books ORDER BY book_index").map {
            Book(index: $0.int("book_index") ?? 0, osis: $0.string("osis") ?? "", name: $0.string("name") ?? "",
                 testament: $0.string("testament") ?? "", chapters: $0.int("chapters") ?? 0)
        }
    }

    /// Normalised alias → book_index.
    public func aliases() throws -> [String: Int] {
        var result: [String: Int] = [:]
        for row in try db.query("SELECT alias, book_index FROM book_aliases") {
            if let alias = row.string("alias"), let index = row.int("book_index") { result[alias] = index }
        }
        return result
    }

    /// All verses of one chapter in order.
    public func verses(book: Int, chapter: Int, translation: String? = nil) throws -> [Verse] {
        try db.query(
            "SELECT book_index, chapter, verse, text FROM verses WHERE translation = ? AND book_index = ? AND chapter = ? ORDER BY verse",
            [translation ?? self.translation, book, chapter]
        ).map(Self.verse)
    }

    /// Verses in an inclusive `(chapter, verse)` span of one book.
    public func verses(in reference: ScriptureReference, translation: String? = nil) throws -> [Verse] {
        try db.query(
            """
            SELECT book_index, chapter, verse, text FROM verses
            WHERE translation = ? AND book_index = ? AND chapter * 1000 + verse BETWEEN ? AND ?
            ORDER BY chapter, verse
            """,
            [translation ?? self.translation, reference.bookIndex, reference.startKey, reference.endKey]
        ).map(Self.verse)
    }

    // MARK: Chunks

    /// Chunks by id, returned in the order of `ids` (unknown ids are skipped).
    public func chunks(ids: [Int64]) throws -> [Chunk] {
        guard !ids.isEmpty else { return [] }
        let placeholders = Array(repeating: "?", count: ids.count).joined(separator: ",")
        let rows = try db.query("SELECT * FROM chunks WHERE id IN (\(placeholders))", ids)
        let byID = Dictionary(rows.map(Self.chunk).map { ($0.id, $0) }, uniquingKeysWith: { a, _ in a })
        return ids.compactMap { byID[$0] }
    }

    /// Scripture chunks overlapping `reference`, ordered by (chapter, verse_start) — no cap (CONTRACTS §3).
    public func scriptureChunkIDs(overlapping reference: ScriptureReference) throws -> [Int64] {
        try db.query(
            """
            SELECT id FROM chunks
            WHERE source_type = 'scripture' AND book_index = ?
              AND chapter * 1000 + verse_start <= ?
              AND COALESCE(chapter_end, chapter) * 1000 + verse_end >= ?
            ORDER BY book_index, chapter, verse_start, id
            """,
            [reference.bookIndex, reference.endKey, reference.startKey]
        ).compactMap { $0.int64("id") }
    }

    /// `chunk_id` for every row of `rag_embeddings.f16` (index = row).
    public func embeddingRowChunkIDs() throws -> [Int64] {
        try db.query("SELECT row, chunk_id FROM embedding_rows ORDER BY row").compactMap { $0.int64("chunk_id") }
    }

    /// BM25 search exactly as in CONTRACTS §3. `match` is an FTS query from `FTSQueryBuilder`.
    public func ftsSearch(match: String, limit: Int = 30) throws -> [Int64] {
        try db.query(
            "SELECT rowid, bm25(chunks_fts, 2.0, 3.0, 1.0) AS score FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score, rowid LIMIT ?",
            [match, limit]
        ).compactMap { $0.int64("rowid") }
    }

    private static func verse(_ row: SQLRow) -> Verse {
        Verse(bookIndex: row.int("book_index") ?? 0, chapter: row.int("chapter") ?? 0,
              number: row.int("verse") ?? 0, text: row.string("text") ?? "")
    }

    private static func chunk(_ row: SQLRow) -> Chunk {
        Chunk(id: row.int64("id") ?? 0, sourceType: row.string("source_type") ?? "", title: row.string("title") ?? "",
              reference: row.string("reference") ?? "", translation: row.string("translation"),
              bookIndex: row.int("book_index"), chapter: row.int("chapter"), verseStart: row.int("verse_start"),
              chapterEnd: row.int("chapter_end"), verseEnd: row.int("verse_end"), text: row.string("text") ?? "",
              tokenEstimate: row.int("token_estimate") ?? TokenBudget.estimate(row.string("text") ?? ""))
    }
}
