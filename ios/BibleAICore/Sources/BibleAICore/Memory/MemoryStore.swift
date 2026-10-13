import Foundation

/// CRUD for `memories` / `memory_embeddings` in `user.sqlite`.
public struct MemoryStore: Sendable {
    public let database: UserDatabase
    var db: SQLiteDatabase { database.db }

    public init(database: UserDatabase) {
        self.database = database
    }

    private static let select = """
        SELECT m.*, e.vector AS vector FROM memories m
        LEFT JOIN memory_embeddings e ON e.memory_id = m.id
        """

    /// Active (not superseded) memories, newest first.
    public func activeMemories() throws -> [MemoryRecord] {
        try db.query(Self.select + " WHERE m.superseded = 0 ORDER BY m.updated_at DESC").map(Self.record)
    }

    /// Superseded memories (history for the settings screen), newest first.
    public func supersededMemories() throws -> [MemoryRecord] {
        try db.query(Self.select + " WHERE m.superseded = 1 ORDER BY m.updated_at DESC").map(Self.record)
    }

    public func memory(id: String) throws -> MemoryRecord? {
        try db.queryOne(Self.select + " WHERE m.id = ?", [id]).map(Self.record)
    }

    /// The active memory holding `key`, if any.
    public func activeMemory(key: MemoryKey) throws -> MemoryRecord? {
        try db.queryOne(Self.select + " WHERE m.memory_key = ? AND m.superseded = 0", [key.rawValue]).map(Self.record)
    }

    /// Inserts a memory and its embedding.
    public func insert(_ memory: MemoryRecord) throws {
        try db.transaction {
            try db.run("""
                INSERT INTO memories (id, type, memory_key, content, importance, confidence, source_message_id,
                                      created_at, updated_at, last_accessed_at, superseded, superseded_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [memory.id, memory.type.rawValue, memory.key?.rawValue, memory.content, memory.importance,
                      memory.confidence, memory.sourceMessageID, memory.createdAt.unixSeconds,
                      memory.updatedAt.unixSeconds, memory.lastAccessedAt?.unixSeconds, memory.superseded,
                      memory.supersededBy])
            if let embedding = memory.embedding { try setEmbedding(embedding, for: memory.id) }
        }
    }

    /// Replaces a keyed memory: the old row is marked superseded and points at the new one (one transaction).
    public func supersede(_ oldID: String, with new: MemoryRecord) throws {
        try db.transaction {
            // Order matters: free the unique active-key slot, insert, then link (superseded_by is a foreign key).
            try db.run("UPDATE memories SET superseded = 1 WHERE id = ?", [oldID])
            try insert(new)
            try db.run("UPDATE memories SET superseded_by = ? WHERE id = ?", [new.id, oldID])
        }
    }

    /// Bumps `updated_at` (same value re-stated) and keeps the higher importance.
    public func touch(_ id: String, importance: Double? = nil, at date: Date = .now) throws {
        try db.run("UPDATE memories SET updated_at = ?, importance = MAX(importance, COALESCE(?, importance)) WHERE id = ?",
                   [date.unixSeconds, importance, id])
    }

    /// Rewrites a memory's content (near-duplicate merge).
    public func updateContent(_ id: String, content: String, embedding: [Float]?, importance: Double,
                              sourceMessageID: String?, at date: Date = .now) throws {
        try db.transaction {
            try db.run("""
                UPDATE memories SET content = ?, importance = MAX(importance, ?), updated_at = ?,
                       source_message_id = COALESCE(?, source_message_id) WHERE id = ?
                """, [content, importance, date.unixSeconds, sourceMessageID, id])
            if let embedding { try setEmbedding(embedding, for: id) }
        }
    }

    /// Records that memories were injected into a prompt.
    public func markAccessed(_ ids: [String], at date: Date = .now) throws {
        try db.transaction {
            for id in ids { try db.run("UPDATE memories SET last_accessed_at = ? WHERE id = ?", [date.unixSeconds, id]) }
        }
    }

    /// Deletes one memory (history rows that pointed at it keep existing, unlinked).
    public func delete(_ id: String) throws {
        try db.transaction {
            try db.run("UPDATE memories SET superseded_by = NULL WHERE superseded_by = ?", [id])
            try db.run("DELETE FROM memories WHERE id = ?", [id])
        }
    }

    /// Deletes all memories, active and historical.
    public func deleteAll() throws {
        try db.transaction {
            try db.run("UPDATE memories SET superseded_by = NULL")
            try db.run("DELETE FROM memories")
        }
    }

    private func setEmbedding(_ vector: [Float], for id: String) throws {
        try db.run("INSERT OR REPLACE INTO memory_embeddings (memory_id, dim, vector) VALUES (?, ?, ?)",
                   [id, vector.count, EmbeddingMath.blob(from: vector)])
    }

    private static func record(_ row: SQLRow) -> MemoryRecord {
        MemoryRecord(
            id: row.string("id") ?? "",
            type: MemoryType(rawValue: row.string("type") ?? "") ?? .note,
            key: row.string("memory_key").flatMap(MemoryKey.init(rawValue:)),
            content: row.string("content") ?? "",
            importance: row.double("importance") ?? 0.5,
            confidence: row.double("confidence") ?? 1.0,
            sourceMessageID: row.string("source_message_id"),
            createdAt: Date(unixSeconds: row.double("created_at") ?? 0),
            updatedAt: Date(unixSeconds: row.double("updated_at") ?? 0),
            lastAccessedAt: row.double("last_accessed_at").map(Date.init(unixSeconds:)),
            superseded: (row.int("superseded") ?? 0) != 0,
            supersededBy: row.string("superseded_by"),
            embedding: row.data("vector").map(EmbeddingMath.vector(from:))
        )
    }
}
