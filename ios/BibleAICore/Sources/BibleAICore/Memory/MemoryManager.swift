import Foundation

/// Applies extracted memories to `user.sqlite` with the consolidation rules of CONTRACTS §4.
public struct MemoryManager: Sendable {
    /// What happened to one extracted item.
    public enum Outcome: Sendable, Equatable {
        /// New row.
        case inserted(id: String)
        /// Same keyed value re-stated; `updated_at` bumped.
        case touched(id: String)
        /// Keyed value changed; old row superseded by the new one.
        case superseded(oldID: String, newID: String)
        /// Unkeyed near-duplicate (cosine ≥ threshold); existing row's content updated.
        case merged(id: String)
    }

    public static let duplicateThreshold: Float = 0.92

    public let store: MemoryStore
    public let embedder: (any EmbeddingEngine)?

    public init(store: MemoryStore, embedder: (any EmbeddingEngine)?) {
        self.store = store
        self.embedder = embedder
    }

    /// Consolidates `memories` (already validated by `MemoryExtractor`). Each item is applied in its own transaction.
    @discardableResult
    public func consolidate(_ memories: [ExtractedMemory], sourceMessageID: String? = nil, now: Date = .now) async throws -> [Outcome] {
        guard !memories.isEmpty else { return [] }
        // Memories are embedded as documents with `title: none` (CONTRACTS §2).
        let vectors: [[Float]?]
        if let embedder, let embedded = try? await embedder.embed(memories.map(\.value), kind: .document(title: nil)),
           embedded.count == memories.count {
            vectors = embedded
        } else {
            vectors = Array(repeating: nil, count: memories.count)
        }
        var outcomes: [Outcome] = []
        for (memory, vector) in zip(memories, vectors) {
            outcomes.append(try apply(memory, embedding: vector, sourceMessageID: sourceMessageID, now: now))
        }
        return outcomes
    }

    func apply(_ memory: ExtractedMemory, embedding: [Float]?, sourceMessageID: String?, now: Date) throws -> Outcome {
        try store.db.transaction {
            let record = MemoryRecord(type: memory.type, key: memory.key, content: memory.value,
                                      importance: memory.importance, confidence: memory.confidence,
                                      sourceMessageID: sourceMessageID, createdAt: now, embedding: embedding)
            if let key = memory.key {
                guard let active = try store.activeMemory(key: key) else {
                    try store.insert(record)
                    return .inserted(id: record.id)
                }
                if Self.normalise(active.content) == Self.normalise(memory.value) {
                    try store.touch(active.id, importance: memory.importance, at: now)
                    return .touched(id: active.id)
                }
                try store.supersede(active.id, with: record)
                return .superseded(oldID: active.id, newID: record.id)
            }

            let candidates = try store.activeMemories().filter { $0.key == nil }
            if let exact = candidates.first(where: { Self.normalise($0.content) == Self.normalise(memory.value) }) {
                try store.touch(exact.id, importance: memory.importance, at: now)
                return .touched(id: exact.id)
            }
            if let embedding,
               let best = candidates
                .compactMap({ c in c.embedding.map { (c, EmbeddingMath.dot($0, embedding)) } })
                .max(by: { $0.1 < $1.1 }),
               best.1 >= Self.duplicateThreshold {
                try store.updateContent(best.0.id, content: memory.value, embedding: embedding,
                                        importance: memory.importance, sourceMessageID: sourceMessageID, at: now)
                return .merged(id: best.0.id)
            }
            try store.insert(record)
            return .inserted(id: record.id)
        }
    }

    /// Case- and whitespace-insensitive comparison form.
    static func normalise(_ text: String) -> String {
        text.lowercased().split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }
}
