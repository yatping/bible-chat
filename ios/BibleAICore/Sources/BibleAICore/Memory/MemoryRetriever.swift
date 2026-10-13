import Foundation

/// A memory selected for a prompt with its score.
public struct ScoredMemory: Sendable, Hashable {
    public var memory: MemoryRecord
    public var score: Double
    /// True for keyed preferences, which are always included.
    public var pinned: Bool
}

/// Selects memories for the MEMORY section (CONTRACTS §4).
///
/// `score = 0.60 × semantic + 0.25 × importance + 0.15 × recency`, `recency = 0.5 ^ (age_days / half_life)`.
/// Active keyed preferences are always included; the rest is filled by score (≥ 0.35), up to 4 in total.
public struct MemoryRetriever: Sendable {
    public var maxMemories = 4
    public var minScore = 0.35
    public let store: MemoryStore

    public init(store: MemoryStore) {
        self.store = store
    }

    /// - Parameter queryVector: the embedded question (nil → semantic similarity 0).
    public func retrieve(queryVector: [Float]?, now: Date = .now, markAccessed: Bool = true) throws -> [ScoredMemory] {
        let active = try store.activeMemories()
        let scored = active.map { memory in
            ScoredMemory(memory: memory, score: Self.score(memory, queryVector: queryVector, now: now),
                         pinned: memory.type == .preference && memory.key != nil)
        }
        let pinned = scored.filter(\.pinned).sorted {
            ($0.memory.importance, $0.memory.updatedAt) > ($1.memory.importance, $1.memory.updatedAt)
        }
        let others = scored.filter { !$0.pinned && $0.score >= minScore }.sorted { $0.score > $1.score }
        let selected = Array((pinned + others).prefix(maxMemories))
        if markAccessed, !selected.isEmpty {
            try store.markAccessed(selected.map(\.memory.id), at: now)
        }
        return selected
    }

    public static func score(_ memory: MemoryRecord, queryVector: [Float]?, now: Date) -> Double {
        let semantic = queryVector.flatMap { q in memory.embedding.map { Double(EmbeddingMath.dot(q, $0)) } } ?? 0
        return 0.60 * semantic + 0.25 * memory.importance + 0.15 * recency(memory, now: now)
    }

    /// `0.5 ^ (age_days / half_life_days)`; 1 for types without decay.
    public static func recency(_ memory: MemoryRecord, now: Date) -> Double {
        guard let halfLife = memory.type.halfLifeDays else { return 1 }
        let ageDays = max(0, now.timeIntervalSince(memory.updatedAt) / 86_400)
        return pow(0.5, ageDays / halfLife)
    }
}
