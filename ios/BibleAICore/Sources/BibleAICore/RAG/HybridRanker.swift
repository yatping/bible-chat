/// Weighted reciprocal rank fusion (CONTRACTS §3), golden-tested with `rrf_golden.json`.
public enum HybridRanker {
    /// One ranked list of chunk ids (best first) and its weight.
    public struct RankedList: Sendable {
        public var ids: [Int64]
        public var weight: Double

        public init(ids: [Int64], weight: Double) {
            self.ids = ids
            self.weight = weight
        }
    }

    public static let k: Double = 60
    public static let referenceWeight: Double = 2.0
    public static let semanticWeight: Double = 1.0
    public static let lexicalWeight: Double = 1.0

    /// `score(d) = Σ w / (k + rank)` with 1-based ranks; ties broken by lower id. Returns the top `topK` ids.
    public static func fuse(_ lists: [RankedList], k: Double = HybridRanker.k, topK: Int) -> [Int64] {
        var scores: [Int64: Double] = [:]
        for list in lists {
            var seen = Set<Int64>()
            for (offset, id) in list.ids.enumerated() where seen.insert(id).inserted {
                scores[id, default: 0] += list.weight / (k + Double(offset + 1))
            }
        }
        return scores
            .sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }
            .prefix(max(0, topK))
            .map(\.key)
    }
}
