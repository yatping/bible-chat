import Foundation

/// Builds the FTS5 MATCH expression of CONTRACTS §3 (golden-tested with `fts_query_golden.json`).
///
/// lowercase → tokens `[\p{L}\p{N}]+` → drop stopwords and tokens shorter than 2 → de-duplicate →
/// quote each → join with ` OR `. Returns nil when no tokens remain (skip lexical search).
public struct FTSQueryBuilder: Sendable {
    public let stopwords: Set<String>

    public init(stopwords: Set<String> = FTSQueryBuilder.sharedStopwords) {
        self.stopwords = stopwords
    }

    /// Stopwords from `shared/stopwords.txt`.
    public static let sharedStopwords: Set<String> = Set(
        SharedTexts.stopwords.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    )

    private static let tokenPattern = try! NSRegularExpression(pattern: #"[\p{L}\p{N}]+"#)

    public func query(for text: String) -> String? {
        let tokens = Self.tokens(in: text.lowercased())
        var seen = Set<String>()
        var kept: [String] = []
        for token in tokens where token.unicodeScalars.count >= 2 && !stopwords.contains(token) {
            if seen.insert(token).inserted { kept.append(token) }
        }
        guard !kept.isEmpty else { return nil }
        return kept.map { "\"\($0)\"" }.joined(separator: " OR ")
    }

    static func tokens(in text: String) -> [String] {
        let ns = text as NSString
        return tokenPattern.matches(in: text, range: NSRange(location: 0, length: ns.length)).map { ns.substring(with: $0.range) }
    }
}

/// Lexical retrieval: BM25 over `chunks_fts`.
public struct FTSRetriever: Sendable {
    public let database: ContentDatabase
    public let builder: FTSQueryBuilder

    public init(database: ContentDatabase, builder: FTSQueryBuilder = FTSQueryBuilder()) {
        self.database = database
        self.builder = builder
    }

    /// Chunk ids ranked by BM25 (best first); empty when the query has no searchable tokens.
    public func search(_ text: String, limit: Int = 30) throws -> [Int64] {
        guard let match = builder.query(for: text) else { return [] }
        return try database.ftsSearch(match: match, limit: limit)
    }
}
