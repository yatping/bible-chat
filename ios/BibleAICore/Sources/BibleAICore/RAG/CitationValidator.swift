import Foundation

/// Post-generation citation cleanup (CONTRACTS §1), golden-tested with `citation_golden.json`.
public enum CitationValidator {
    public struct Result: Sendable, Equatable {
        /// Answer with grouped citations normalised and invalid ones removed.
        public var text: String
        /// Valid source ids ("S1"), first-appearance order, de-duplicated.
        public var cited: [String]
        /// Removed ids, first-appearance order, de-duplicated.
        public var removed: [String]
    }

    /// Ranges wider than this are not expanded (only their endpoints are kept).
    static let maxRangeExpansion = 20

    private static let item = #"S\d+(?:\s*[-–]\s*S?\d+)?"#
    private static let groupPattern = try! NSRegularExpression(pattern: #"\[\s*("# + item + #"(?:\s*[,;]\s*"# + item + #")*)\s*\]"#)
    private static let itemPattern = try! NSRegularExpression(pattern: #"S(\d+)(?:\s*[-–]\s*S?(\d+))?"#)
    private static let citationPattern = try! NSRegularExpression(pattern: #"[ \t]*\[S(\d+)\]"#)

    /// Validates `text` against sources `S1…S{sourceCount}`.
    public static func validate(_ text: String, sourceCount: Int) -> Result {
        let normalised = normaliseGroups(text)
        let ns = normalised as NSString
        var output = ""
        var cursor = 0
        var cited: [String] = []
        var removed: [String] = []
        for match in citationPattern.matches(in: normalised, range: NSRange(location: 0, length: ns.length)) {
            let number = Int(ns.substring(with: match.range(at: 1))) ?? 0
            let id = "S\(number)"
            if number >= 1, number <= sourceCount {
                if !cited.contains(id) { cited.append(id) }
                continue
            }
            if !removed.contains(id) { removed.append(id) }
            output += ns.substring(with: NSRange(location: cursor, length: match.range.location - cursor))
            cursor = match.range.location + match.range.length
        }
        output += ns.substring(from: cursor)
        return Result(text: output, cited: cited, removed: removed)
    }

    /// `[S1, S2]` → `[S1][S2]`, `[S2-S4]` → `[S2][S3][S4]`.
    static func normaliseGroups(_ text: String) -> String {
        let ns = text as NSString
        var output = ""
        var cursor = 0
        for match in groupPattern.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            output += ns.substring(with: NSRange(location: cursor, length: match.range.location - cursor))
            let inner = ns.substring(with: match.range(at: 1)) as NSString
            var expanded = ""
            for item in itemPattern.matches(in: inner as String, range: NSRange(location: 0, length: inner.length)) {
                let start = Int(inner.substring(with: item.range(at: 1))) ?? 0
                if item.range(at: 2).location != NSNotFound, let end = Int(inner.substring(with: item.range(at: 2))), end >= start {
                    let ids = end - start <= maxRangeExpansion ? Array(start...end) : [start, end]
                    expanded += ids.map { "[S\($0)]" }.joined()
                } else {
                    expanded += "[S\(start)]"
                }
            }
            output += expanded
            cursor = match.range.location + match.range.length
        }
        output += ns.substring(from: cursor)
        return output
    }
}
