import Foundation

/// An inclusive span `(chapter, verseStart) … (chapterEnd, verseEnd)` within one book.
/// Chapter-only references span the whole chapter (`verseEnd == ScriptureReference.wholeChapter`).
public struct ScriptureReference: Sendable, Hashable {
    public static let wholeChapter = 999

    public var bookIndex: Int
    public var chapter: Int
    public var verseStart: Int
    public var chapterEnd: Int
    public var verseEnd: Int

    public init(bookIndex: Int, chapter: Int, verseStart: Int = 1, chapterEnd: Int? = nil, verseEnd: Int = wholeChapter) {
        self.bookIndex = bookIndex
        self.chapter = chapter
        self.verseStart = verseStart
        self.chapterEnd = chapterEnd ?? chapter
        self.verseEnd = verseEnd
    }

    /// `chapter * 1000 + verse` of the first verse (the encoding used in SQL range checks).
    public var startKey: Int { chapter * 1000 + verseStart }
    /// `chapterEnd * 1000 + verseEnd` of the last verse.
    public var endKey: Int { chapterEnd * 1000 + verseEnd }
}

/// Finds explicit Bible references in free text: "Matthew 18:22", "Mt 18", "1 Cor 13:4-7", "John 3:16-4:2".
///
/// Book names come from `book_aliases` (lowercase, single spaces, no dots). Mirrors `content/bible_content/references.py`:
/// aliases that are also common words ("is", "am", "job", "acts", …) are only accepted when capitalised or followed by
/// `chapter:verse`; a lone number after a single-chapter book is a verse ("Jude 3").
public struct ReferenceParser: Sendable {
    private let aliases: [String: Int]
    private let chapterCounts: [Int: Int]
    private let regex: NSRegularExpression?

    private static let riskyAliases: Set<String> = [
        "is", "am", "as", "he", "so", "do", "re", "la", "da", "ho", "ac", "co", "ex", "job", "acts",
        "mark", "song", "numbers", "ruth", "jude", "amos", "joel", "ob", "na", "es", "ti", "de", "ge",
        "mr", "pr", "jo", "sg", "pp", "ml", "jd", "jm", "lu", "le", "ja", "ro", "ga", "dt", "hg", "mc",
        "jl", "nb", "nu", "ru", "hb", "rv", "lk", "mk", "jn", "eze", "dan", "mal", "mic", "hag",
        "nah", "jer", "lam", "col", "tit", "jam", "jas", "pro", "act", "rev", "est", "joe", "jon",
    ]

    /// - Parameters:
    ///   - aliases: normalised alias → book_index (from `ContentDatabase.aliases()`).
    ///   - chapterCounts: book_index → number of chapters, used to reject impossible references.
    public init(aliases: [String: Int], chapterCounts: [Int: Int] = [:]) {
        self.aliases = aliases
        self.chapterCounts = chapterCounts
        let alternatives = aliases.keys
            .sorted { $0.count != $1.count ? $0.count > $1.count : $0 < $1 }
            .map { alias in
                alias.split(separator: " ").map { NSRegularExpression.escapedPattern(for: String($0)) }
                    .joined(separator: #"[\s.]+"#) + #"\.?"#
            }
        let pattern = #"(?<![\p{L}\p{N}_])(?<book>"# + alternatives.joined(separator: "|") + ")"
            + #"\s*(?<c1>\d{1,3})"#
            + #"(?:\s*[:.]\s*(?<v1>\d{1,3})[a-c]?)?"#
            + #"(?:\s*[-–—]\s*(?<n2>\d{1,3})(?:\s*[:.]\s*(?<v2>\d{1,3})[a-c]?)?)?"#
            + #"(?![:\p{L}\p{N}_])"#
        regex = alternatives.isEmpty ? nil : try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
    }

    /// All references in `text`, in order of appearance, without duplicates.
    public func references(in text: String) -> [ScriptureReference] {
        guard let regex else { return [] }
        let ns = text as NSString
        var found: [ScriptureReference] = []
        for match in regex.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            func group(_ name: String) -> String? {
                let range = match.range(withName: name)
                return range.location == NSNotFound ? nil : ns.substring(with: range)
            }
            guard let bookText = group("book") else { continue }
            let alias = Self.normalise(bookText)
            guard let book = aliases[alias] ?? aliases[alias.replacingOccurrences(of: " ", with: "")] else { continue }
            let v1 = group("v1").flatMap(Int.init)
            if Self.riskyAliases.contains(alias), !(bookText.first?.isUppercase ?? false), v1 == nil { continue }
            guard let c1 = group("c1").flatMap(Int.init) else { continue }
            let reference = build(book: book, c1: c1, v1: v1, n2: group("n2").flatMap(Int.init), v2: group("v2").flatMap(Int.init))
            if let reference, !found.contains(reference) { found.append(reference) }
        }
        return found
    }

    /// The first reference in `text`, if any.
    public func reference(in text: String) -> ScriptureReference? {
        references(in: text).first
    }

    private func build(book: Int, c1: Int, v1: Int?, n2: Int?, v2: Int?) -> ScriptureReference? {
        var c1 = c1, v1 = v1
        let chapters = chapterCounts[book]
        if chapters == 1, v1 == nil, c1 != 1 {
            // "Jude 3" means Jude 1:3.
            v1 = c1
            c1 = 1
        }
        let reference: ScriptureReference
        if let v1 {
            if let n2, let v2 {
                reference = ScriptureReference(bookIndex: book, chapter: c1, verseStart: v1, chapterEnd: n2, verseEnd: v2)
            } else if let n2 {
                reference = ScriptureReference(bookIndex: book, chapter: c1, verseStart: v1, chapterEnd: c1, verseEnd: n2)
            } else {
                reference = ScriptureReference(bookIndex: book, chapter: c1, verseStart: v1, chapterEnd: c1, verseEnd: v1)
            }
        } else {
            reference = ScriptureReference(bookIndex: book, chapter: c1, verseStart: 1, chapterEnd: n2 ?? c1,
                                           verseEnd: v2 ?? ScriptureReference.wholeChapter)
        }
        guard reference.chapter >= 1, reference.chapter <= reference.chapterEnd, reference.verseStart >= 1,
              reference.startKey <= reference.endKey else { return nil }
        if let chapters, reference.chapterEnd > chapters { return nil }
        return reference
    }

    static func normalise(_ text: String) -> String {
        text.lowercased().replacingOccurrences(of: ".", with: " ")
            .split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }
}
