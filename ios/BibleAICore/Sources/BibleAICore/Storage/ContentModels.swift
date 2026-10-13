/// A Bible book from `books`.
public struct Book: Sendable, Hashable, Identifiable, Codable {
    public var index: Int
    public var osis: String
    public var name: String
    public var testament: String
    public var chapters: Int

    public var id: Int { index }

    public init(index: Int, osis: String, name: String, testament: String, chapters: Int) {
        self.index = index
        self.osis = osis
        self.name = name
        self.testament = testament
        self.chapters = chapters
    }
}

/// One verse from `verses`.
public struct Verse: Sendable, Hashable, Identifiable, Codable {
    public var bookIndex: Int
    public var chapter: Int
    public var number: Int
    public var text: String

    public var id: Int { bookIndex * 1_000_000 + chapter * 1_000 + number }

    public init(bookIndex: Int, chapter: Int, number: Int, text: String) {
        self.bookIndex = bookIndex
        self.chapter = chapter
        self.number = number
        self.text = text
    }
}

/// A retrieval unit from `chunks`.
public struct Chunk: Sendable, Hashable, Identifiable, Codable {
    public var id: Int64
    public var sourceType: String
    public var title: String
    public var reference: String
    public var translation: String?
    public var bookIndex: Int?
    public var chapter: Int?
    public var verseStart: Int?
    public var chapterEnd: Int?
    public var verseEnd: Int?
    public var text: String
    public var tokenEstimate: Int

    public init(id: Int64, sourceType: String, title: String, reference: String, translation: String? = nil,
                bookIndex: Int? = nil, chapter: Int? = nil, verseStart: Int? = nil, chapterEnd: Int? = nil,
                verseEnd: Int? = nil, text: String, tokenEstimate: Int) {
        self.id = id
        self.sourceType = sourceType
        self.title = title
        self.reference = reference
        self.translation = translation
        self.bookIndex = bookIndex
        self.chapter = chapter
        self.verseStart = verseStart
        self.chapterEnd = chapterEnd
        self.verseEnd = verseEnd
        self.text = text
        self.tokenEstimate = tokenEstimate
    }
}
