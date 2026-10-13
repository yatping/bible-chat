import Foundation
import Testing
@testable import BibleAICore

@Suite("Reference parser")
struct ReferenceParserTests {
    let parser = ReferenceParser(aliases: TestContent.aliases,
                                 chapterCounts: Dictionary(uniqueKeysWithValues: TestContent.books.map { ($0.0, $0.5) }))

    @Test func verse() {
        #expect(parser.reference(in: "What does Matthew 18:22 mean?") ==
                ScriptureReference(bookIndex: 40, chapter: 18, verseStart: 22, chapterEnd: 18, verseEnd: 22))
    }

    @Test func chapterOnlyAbbreviation() {
        #expect(parser.reference(in: "Summarise Mt 18") ==
                ScriptureReference(bookIndex: 40, chapter: 18, verseStart: 1, chapterEnd: 18, verseEnd: ScriptureReference.wholeChapter))
    }

    @Test func numberedBookWithVerseRange() {
        let expected = ScriptureReference(bookIndex: 46, chapter: 13, verseStart: 4, chapterEnd: 13, verseEnd: 7)
        #expect(parser.reference(in: "Love in 1 Cor 13:4-7") == expected)
        #expect(parser.reference(in: "love in 1cor 13:4–7") == expected)
        #expect(parser.reference(in: "1 Cor. 13:4-7") == expected)
        #expect(parser.reference(in: "First Corinthians 13:4-7") == expected)
    }

    @Test func crossChapterRange() {
        #expect(parser.reference(in: "john 3:16-4:2") ==
                ScriptureReference(bookIndex: 43, chapter: 3, verseStart: 16, chapterEnd: 4, verseEnd: 2))
    }

    @Test func singleChapterBookAndValidation() {
        #expect(parser.reference(in: "Jude 3") == ScriptureReference(bookIndex: 65, chapter: 1, verseStart: 3, chapterEnd: 1, verseEnd: 3))
        #expect(parser.reference(in: "Romans 17") == nil) // Romans has 16 chapters
        #expect(parser.reference(in: "Genesis 3:5-2") == nil)
    }

    @Test func riskyAliasesNeedCapitalOrVerse() {
        #expect(parser.reference(in: "ro 8 is long") == nil)
        #expect(parser.reference(in: "Ro 8")?.bookIndex == 45)
        #expect(parser.reference(in: "ro 8:28")?.bookIndex == 45)
        #expect(parser.references(in: "Compare Genesis 1 and John 3:16, then Genesis 1 again").count == 2)
        #expect(parser.reference(in: "Why did Jesus tell Peter to forgive 70x7?") == nil)
    }
}

@Suite("Content DB, FTS and RAG engine")
struct RAGEngineTests {
    @Test func versesAndBooks() throws {
        let content = try TestContent.make()
        let genesis = try content.database.verses(book: 1, chapter: 1)
        #expect(genesis.count == 31)
        #expect(genesis[0].text == "In the beginning, God created the heavens and the earth.")
        let span = try content.database.verses(in: ScriptureReference(bookIndex: 43, chapter: 3, verseStart: 16, chapterEnd: 3, verseEnd: 17))
        #expect(span.map(\.number) == [16, 17])
        #expect(try content.database.books().map(\.name).contains("Romans"))
    }

    @Test func ftsFindsForgiveness() throws {
        let content = try TestContent.make()
        let ids = try FTSRetriever(database: content.database).search("Why did Jesus tell Peter to forgive 70x7?")
        #expect(Set(ids.prefix(2)) == [2, 100]) // Matthew 18:21-35 and the commentary
        #expect(try FTSRetriever(database: content.database).search("who is it").isEmpty)
    }

    @Test func referenceLookupFindsOverlappingChunks() throws {
        let content = try TestContent.make()
        let db = content.database
        #expect(try db.scriptureChunkIDs(overlapping: ScriptureReference(bookIndex: 43, chapter: 3, verseStart: 16, chapterEnd: 3, verseEnd: 16)) == [3])
        #expect(try db.scriptureChunkIDs(overlapping: ScriptureReference(bookIndex: 43, chapter: 3)) == [3, 4])
        #expect(try db.scriptureChunkIDs(overlapping: ScriptureReference(bookIndex: 1, chapter: 2, verseStart: 2, chapterEnd: 2, verseEnd: 2)) == [1])
        #expect(try db.scriptureChunkIDs(overlapping: ScriptureReference(bookIndex: 45, chapter: 6)).isEmpty)
    }

    @Test func semanticSearchRanksMatchingPericope() async throws {
        let content = try TestContent.make()
        let q = try await content.embedder.embed("Nicodemus born again of water and Spirit", kind: .query)
        #expect(try content.semantic.search(queryVector: q, limit: 3).contains(3))
    }

    @Test func hybridRetrievalForForgivenessQuestion() async throws {
        let content = try TestContent.make()
        let sources = try await content.ragEngine().retrieve("Why did Jesus tell Peter to forgive 70x7?")
        // The fake embedder favours the short commentary; both must lead the fused list.
        #expect(Set(sources.prefix(2).map(\.reference)) == ["Matthew 18:21-35", "Matthew 18 commentary"])
        #expect(sources.map(\.id) == (1...sources.count).map { "S\($0)" })
        #expect(sources.count <= 6)
    }

    @Test func explicitReferenceDominates() async throws {
        let content = try TestContent.make()
        let rag = try content.ragEngine()
        #expect(try await rag.retrieve("Explain Romans 8:28").first?.reference == "Romans 8:18-39")
        #expect(try await rag.retrieve("Explain Romans 5.").prefix(2).map(\.reference).sorted() == ["Romans 5:1-11", "Romans 5:12-21"])
        #expect(try await rag.retrieve("John 3:16").first?.reference == "John 3:1-21")
    }

    @Test func tokenBudgetTrimsButKeepsOne() async throws {
        let content = try TestContent.make()
        var config = RAGConfiguration()
        config.sourceTokenBudget = 10
        let sources = try await content.ragEngine(configuration: config).retrieve("Explain Genesis 1")
        #expect(sources.count == 1)
        #expect(sources[0].reference == "Genesis 1:1-2:3")

        let chunks = (1...4).map { Chunk(id: Int64($0), sourceType: "scripture", title: "", reference: "", text: "", tokenEstimate: 1000) }
        #expect(RAGEngine.trim(chunks, toTokens: 2500).count == 2)
    }

    @Test func worksWithoutSemanticIndex() async throws {
        let content = try TestContent.make()
        let rag = try RAGEngine(database: content.database, semantic: nil, embedder: nil)
        #expect(try await rag.retrieve("forgive seventy times seven").isEmpty == false)
    }
}
