import Foundation
import Testing
@testable import BibleAICore

/// Shared fixtures in `shared/fixtures/` — the same files are checked by the Python components.
@Suite("Golden fixtures")
struct GoldenTests {
    // MARK: Prompt (§1)

    struct PromptGolden: Decodable {
        struct Case: Decodable {
            struct Input: Decodable {
                struct Source: Decodable { let type: String; let reference: String; let text: String }
                let memories: [String]
                let summary: String?
                let sources: [Source]
                let recent_messages: [ChatMessage]
                let question: String
            }
            let name: String
            let input: Input
            let expected_messages: [ChatMessage]
        }
        let cases: [Case]
    }

    @Test func promptGolden() throws {
        let golden = try Repo.sharedJSON("prompt_golden.json", as: PromptGolden.self)
        #expect(golden.cases.count >= 3)
        let builder = PromptBuilder()
        for c in golden.cases {
            let input = PromptInput(
                memories: c.input.memories,
                summary: c.input.summary,
                sources: c.input.sources.map { PromptSource(type: $0.type, reference: $0.reference, text: $0.text) },
                recentMessages: c.input.recent_messages,
                question: c.input.question
            )
            #expect(builder.messages(for: input) == c.expected_messages, "case \(c.name)")
        }
    }

    // MARK: Citations (§1)

    struct CitationGolden: Decodable {
        struct Case: Decodable {
            let name: String; let text: String; let source_count: Int
            let expected_text: String; let cited: [String]; let removed: [String]
        }
        let cases: [Case]
    }

    @Test func citationGolden() throws {
        let golden = try Repo.sharedJSON("citation_golden.json", as: CitationGolden.self)
        for c in golden.cases {
            let result = CitationValidator.validate(c.text, sourceCount: c.source_count)
            #expect(result.text == c.expected_text, "case \(c.name)")
            #expect(result.cited == c.cited, "case \(c.name)")
            #expect(result.removed == c.removed, "case \(c.name)")
        }
    }

    // MARK: FTS query (§3)

    struct FTSGolden: Decodable {
        struct Case: Decodable { let query: String; let expected: String? }
        let cases: [Case]
    }

    @Test func ftsQueryGolden() throws {
        let golden = try Repo.sharedJSON("fts_query_golden.json", as: FTSGolden.self)
        let builder = FTSQueryBuilder()
        for c in golden.cases {
            #expect(builder.query(for: c.query) == c.expected, "query \(c.query)")
        }
    }

    // MARK: RRF (§3)

    struct RRFGolden: Decodable {
        struct Case: Decodable {
            struct List: Decodable { let name: String; let weight: Double; let ids: [Int64] }
            let name: String; let lists: [List]; let top_k: Int; let expected: [Int64]
        }
        let k: Double
        let cases: [Case]
    }

    @Test func rrfGolden() throws {
        let golden = try Repo.sharedJSON("rrf_golden.json", as: RRFGolden.self)
        #expect(golden.k == HybridRanker.k)
        for c in golden.cases {
            let lists = c.lists.map { HybridRanker.RankedList(ids: $0.ids, weight: $0.weight) }
            #expect(HybridRanker.fuse(lists, k: golden.k, topK: c.top_k) == c.expected, "case \(c.name)")
        }
        // The weights the app uses match the fixture's list weights.
        let weights = Dictionary(uniqueKeysWithValues: golden.cases[0].lists.map { ($0.name, $0.weight) })
        #expect(weights["reference"] == HybridRanker.referenceWeight)
        #expect(weights["semantic"] == HybridRanker.semanticWeight)
        #expect(weights["lexical"] == HybridRanker.lexicalWeight)
    }

    // MARK: Memory extraction (§4)

    struct MemoryGolden: Decodable {
        struct Item: Decodable {
            let type: String; let key: String?; let value: String; let importance: Double; let confidence: Double
        }
        struct Case: Decodable { let name: String; let raw: String; let expected: [Item] }
        let cases: [Case]
    }

    @Test func memoryExtractionGolden() throws {
        let golden = try Repo.sharedJSON("memory_extraction_golden.json", as: MemoryGolden.self)
        for c in golden.cases {
            let parsed = MemoryExtractor.parse(c.raw)
            let expected = c.expected.map {
                ExtractedMemory(type: MemoryType(rawValue: $0.type)!, key: $0.key.flatMap(MemoryKey.init(rawValue:)),
                                value: $0.value, importance: $0.importance, confidence: $0.confidence)
            }
            #expect(parsed == expected, "case \(c.name)")
        }
    }
}
