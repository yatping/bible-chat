import Foundation
import Testing
@testable import BibleAICore

@Suite("Chat template (Gemma 4 fallback)")
struct ChatTemplateTests {
    @Test func rendersSystemUserAndGenerationPrompt() {
        let text = ChatTemplate.gemma4([.system("Be brief.\n"), .user("  Hi  ")])
        #expect(text == "<|turn>system\nBe brief.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n")
    }

    @Test func multiTurnWithBOSAndThinkingStripped() {
        let text = ChatTemplate.gemma4(
            [.system("S"), .user("Q1"), .assistant("<|channel>thought\nhmm<channel|>A1 [S1]"), .user("Q2")],
            includeBOS: true)
        #expect(text == "<bos><|turn>system\nS<turn|>\n<|turn>user\nQ1<turn|>\n<|turn>model\nA1 [S1]<turn|>\n<|turn>user\nQ2<turn|>\n<|turn>model\n")
    }

    @Test func noSystemTurnWithoutSystemMessage() {
        #expect(ChatTemplate.gemma4([.user("Q")], addGenerationPrompt: false) == "<|turn>user\nQ<turn|>\n")
    }

    @Test func consecutiveAssistantMessagesShareATurn() {
        let text = ChatTemplate.gemma4([.user("Q"), .assistant("A"), .assistant("B")], addGenerationPrompt: false)
        #expect(text == "<|turn>user\nQ<turn|>\n<|turn>model\nAB<turn|>\n")
    }
}

@Suite("Token budget")
struct TokenBudgetTests {
    @Test func estimateIsCeilCharsOver4() {
        #expect(TokenBudget.estimate("") == 0)
        #expect(TokenBudget.estimate("abcd") == 1)
        #expect(TokenBudget.estimate("abcde") == 2)
        #expect(TokenBudget.estimate("“é”") == 1) // scalars, like Python len()
    }

    @Test func recentWindowKeepsLast16AndTrimsOldestFirst() {
        let budget = TokenBudget.default
        let short = (0..<20).map { ChatMessage.user("m\($0)") }
        #expect(budget.trimRecent(short).map(\.content) == (4..<20).map { "m\($0)" })

        let long = (0..<4).map { ChatMessage.assistant(String(repeating: "x", count: 1600) + "\($0)") } // ~400 tokens each
        let trimmed = budget.trimRecent(long)
        #expect(trimmed.count == 2)
        #expect(trimmed.last?.content.hasSuffix("3") == true)
    }

    @Test func memoriesAndSummaryRespectBudgets() {
        let budget = TokenBudget.default
        let memories = (0..<5).map { _ in String(repeating: "m", count: 400) } // 100 tokens each
        #expect(budget.trimMemories(memories).count == 3)
        let summary = budget.trimSummary(String(repeating: "word ", count: 500))!
        #expect(TokenBudget.estimate(summary) <= 300)
    }

    @Test func generationDefaultsMatchContract() {
        let a = GenerationConfig.answer
        #expect(a.temperature == 0.3 && a.topP == 0.9 && a.topK == 40 && a.repeatPenalty == 1.1 && a.maxTokens == 1200)
        #expect(GenerationConfig.deterministic.temperature == 0 && GenerationConfig.deterministic.maxTokens == 256)
        #expect(TokenBudget.default.contextLength == 6144)
    }
}

@Suite("Embeddings")
struct EmbeddingTests {
    @Test func promptsMatchContract() {
        #expect(EmbeddingPrompt.query("forgive") == "task: search result | query: forgive")
        #expect(EmbeddingPrompt.document(title: "Matthew 18:21-35", text: "t") == "title: Matthew 18:21-35 | text: t")
        #expect(EmbeddingPrompt.document(title: "", text: "t") == "title: none | text: t")
        #expect(EmbeddingPrompt.format("m", kind: .document()) == "title: none | text: m")
    }

    @Test func truncatesTo256AndNormalises() async throws {
        let raw = (0..<768).map { Float($0 % 7) + 1 }
        let v = EmbeddingMath.truncateAndNormalize(raw)
        #expect(v.count == 256)
        #expect(abs(EmbeddingMath.dot(v, v) - 1) < 1e-5)
        #expect(v[0] / v[1] == raw[0] / raw[1])

        let embedded = try await FakeEmbedder().embed(["a forgiving king"], kind: .query)
        #expect(embedded[0].count == 256)
    }

    @Test func float32BlobRoundTrip() {
        let v: [Float] = [0, 1.5, -2.25, .pi]
        #expect(EmbeddingMath.vector(from: EmbeddingMath.blob(from: v)) == v)
        #expect(EmbeddingMath.blob(from: [1]) == Data([0x00, 0x00, 0x80, 0x3F]))
    }
}

@Suite("Vector store")
struct VectorStoreTests {
    @Test func float16RoundTripIsLittleEndianBinary16() throws {
        let data = VectorStore.float16Data(from: [[1, -2, 0.5, 65504]])
        #expect(data == Data([0x00, 0x3C, 0x00, 0xC0, 0x00, 0x38, 0xFF, 0x7B]))
        let store = try VectorStore(float16Data: data, count: 1, dimension: 4)
        #expect(try store.scores(for: [1, 0, 0, 0]) == [1])
        #expect(try store.scores(for: [0, 1, 0, 0]) == [-2])
    }

    @Test func topKMatchesBruteForce() throws {
        var generator = SplitMix(seed: 42)
        let count = 500, dim = 256
        let vectors = (0..<count).map { _ in EmbeddingMath.truncateAndNormalize((0..<dim).map { _ in generator.nextFloat() }) }
        let store = try VectorStore(float16Data: VectorStore.float16Data(from: vectors), count: count, dimension: dim)
        let query = vectors[123]
        let top = try store.topK(query, k: 10)
        #expect(top.first?.row == 123)
        #expect(abs((top.first?.score ?? 0) - 1) < 2e-3) // f16 precision

        let scores = try store.scores(for: query)
        let expected = scores.enumerated().sorted { $0.element != $1.element ? $0.element > $1.element : $0.offset < $1.offset }
            .prefix(10).map(\.offset)
        #expect(top.map(\.row) == expected)
        #expect(top.map(\.score) == top.map(\.score).sorted(by: >))
    }

    @Test func topKTieBreaksByLowerRow() {
        let top = VectorStore.topK([0.5, 0.9, 0.5, 0.9, 0.1], k: 3)
        #expect(top.map(\.row) == [1, 3, 0])
    }

    @Test func rejectsWrongSizes() {
        #expect(throws: VectorStore.Error.self) { try VectorStore(float16Data: Data(count: 10), count: 2, dimension: 4) }
        let store = try? VectorStore(float16Data: Data(count: 16), count: 2, dimension: 4)
        #expect(throws: VectorStore.Error.self) { try store?.scores(for: [1, 2]) }
    }
}

/// Deterministic PRNG for tests.
struct SplitMix {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    mutating func nextFloat() -> Float { Float(next() >> 40) / Float(1 << 24) * 2 - 1 }
}

@Suite("UTF-8 stream decoder")
struct UTF8StreamDecoderTests {
    @Test func holdsPartialMultibyteSequences() {
        var decoder = UTF8StreamDecoder()
        let bytes = Array("a“é”🙏".utf8)
        var out = ""
        for byte in bytes {
            out += decoder.append([byte])
        }
        out += decoder.finish()
        #expect(out == "a“é”🙏")

        var split = UTF8StreamDecoder()
        let emoji = Array("🙏".utf8)
        #expect(split.append(emoji.prefix(2)) == "")
        #expect(split.append(emoji.suffix(2) + Array(" ok".utf8)) == "🙏 ok")
    }

    @Test func invalidRemainderIsReplacedOnFinish() {
        var decoder = UTF8StreamDecoder()
        #expect(decoder.append([0x61, 0xE2, 0x80]) == "a")
        #expect(decoder.finish() == "\u{FFFD}")
    }
}
