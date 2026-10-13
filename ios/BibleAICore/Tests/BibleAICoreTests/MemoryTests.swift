import Foundation
import Testing
@testable import BibleAICore

@Suite("User database and memory consolidation")
struct MemoryConsolidationTests {
    let database: UserDatabase
    let store: MemoryStore
    let manager: MemoryManager

    init() throws {
        database = try UserDatabase.inMemory()
        store = MemoryStore(database: database)
        manager = MemoryManager(store: store, embedder: FakeEmbedder())
    }

    @Test func schemaIsCreatedFromSharedFile() throws {
        #expect(try database.db.queryOne("PRAGMA user_version")?.values.first == .integer(1))
        let tables = try database.db.query("SELECT name FROM sqlite_master WHERE type = 'table'").compactMap { $0.string("name") }
        #expect(Set(tables).isSuperset(of: ["memories", "memory_embeddings", "conversations", "messages"]))
        #expect(try database.db.queryOne("PRAGMA foreign_keys")?.values.first == .integer(1))
    }

    @Test func reopeningFileKeepsData() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("user-\(UUID().uuidString).sqlite")
        try ConversationStore(database: UserDatabase(url: url)).createConversation(title: "kept")
        #expect(try ConversationStore(database: UserDatabase(url: url)).conversations().map(\.title) == ["kept"])
    }

    @Test func translationChangeSupersedesAndKeepsHistory() async throws {
        let web = ExtractedMemory(type: .preference, key: .bibleTranslation, value: "Preferred Bible translation is WEB.", importance: 0.8)
        let asv = ExtractedMemory(type: .preference, key: .bibleTranslation, value: "Preferred Bible translation is ASV.", importance: 0.8)

        let first = try await manager.consolidate([web])
        guard case .inserted(let webID) = first.first else { Issue.record("expected insert"); return }

        let second = try await manager.consolidate([asv])
        guard case .superseded(let oldID, let newID) = second.first else { Issue.record("expected supersede"); return }
        #expect(oldID == webID)

        let active = try store.activeMemories()
        #expect(active.map(\.content) == ["Preferred Bible translation is ASV."])
        #expect(active.first?.embedding?.count == 256)
        let history = try store.supersededMemories()
        #expect(history.map(\.content) == ["Preferred Bible translation is WEB."])
        #expect(history.first?.supersededBy == newID)
    }

    @Test func sameValueOnlyTouches() async throws {
        let t0 = Date(timeIntervalSince1970: 1_000)
        let t1 = Date(timeIntervalSince1970: 2_000)
        let m = ExtractedMemory(type: .preference, key: .answerLength, value: "User prefers short answers.")
        try await manager.consolidate([m], now: t0)
        let again = ExtractedMemory(type: .preference, key: .answerLength, value: "  user prefers   SHORT answers. ")
        let outcome = try await manager.consolidate([again], now: t1)
        guard case .touched = outcome.first else { Issue.record("expected touch, got \(outcome)"); return }
        let active = try store.activeMemories()
        #expect(active.count == 1)
        #expect(active[0].updatedAt == t1 && active[0].createdAt == t0)
        #expect(try store.supersededMemories().isEmpty)
    }

    @Test func uniqueActiveKeyInvariantIsEnforcedBySchema() throws {
        try store.insert(MemoryRecord(type: .preference, key: .bibleTranslation, content: "WEB"))
        #expect(throws: SQLiteError.self) {
            try store.insert(MemoryRecord(type: .preference, key: .bibleTranslation, content: "KJV"))
        }
        // A superseded row with the same key is allowed.
        try store.insert(MemoryRecord(type: .preference, key: .bibleTranslation, content: "KJV", superseded: true))
    }

    @Test func unkeyedNearDuplicateIsMerged() async throws {
        let a = ExtractedMemory(type: .studyInterest, key: nil, value: "Studying grace in Romans.")
        let b = ExtractedMemory(type: .studyInterest, key: nil, value: "Studying grace in Romans!!")
        let c = ExtractedMemory(type: .goal, key: nil, value: "Memorise Psalm 23 this month.")
        try await manager.consolidate([a])
        let outcomes = try await manager.consolidate([b, c])
        guard case .merged = outcomes[0], case .inserted = outcomes[1] else { Issue.record("got \(outcomes)"); return }
        let active = try store.activeMemories()
        #expect(active.count == 2)
        #expect(active.contains { $0.content == "Studying grace in Romans!!" })
    }

    @Test func deleteOneAndAll() async throws {
        try await manager.consolidate([ExtractedMemory(type: .preference, key: .bibleTranslation, value: "WEB")])
        try await manager.consolidate([ExtractedMemory(type: .preference, key: .bibleTranslation, value: "ASV")])
        try await manager.consolidate([ExtractedMemory(type: .note, key: nil, value: "Leads a youth group.")])
        let asv = try #require(try store.activeMemory(key: .bibleTranslation))
        try store.delete(asv.id) // referenced by the WEB history row
        #expect(try store.activeMemory(key: .bibleTranslation) == nil)
        #expect(try store.supersededMemories().first?.supersededBy == nil)
        try store.deleteAll()
        #expect(try store.activeMemories().isEmpty && store.supersededMemories().isEmpty)
        #expect(try database.db.queryOne("SELECT COUNT(*) AS n FROM memory_embeddings")?.int("n") == 0)
    }
}

@Suite("Memory retrieval scoring")
struct MemoryRetrieverTests {
    let now = Date(timeIntervalSince1970: 10_000_000)

    @Test func scoreFormulaAndDecay() {
        let q = EmbeddingMath.truncateAndNormalize([1, 0])
        let m = MemoryRecord(type: .progress, key: nil, content: "Read Romans 1-4.", importance: 0.4,
                             createdAt: now.addingTimeInterval(-3 * 86_400), embedding: q)
        // semantic 1, importance 0.4, progress half-life 3 days → recency 0.5
        #expect(abs(MemoryRetriever.score(m, queryVector: q, now: now) - (0.6 + 0.1 + 0.075)) < 1e-6)

        var goal = m
        goal.type = .goal
        goal.updatedAt = now.addingTimeInterval(-90 * 86_400)
        #expect(abs(MemoryRetriever.recency(goal, now: now) - 0.5) < 1e-9)
        goal.type = .note
        #expect(MemoryRetriever.recency(goal, now: now) == 1)
        goal.type = .studyInterest
        goal.updatedAt = now.addingTimeInterval(-60 * 86_400)
        #expect(abs(MemoryRetriever.recency(goal, now: now) - 0.25) < 1e-9)
    }

    @Test func preferencesPinnedThenScoreCapAndMinimum() throws {
        let store = MemoryStore(database: try UserDatabase.inMemory())
        let q = EmbeddingMath.truncateAndNormalize([1, 0, 0])
        let orthogonal = EmbeddingMath.truncateAndNormalize([0, 1, 0])
        try store.insert(MemoryRecord(type: .preference, key: .bibleTranslation, content: "WEB", importance: 0.8, createdAt: now, embedding: orthogonal))
        try store.insert(MemoryRecord(type: .preference, key: .answerLength, content: "Short", importance: 0.7, createdAt: now, embedding: orthogonal))
        try store.insert(MemoryRecord(type: .studyInterest, key: nil, content: "relevant A", importance: 0.5, createdAt: now, embedding: q))
        try store.insert(MemoryRecord(type: .note, key: nil, content: "relevant B", importance: 0.9, createdAt: now, embedding: q))
        try store.insert(MemoryRecord(type: .goal, key: nil, content: "relevant C", importance: 0.1, createdAt: now, embedding: q))
        // Irrelevant and old: 0.6×0 + 0.25×0.5 + 0.15×~0 < 0.35 → excluded.
        try store.insert(MemoryRecord(type: .progress, key: nil, content: "stale", importance: 0.5,
                                      createdAt: now.addingTimeInterval(-60 * 86_400), embedding: orthogonal))

        let selected = try MemoryRetriever(store: store).retrieve(queryVector: q, now: now)
        #expect(selected.map(\.memory.content) == ["WEB", "Short", "relevant B", "relevant A"])
        #expect(selected.prefix(2).allSatisfy { $0.pinned })
        #expect(try store.memory(id: selected[0].memory.id)?.lastAccessedAt == now)

        let noQuery = try MemoryRetriever(store: store).retrieve(queryVector: nil, now: now)
        #expect(!noQuery.map(\.memory.content).contains("stale"))
    }
}

@Suite("Conversation summariser")
struct SummarizerTests {
    @Test func triggerRule() {
        #expect(ConversationSummarizer.pendingRange(totalMessages: 16, covered: 0) == nil) // 16 uncovered, not > 16
        #expect(ConversationSummarizer.pendingRange(totalMessages: 17, covered: 0) == 0..<9)
        #expect(ConversationSummarizer.pendingRange(totalMessages: 25, covered: 9) == nil)
        #expect(ConversationSummarizer.pendingRange(totalMessages: 26, covered: 9) == 9..<18)
    }

    @Test func summarisesOlderMessagesWithSharedPrompt() async throws {
        let store = ConversationStore(database: try UserDatabase.inMemory())
        let conversation = try store.createConversation()
        for i in 0..<25 {
            try store.append(StoredMessage(conversationID: conversation.id, role: i.isMultiple(of: 2) ? .user : .assistant,
                                           content: "message \(i)", createdAt: Date(timeIntervalSince1970: Double(i))))
        }
        let llm = MockLLMEngine { _, _ in "  Discussed messages 0 to 16. " }
        let summarizer = ConversationSummarizer(llm: llm, conversations: store)
        #expect(try await summarizer.summarizeIfNeeded(conversationID: conversation.id))
        let updated = try #require(try store.conversation(id: conversation.id))
        #expect(updated.summary == "Discussed messages 0 to 16.")
        #expect(updated.summaryMessageCount == 17)
        let prompt = try #require(llm.requests.first?.first?.content)
        #expect(prompt.contains("PREVIOUS SUMMARY:\n(none)"))
        #expect(prompt.contains("User: message 0\nAssistant: message 1"))
        #expect(prompt.contains("message 16") && !prompt.contains("message 17"))
        #expect(try await summarizer.summarizeIfNeeded(conversationID: conversation.id) == false)
    }
}
