import Foundation

/// Conversations and messages in `user.sqlite`.
public struct ConversationStore: Sendable {
    public let database: UserDatabase
    private var db: SQLiteDatabase { database.db }

    public init(database: UserDatabase) {
        self.database = database
    }

    @discardableResult
    public func createConversation(title: String? = nil, now: Date = .now) throws -> Conversation {
        let conversation = Conversation(title: title, createdAt: now, updatedAt: now)
        try db.run("INSERT INTO conversations (id, title, summary, summary_message_count, created_at, updated_at) VALUES (?, ?, NULL, 0, ?, ?)",
                   [conversation.id, title, now.unixSeconds, now.unixSeconds])
        return conversation
    }

    /// All conversations, most recently updated first.
    public func conversations() throws -> [Conversation] {
        try db.query("SELECT * FROM conversations ORDER BY updated_at DESC").map(Self.conversation)
    }

    public func conversation(id: String) throws -> Conversation? {
        try db.queryOne("SELECT * FROM conversations WHERE id = ?", [id]).map(Self.conversation)
    }

    public func rename(_ id: String, title: String) throws {
        try db.run("UPDATE conversations SET title = ? WHERE id = ?", [title, id])
    }

    /// Deletes a conversation and (by cascade) its messages.
    public func delete(_ id: String) throws {
        try db.run("DELETE FROM conversations WHERE id = ?", [id])
    }

    /// Stores the rolling summary and how many messages (oldest first) it covers.
    public func updateSummary(_ id: String, summary: String, coveredMessageCount: Int) throws {
        try db.run("UPDATE conversations SET summary = ?, summary_message_count = ? WHERE id = ?",
                   [summary, coveredMessageCount, id])
    }

    /// Appends a message, bumps `updated_at` and sets the title from the first user message.
    @discardableResult
    public func append(_ message: StoredMessage) throws -> StoredMessage {
        let sourcesJSON: String? = try message.sources.map { sources in
            String(decoding: try JSONEncoder().encode(sources), as: UTF8.self)
        }
        try db.transaction {
            try db.run("INSERT INTO messages (id, conversation_id, role, content, sources_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                       [message.id, message.conversationID, message.role.rawValue, message.content, sourcesJSON,
                        message.createdAt.unixSeconds])
            try db.run("UPDATE conversations SET updated_at = ? WHERE id = ?", [message.createdAt.unixSeconds, message.conversationID])
            if message.role == .user {
                try db.run("UPDATE conversations SET title = ? WHERE id = ? AND title IS NULL",
                           [Self.title(from: message.content), message.conversationID])
            }
        }
        return message
    }

    /// Messages of a conversation, oldest first.
    public func messages(in conversationID: String) throws -> [StoredMessage] {
        try db.query("SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at, rowid", [conversationID])
            .map(Self.message)
    }

    public func messageCount(in conversationID: String) throws -> Int {
        try db.queryOne("SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?", [conversationID])?.int("n") ?? 0
    }

    static func title(from question: String) -> String {
        let line = question.trimmingCharacters(in: .whitespacesAndNewlines).components(separatedBy: .newlines).first ?? ""
        return line.count > 60 ? String(line.prefix(59)) + "…" : line
    }

    private static func conversation(_ row: SQLRow) -> Conversation {
        Conversation(id: row.string("id") ?? "", title: row.string("title"), summary: row.string("summary"),
                     summaryMessageCount: row.int("summary_message_count") ?? 0,
                     createdAt: Date(unixSeconds: row.double("created_at") ?? 0),
                     updatedAt: Date(unixSeconds: row.double("updated_at") ?? 0))
    }

    private static func message(_ row: SQLRow) -> StoredMessage {
        let sources = row.data("sources_json").flatMap { try? JSONDecoder().decode([CitedSource].self, from: $0) }
        return StoredMessage(id: row.string("id") ?? "", conversationID: row.string("conversation_id") ?? "",
                             role: StoredMessage.Role(rawValue: row.string("role") ?? "") ?? .user,
                             content: row.string("content") ?? "", sources: sources,
                             createdAt: Date(unixSeconds: row.double("created_at") ?? 0))
    }
}
