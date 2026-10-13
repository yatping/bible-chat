import Foundation

/// A row of `conversations`.
public struct Conversation: Sendable, Hashable, Identifiable {
    public var id: String
    public var title: String?
    public var summary: String?
    public var summaryMessageCount: Int
    public var createdAt: Date
    public var updatedAt: Date

    public init(id: String = UUID().uuidString, title: String? = nil, summary: String? = nil,
                summaryMessageCount: Int = 0, createdAt: Date = .now, updatedAt: Date = .now) {
        self.id = id
        self.title = title
        self.summary = summary
        self.summaryMessageCount = summaryMessageCount
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }
}

/// A row of `messages`.
public struct StoredMessage: Sendable, Hashable, Identifiable {
    public enum Role: String, Sendable, Codable {
        case user, assistant
    }

    public var id: String
    public var conversationID: String
    public var role: Role
    public var content: String
    /// Cited sources (assistant only), persisted as `sources_json`.
    public var sources: [CitedSource]?
    public var createdAt: Date

    public init(id: String = UUID().uuidString, conversationID: String, role: Role, content: String,
                sources: [CitedSource]? = nil, createdAt: Date = .now) {
        self.id = id
        self.conversationID = conversationID
        self.role = role
        self.content = content
        self.sources = sources
        self.createdAt = createdAt
    }

    public var chatMessage: ChatMessage {
        ChatMessage(role: role == .user ? .user : .assistant, content: content)
    }
}

/// `memories.type`.
public enum MemoryType: String, Sendable, Codable, CaseIterable, Hashable {
    case preference, goal, studyInterest = "study_interest", progress, note

    /// Recency half-life in days; nil = no decay (CONTRACTS §4).
    public var halfLifeDays: Double? {
        switch self {
        case .preference, .note: nil
        case .goal: 90
        case .studyInterest: 30
        case .progress: 3
        }
    }

    public var displayName: String {
        switch self {
        case .preference: "Preferences"
        case .goal: "Goals"
        case .studyInterest: "Study interests"
        case .progress: "Progress"
        case .note: "Notes"
        }
    }
}

/// Allowed `memories.memory_key` slots (one active memory per key).
public enum MemoryKey: String, Sendable, Codable, CaseIterable, Hashable {
    case bibleTranslation = "bible_translation"
    case answerStyle = "answer_style"
    case answerLength = "answer_length"
    case explanationLevel = "explanation_level"
    case theologicalTradition = "theological_tradition"
    case currentStudy = "current_study"
    case readingPlan = "reading_plan"
}

/// A row of `memories` (+ its embedding).
public struct MemoryRecord: Sendable, Hashable, Identifiable {
    public var id: String
    public var type: MemoryType
    public var key: MemoryKey?
    public var content: String
    public var importance: Double
    public var confidence: Double
    public var sourceMessageID: String?
    public var createdAt: Date
    public var updatedAt: Date
    public var lastAccessedAt: Date?
    public var superseded: Bool
    public var supersededBy: String?
    public var embedding: [Float]?

    public init(id: String = UUID().uuidString, type: MemoryType, key: MemoryKey?, content: String,
                importance: Double = 0.5, confidence: Double = 1.0, sourceMessageID: String? = nil,
                createdAt: Date = .now, updatedAt: Date? = nil, lastAccessedAt: Date? = nil,
                superseded: Bool = false, supersededBy: String? = nil, embedding: [Float]? = nil) {
        self.id = id
        self.type = type
        self.key = key
        self.content = content
        self.importance = importance
        self.confidence = confidence
        self.sourceMessageID = sourceMessageID
        self.createdAt = createdAt
        self.updatedAt = updatedAt ?? createdAt
        self.lastAccessedAt = lastAccessedAt
        self.superseded = superseded
        self.supersededBy = supersededBy
        self.embedding = embedding
    }
}

extension Date {
    var unixSeconds: Double { timeIntervalSince1970 }
    init(unixSeconds: Double) { self.init(timeIntervalSince1970: unixSeconds) }
}
