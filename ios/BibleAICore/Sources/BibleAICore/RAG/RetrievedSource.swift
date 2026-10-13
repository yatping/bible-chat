/// A source shown to the model and the user, numbered `S1…Sn` by the app.
public struct RetrievedSource: Sendable, Hashable, Identifiable, Codable {
    /// "S1", "S2", …
    public var id: String
    public var chunk: Chunk

    public init(id: String, chunk: Chunk) {
        self.id = id
        self.chunk = chunk
    }

    public var reference: String { chunk.reference }
    public var sourceType: String { chunk.sourceType }
    public var text: String { chunk.text }

    public var promptSource: PromptSource {
        PromptSource(type: chunk.sourceType, reference: chunk.reference, text: chunk.text)
    }

    public var cited: CitedSource {
        CitedSource(id: id, reference: chunk.reference, type: chunk.sourceType, chunkID: chunk.id)
    }

    /// Numbers chunks `S1…Sn` in order.
    public static func numbered(_ chunks: [Chunk]) -> [RetrievedSource] {
        chunks.enumerated().map { RetrievedSource(id: "S\($0.offset + 1)", chunk: $0.element) }
    }
}

/// Element of `messages.sources_json`: `{id, reference, type, chunk_id}`.
public struct CitedSource: Sendable, Hashable, Identifiable, Codable {
    public var id: String
    public var reference: String
    public var type: String
    public var chunkID: Int64

    public init(id: String, reference: String, type: String, chunkID: Int64) {
        self.id = id
        self.reference = reference
        self.type = type
        self.chunkID = chunkID
    }

    enum CodingKeys: String, CodingKey {
        case id, reference, type
        case chunkID = "chunk_id"
    }
}
