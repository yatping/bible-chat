import Foundation

/// Downloadable model files (shape of `shared/model_manifest.example.json`).
public struct ModelManifest: Codable, Sendable, Hashable {
    public var version: Int
    public var models: [Entry]

    public enum Role: String, Codable, Sendable, Hashable {
        case generator, embedder
    }

    public struct Entry: Codable, Sendable, Hashable, Identifiable {
        public var id: String
        public var role: Role
        public var filename: String
        public var url: URL
        /// Lowercase hex sha256 of the file.
        public var sha256: String
        /// Expected size in bytes (0 = unknown).
        public var bytes: Int64
        public var contextLength: Int?
        public var embeddingDim: Int?
        public var truncateDim: Int?

        public init(id: String, role: Role, filename: String, url: URL, sha256: String, bytes: Int64,
                    contextLength: Int? = nil, embeddingDim: Int? = nil, truncateDim: Int? = nil) {
            self.id = id
            self.role = role
            self.filename = filename
            self.url = url
            self.sha256 = sha256
            self.bytes = bytes
            self.contextLength = contextLength
            self.embeddingDim = embeddingDim
            self.truncateDim = truncateDim
        }

        enum CodingKeys: String, CodingKey {
            case id, role, filename, url, sha256, bytes
            case contextLength = "context_length"
            case embeddingDim = "embedding_dim"
            case truncateDim = "truncate_dim"
        }

        /// True when `sha256` holds a real 64-char hex digest (the example manifest has a placeholder).
        public var hasChecksum: Bool {
            sha256.count == 64 && sha256.allSatisfy(\.isHexDigit)
        }
    }

    public init(version: Int, models: [Entry]) {
        self.version = version
        self.models = models
    }

    public func model(role: Role) -> Entry? {
        models.first { $0.role == role }
    }
}
