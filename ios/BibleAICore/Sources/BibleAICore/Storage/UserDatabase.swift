import Foundation

/// The writable `user.sqlite` (schema: `shared/schema_user.sql`, embedded as `SharedTexts.userSchema`).
public final class UserDatabase: Sendable {
    public static let schemaVersion = 1

    public let db: SQLiteDatabase

    /// Opens or creates the database at `url` and applies the schema if needed.
    public convenience init(url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try self.init(database: SQLiteDatabase(path: url.path))
        try? db.execute("PRAGMA journal_mode = WAL")
    }

    /// An in-memory database (tests, previews).
    public static func inMemory() throws -> UserDatabase {
        try UserDatabase(database: SQLiteDatabase())
    }

    init(database: SQLiteDatabase) throws {
        db = database
        try db.execute("PRAGMA foreign_keys = ON")
        let version = try db.queryOne("PRAGMA user_version")?.values.first
        if version == .integer(0) || version == nil {
            try db.transaction { try db.execute(SharedTexts.userSchema) }
            try db.execute("PRAGMA foreign_keys = ON")
        }
    }
}
