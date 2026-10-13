import Foundation
import SQLite3

/// A value bound to, or read from, an SQLite statement.
public enum SQLValue: Sendable, Equatable {
    case null
    case integer(Int64)
    case real(Double)
    case text(String)
    case blob(Data)
}

/// Types that can be bound as statement parameters.
public protocol SQLValueConvertible: Sendable {
    var sqlValue: SQLValue { get }
}

extension SQLValue: SQLValueConvertible { public var sqlValue: SQLValue { self } }
extension String: SQLValueConvertible { public var sqlValue: SQLValue { .text(self) } }
extension Int: SQLValueConvertible { public var sqlValue: SQLValue { .integer(Int64(self)) } }
extension Int64: SQLValueConvertible { public var sqlValue: SQLValue { .integer(self) } }
extension Double: SQLValueConvertible { public var sqlValue: SQLValue { .real(self) } }
extension Data: SQLValueConvertible { public var sqlValue: SQLValue { .blob(self) } }
extension Bool: SQLValueConvertible { public var sqlValue: SQLValue { .integer(self ? 1 : 0) } }
extension Optional: SQLValueConvertible where Wrapped: SQLValueConvertible {
    public var sqlValue: SQLValue { self?.sqlValue ?? .null }
}

/// An SQLite error with the extended result code and message.
public struct SQLiteError: Error, CustomStringConvertible, Sendable {
    public let code: Int32
    public let message: String
    public let sql: String?

    public var description: String {
        "SQLite error \(code): \(message)" + (sql.map { " [\($0)]" } ?? "")
    }
}

/// One result row. Columns are addressable by index or name.
public struct SQLRow: Sendable {
    public let values: [SQLValue]
    let columnIndex: [String: Int]

    public subscript(_ column: String) -> SQLValue {
        guard let index = columnIndex[column] else { return .null }
        return values[index]
    }

    public func int(_ column: String) -> Int? {
        switch self[column] {
        case .integer(let v): Int(v)
        case .real(let v): Int(v)
        case .text(let s): Int(s)
        default: nil
        }
    }

    public func int64(_ column: String) -> Int64? {
        if case .integer(let v) = self[column] { return v }
        return int(column).map(Int64.init)
    }

    public func double(_ column: String) -> Double? {
        switch self[column] {
        case .real(let v): v
        case .integer(let v): Double(v)
        case .text(let s): Double(s)
        default: nil
        }
    }

    public func string(_ column: String) -> String? {
        switch self[column] {
        case .text(let s): s
        case .integer(let v): String(v)
        case .real(let v): String(v)
        default: nil
        }
    }

    public func data(_ column: String) -> Data? {
        switch self[column] {
        case .blob(let d): d
        case .text(let s): Data(s.utf8)
        default: nil
        }
    }
}

/// A thread-safe SQLite connection with a prepared-statement cache.
///
/// All access goes through a recursive lock, so one instance can be shared between actors and tasks.
/// Statements are prepared once per SQL string and reused.
public final class SQLiteDatabase: @unchecked Sendable {
    private var handle: OpaquePointer?
    private var statements: [String: OpaquePointer] = [:]
    private let lock = NSRecursiveLock()

    /// Opens (or creates) the database at `path`.
    /// - Parameters:
    ///   - readOnly: opens with `SQLITE_OPEN_READONLY`.
    ///   - immutable: for bundled read-only files; tells SQLite the file never changes (no locking/journal).
    public init(path: String, readOnly: Bool = false, immutable: Bool = false) throws {
        var flags = SQLITE_OPEN_FULLMUTEX | SQLITE_OPEN_URI
        flags |= readOnly ? SQLITE_OPEN_READONLY : (SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE)
        var target = path
        if immutable {
            var components = URLComponents()
            components.scheme = "file"
            components.path = path
            components.queryItems = [URLQueryItem(name: "immutable", value: "1")]
            target = components.string ?? path
        }
        var db: OpaquePointer?
        let rc = sqlite3_open_v2(target, &db, flags, nil)
        guard rc == SQLITE_OK, let db else {
            let message = db.map { String(cString: sqlite3_errmsg($0)) } ?? "cannot open \(path)"
            sqlite3_close(db)
            throw SQLiteError(code: rc, message: message, sql: nil)
        }
        handle = db
        sqlite3_extended_result_codes(db, 1)
        sqlite3_busy_timeout(db, 5_000)
    }

    /// An in-memory database (tests, previews).
    public convenience init() throws {
        try self.init(path: ":memory:")
    }

    deinit {
        for statement in statements.values { sqlite3_finalize(statement) }
        sqlite3_close_v2(handle)
    }

    /// Runs one or more SQL statements without parameters (schema scripts, pragmas).
    public func execute(_ sql: String) throws {
        try locked {
            var error: UnsafeMutablePointer<CChar>?
            let rc = sqlite3_exec(handle, sql, nil, nil, &error)
            if rc != SQLITE_OK {
                let message = error.map { String(cString: $0) } ?? "unknown error"
                sqlite3_free(error)
                throw SQLiteError(code: rc, message: message, sql: sql)
            }
        }
    }

    /// Runs a single statement that returns no rows. Returns the number of changed rows.
    @discardableResult
    public func run(_ sql: String, _ parameters: [any SQLValueConvertible] = []) throws -> Int {
        try locked {
            let statement = try prepared(sql, parameters)
            defer { sqlite3_reset(statement) }
            var rc = sqlite3_step(statement)
            while rc == SQLITE_ROW { rc = sqlite3_step(statement) }
            guard rc == SQLITE_DONE else { throw lastError(rc, sql) }
            return Int(sqlite3_changes(handle))
        }
    }

    /// Runs a query and returns all rows.
    public func query(_ sql: String, _ parameters: [any SQLValueConvertible] = []) throws -> [SQLRow] {
        try locked {
            let statement = try prepared(sql, parameters)
            defer { sqlite3_reset(statement) }
            let columnCount = sqlite3_column_count(statement)
            var columnIndex: [String: Int] = [:]
            for i in 0..<columnCount {
                columnIndex[String(cString: sqlite3_column_name(statement, i))] = Int(i)
            }
            var rows: [SQLRow] = []
            while true {
                let rc = sqlite3_step(statement)
                if rc == SQLITE_DONE { break }
                guard rc == SQLITE_ROW else { throw lastError(rc, sql) }
                var values: [SQLValue] = []
                values.reserveCapacity(Int(columnCount))
                for i in 0..<columnCount { values.append(columnValue(statement, i)) }
                rows.append(SQLRow(values: values, columnIndex: columnIndex))
            }
            return rows
        }
    }

    /// First row of a query, if any.
    public func queryOne(_ sql: String, _ parameters: [any SQLValueConvertible] = []) throws -> SQLRow? {
        try query(sql, parameters).first
    }

    /// Runs `body` inside `BEGIN IMMEDIATE … COMMIT`, rolling back if it throws.
    /// Nested calls join the outer transaction.
    public func transaction<T>(_ body: () throws -> T) throws -> T {
        try locked {
            if sqlite3_get_autocommit(handle) == 0 { return try body() }
            try execute("BEGIN IMMEDIATE")
            do {
                let result = try body()
                try execute("COMMIT")
                return result
            } catch {
                try? execute("ROLLBACK")
                throw error
            }
        }
    }

    // MARK: - Private

    private func locked<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }

    private func prepared(_ sql: String, _ parameters: [any SQLValueConvertible]) throws -> OpaquePointer {
        let statement: OpaquePointer
        if let cached = statements[sql] {
            statement = cached
        } else {
            var raw: OpaquePointer?
            let rc = sqlite3_prepare_v3(handle, sql, -1, UInt32(SQLITE_PREPARE_PERSISTENT), &raw, nil)
            guard rc == SQLITE_OK, let raw else { throw lastError(rc, sql) }
            statements[sql] = raw
            statement = raw
        }
        sqlite3_clear_bindings(statement)
        let expected = Int(sqlite3_bind_parameter_count(statement))
        guard expected == parameters.count else {
            throw SQLiteError(code: SQLITE_RANGE, message: "expected \(expected) parameters, got \(parameters.count)", sql: sql)
        }
        for (offset, parameter) in parameters.enumerated() {
            let index = Int32(offset + 1)
            let rc: Int32
            switch parameter.sqlValue {
            case .null: rc = sqlite3_bind_null(statement, index)
            case .integer(let v): rc = sqlite3_bind_int64(statement, index, v)
            case .real(let v): rc = sqlite3_bind_double(statement, index, v)
            case .text(let s): rc = sqlite3_bind_text(statement, index, s, -1, sqliteTransient)
            case .blob(let d):
                rc = d.withUnsafeBytes { buffer in
                    sqlite3_bind_blob(statement, index, buffer.baseAddress, Int32(buffer.count), sqliteTransient)
                }
            }
            guard rc == SQLITE_OK else { throw lastError(rc, sql) }
        }
        return statement
    }

    private func columnValue(_ statement: OpaquePointer, _ i: Int32) -> SQLValue {
        switch sqlite3_column_type(statement, i) {
        case SQLITE_INTEGER: return .integer(sqlite3_column_int64(statement, i))
        case SQLITE_FLOAT: return .real(sqlite3_column_double(statement, i))
        case SQLITE_TEXT: return .text(String(cString: sqlite3_column_text(statement, i)))
        case SQLITE_BLOB:
            let count = Int(sqlite3_column_bytes(statement, i))
            guard count > 0, let bytes = sqlite3_column_blob(statement, i) else { return .blob(Data()) }
            return .blob(Data(bytes: bytes, count: count))
        default: return .null
        }
    }

    private func lastError(_ rc: Int32, _ sql: String) -> SQLiteError {
        SQLiteError(code: rc, message: String(cString: sqlite3_errmsg(handle)), sql: sql)
    }
}

/// `SQLITE_TRANSIENT`: SQLite copies the bound buffer.
private let sqliteTransient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
