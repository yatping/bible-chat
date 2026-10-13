import Foundation

/// A validated memory candidate parsed from the extraction model's output.
public struct ExtractedMemory: Sendable, Hashable {
    public var type: MemoryType
    public var key: MemoryKey?
    public var value: String
    public var importance: Double
    public var confidence: Double

    public init(type: MemoryType, key: MemoryKey?, value: String, importance: Double = 0.5, confidence: Double = 1.0) {
        self.type = type
        self.key = key
        self.value = value
        self.importance = importance
        self.confidence = confidence
    }
}

/// Builds the extraction prompt and parses/validates the model's JSON (CONTRACTS §4).
/// The model never writes to the database; only items that pass validation reach `MemoryManager`.
public enum MemoryExtractor {
    public static let maxItems = 5
    public static let maxValueLength = 300
    public static let minConfidence = 0.5

    /// `shared/memory_extraction_prompt.txt` with the user's message inserted.
    public static func prompt(for userMessage: String) -> String {
        var template = SharedTexts.memoryExtractionPrompt
        while template.hasSuffix("\n") { template.removeLast() }
        return template.replacingOccurrences(of: "{{user_message}}",
                                             with: userMessage.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    /// Chat messages for the extraction request (a single user turn).
    public static func messages(for userMessage: String) -> [ChatMessage] {
        [.user(prompt(for: userMessage))]
    }

    /// Parses the first balanced `{…}` object in `raw` and returns the valid items (at most 5).
    public static func parse(_ raw: String) -> [ExtractedMemory] {
        guard let json = firstJSONObject(in: raw),
              let object = try? JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any],
              let items = object["memories"] as? [Any] else { return [] }
        var result: [ExtractedMemory] = []
        for case let item as [String: Any] in items {
            guard let memory = validate(item) else { continue }
            result.append(memory)
            if result.count == maxItems { break }
        }
        return result
    }

    static func validate(_ item: [String: Any]) -> ExtractedMemory? {
        guard let typeName = item["type"] as? String,
              let type = MemoryType(rawValue: typeName.trimmingCharacters(in: .whitespaces).lowercased()) else { return nil }
        let key = (item["key"] as? String).flatMap { MemoryKey(rawValue: $0.trimmingCharacters(in: .whitespaces).lowercased()) }
        guard let rawValue = item["value"] as? String else { return nil }
        let value = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty, value.count <= maxValueLength else { return nil }
        let importance = clamp(number(item["importance"]) ?? 0.5)
        let confidence = clamp(number(item["confidence"]) ?? 1.0)
        guard confidence >= minConfidence else { return nil }
        return ExtractedMemory(type: type, key: key, value: value, importance: importance, confidence: confidence)
    }

    private static func number(_ value: Any?) -> Double? {
        switch value {
        case let n as NSNumber where CFGetTypeID(n) != CFBooleanGetTypeID(): n.doubleValue
        case let s as String: Double(s)
        default: nil
        }
    }

    private static func clamp(_ value: Double) -> Double {
        value.isFinite ? min(max(value, 0), 1) : 0.5
    }

    /// The first balanced `{…}` substring, honouring JSON string literals and escapes.
    static func firstJSONObject(in text: String) -> String? {
        let scalars = Array(text.unicodeScalars)
        guard let start = scalars.firstIndex(of: "{") else { return nil }
        var depth = 0
        var inString = false
        var escaped = false
        for index in start..<scalars.count {
            let c = scalars[index]
            if inString {
                if escaped { escaped = false } else if c == "\\" { escaped = true } else if c == "\"" { inString = false }
                continue
            }
            switch c {
            case "\"": inString = true
            case "{": depth += 1
            case "}":
                depth -= 1
                if depth == 0 {
                    var view = String.UnicodeScalarView()
                    view.append(contentsOf: scalars[start...index])
                    return String(view)
                }
            default: break
            }
        }
        return nil
    }
}
