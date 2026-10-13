import Foundation

/// Removes Gemma 4 thought-channel blocks (`<|channel>thought…<channel|>`) from streamed model output.
///
/// Even with thinking disabled in the chat template, base Gemma 4 E2B sometimes opens a thought channel before
/// answering. The fine-tune is trained on plain answers, but the app must never show (or store, or parse as memory
/// JSON) reasoning text, so engines pass every decoded piece through this filter. Markers may be split across
/// pieces; a possible partial marker at the end of the buffer is held back until the next piece decides it.
public struct ThoughtChannelFilter: Sendable {
    public static let openMarker = "<|channel>"
    public static let closeMarker = "<channel|>"

    private var buffer = ""
    private var inThought = false

    public init() {}

    /// Feeds one decoded piece and returns the text that is safe to show now.
    public mutating func feed(_ piece: String) -> String {
        buffer += piece
        var visible = ""
        while true {
            let marker = inThought ? Self.closeMarker : Self.openMarker
            if let range = buffer.range(of: marker) {
                if !inThought { visible += buffer[..<range.lowerBound] }
                buffer.removeSubrange(..<range.upperBound)
                inThought.toggle()
                continue
            }
            let held = Self.partialMarkerSuffixLength(buffer, marker: marker)
            let end = buffer.index(buffer.endIndex, offsetBy: -held)
            if !inThought { visible += buffer[..<end] }
            buffer.removeSubrange(..<end)
            return visible
        }
    }

    /// Flushes held-back text at the end of generation. An unterminated thought block is dropped.
    public mutating func finish() -> String {
        defer { buffer = "" }
        return inThought ? "" : buffer
    }

    /// Filters a complete string in one go.
    public static func strip(_ text: String) -> String {
        var filter = ThoughtChannelFilter()
        return filter.feed(text) + filter.finish()
    }

    private static func partialMarkerSuffixLength(_ text: String, marker: String) -> Int {
        for length in stride(from: min(marker.count - 1, text.count), through: 1, by: -1)
        where text.hasSuffix(marker.prefix(length)) {
            return length
        }
        return 0
    }
}
