/// Accumulates token bytes and releases only complete UTF-8 text.
///
/// llama.cpp token pieces can split a multibyte character (emoji, accented letters, curly quotes) across tokens;
/// decoding each piece separately would produce U+FFFD. Bytes of an unfinished sequence are held until complete.
public struct UTF8StreamDecoder: Sendable {
    private var pending: [UInt8] = []

    public init() {}

    /// Appends bytes and returns the text that is now complete (possibly empty).
    public mutating func append<Bytes: Sequence>(_ bytes: Bytes) -> String where Bytes.Element == UInt8 {
        pending.append(contentsOf: bytes)
        let complete = Self.completePrefixLength(pending)
        guard complete > 0 else { return "" }
        let text = String(decoding: pending[..<complete], as: UTF8.self)
        pending.removeFirst(complete)
        return text
    }

    /// Flushes whatever is left (invalid remainders become U+FFFD).
    public mutating func finish() -> String {
        defer { pending.removeAll() }
        return pending.isEmpty ? "" : String(decoding: pending, as: UTF8.self)
    }

    /// Length of the prefix that does not end in the middle of a multibyte sequence.
    static func completePrefixLength(_ bytes: [UInt8]) -> Int {
        let n = bytes.count
        // Look back at most 3 bytes for the lead byte of the last sequence.
        var i = n - 1
        while i >= 0, i >= n - 4 {
            let byte = bytes[i]
            if byte & 0b1100_0000 != 0b1000_0000 { // not a continuation byte
                let length: Int
                switch byte {
                case 0..<0x80: length = 1
                case 0xC0..<0xE0: length = 2
                case 0xE0..<0xF0: length = 3
                case 0xF0..<0xF8: length = 4
                default: length = 1 // invalid lead byte: let the decoder replace it
                }
                return i + length <= n ? n : i
            }
            i -= 1
        }
        return n // only continuation bytes: invalid, release them
    }
}
