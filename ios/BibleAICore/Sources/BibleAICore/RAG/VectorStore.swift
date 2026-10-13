import Accelerate
import Foundation

/// Dense document vectors from `rag_embeddings.f16` (CONTRACTS §2): raw little-endian binary16, row-major
/// `count × dim`, no header.
///
/// The file is memory-mapped and converted once to a Float32 matrix with vImage (≈ count × dim × 4 bytes,
/// e.g. 30k × 256 → 30 MB). All scores are then one matrix–vector product (`vDSP_mmul`).
public final class VectorStore: Sendable {
    public enum Error: Swift.Error, CustomStringConvertible {
        case sizeMismatch(expected: Int, actual: Int)
        case dimensionMismatch(expected: Int, actual: Int)

        public var description: String {
            switch self {
            case .sizeMismatch(let e, let a): "Embedding file has \(a) bytes, expected \(e)."
            case .dimensionMismatch(let e, let a): "Query vector has \(a) dimensions, expected \(e)."
            }
        }
    }

    public let count: Int
    public let dimension: Int
    private let matrix: [Float]

    /// Memory-maps `url` and converts it to Float32.
    public convenience init(url: URL, count: Int, dimension: Int) throws {
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        try self.init(float16Data: data, count: count, dimension: dimension)
    }

    /// Converts in-memory binary16 data.
    public init(float16Data data: Data, count: Int, dimension: Int) throws {
        let expected = count * dimension * 2
        guard data.count == expected else { throw Error.sizeMismatch(expected: expected, actual: data.count) }
        self.count = count
        self.dimension = dimension
        let elements = count * dimension
        var floats = [Float](repeating: 0, count: elements)
        if elements > 0 {
            data.withUnsafeBytes { raw in
                floats.withUnsafeMutableBytes { out in
                    var source = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: raw.baseAddress!),
                                               height: 1, width: vImagePixelCount(elements), rowBytes: elements * 2)
                    var destination = vImage_Buffer(data: out.baseAddress!, height: 1,
                                                    width: vImagePixelCount(elements), rowBytes: elements * 4)
                    vImageConvert_Planar16FtoPlanarF(&source, &destination, vImage_Flags(kvImageNoFlags))
                }
            }
        }
        matrix = floats
    }

    /// Dot product of `query` with every row.
    public func scores(for query: [Float]) throws -> [Float] {
        guard query.count == dimension else { throw Error.dimensionMismatch(expected: dimension, actual: query.count) }
        var result = [Float](repeating: 0, count: count)
        guard count > 0 else { return result }
        matrix.withUnsafeBufferPointer { a in
            query.withUnsafeBufferPointer { b in
                result.withUnsafeMutableBufferPointer { c in
                    // C (count×1) = A (count×dim) · B (dim×1)
                    vDSP_mmul(a.baseAddress!, 1, b.baseAddress!, 1, c.baseAddress!, 1,
                              vDSP_Length(count), 1, vDSP_Length(dimension))
                }
            }
        }
        return result
    }

    /// The `k` best rows (highest score first, ties by lower row).
    public func topK(_ query: [Float], k: Int) throws -> [(row: Int, score: Float)] {
        Self.topK(try scores(for: query), k: k)
    }

    /// Partial selection of the `k` largest scores.
    static func topK(_ scores: [Float], k: Int) -> [(row: Int, score: Float)] {
        guard k > 0 else { return [] }
        var best: [(row: Int, score: Float)] = []
        best.reserveCapacity(k + 1)
        for (row, score) in scores.enumerated() where !score.isNaN {
            if best.count == k, let last = best.last, score <= last.score { continue }
            let index = best.firstIndex { score > $0.score } ?? best.count
            best.insert((row, score), at: index)
            if best.count > k { best.removeLast() }
        }
        return best
    }

    /// Encodes Float32 vectors as raw little-endian binary16 (the `rag_embeddings.f16` layout). Used by tests/tools.
    public static func float16Data(from vectors: [[Float]]) -> Data {
        var flat = vectors.flatMap { $0 }
        var data = Data(count: flat.count * 2)
        guard !flat.isEmpty else { return data }
        let elements = flat.count
        flat.withUnsafeMutableBytes { input in
            data.withUnsafeMutableBytes { output in
                var source = vImage_Buffer(data: input.baseAddress!, height: 1, width: vImagePixelCount(elements), rowBytes: elements * 4)
                var destination = vImage_Buffer(data: output.baseAddress!, height: 1, width: vImagePixelCount(elements), rowBytes: elements * 2)
                vImageConvert_PlanarFtoPlanar16F(&source, &destination, vImage_Flags(kvImageNoFlags))
            }
        }
        return data
    }
}
