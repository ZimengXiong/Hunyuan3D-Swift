import XCTest
import Foundation
import MLX
import HunyuanPaintMLX

/// Fixture resolver for the paint parity tests. Resolves the fixtures directory from
/// `HY3D_FIXTURES` (default `./fixtures`). Absent fixtures throw `XCTSkip`, so `swift test`
/// is green with zero fixtures present. Fixture names are documented in parity/README.md.
struct FixtureStore {
    let dir: String
    init() { dir = ProcessInfo.processInfo.environment["HY3D_FIXTURES"] ?? "./fixtures" }

    func path(_ name: String) -> String { (dir as NSString).appendingPathComponent(name) }
    func exists(_ name: String) -> Bool { FileManager.default.fileExists(atPath: path(name)) }

    func require(_ name: String) throws -> [String: MLXArray] {
        guard exists(name) else { throw XCTSkip("fixture not present: \(path(name))") }
        return try loadArrays(url: URL(fileURLWithPath: path(name)))
    }

    /// Load fixture-embedded weights as a `W`, upcast to fp32; skip if absent.
    func requireW(_ name: String) throws -> W {
        guard exists(name) else { throw XCTSkip("fixture not present: \(path(name))") }
        return W((try loadArrays(url: URL(fileURLWithPath: path(name)))).mapValues { $0.asType(.float32) })
    }
}

/// Parity metrics (cosine / maxabs / PSNR) — ported from the v1 paint-cli parity harness.
enum Metric {
    static func cosine(_ got: MLXArray, _ exp: MLXArray) -> Float {
        let a = got.asType(.float32).reshaped([-1]), b = exp.asType(.float32).reshaped([-1])
        let cos = (a * b).sum() / (sqrt((a * a).sum()) * sqrt((b * b).sum()))
        eval(cos); return cos.item(Float.self)
    }
    static func maxabs(_ got: MLXArray, _ exp: MLXArray) -> Float {
        let m = abs(got.asType(.float32) - exp.asType(.float32)).max()
        eval(m); return m.item(Float.self)
    }
    static func psnr(_ got: MLXArray, _ exp: MLXArray) -> Float {
        let a = got.asType(.float32).reshaped([-1]), b = exp.asType(.float32).reshaped([-1])
        let mse = ((a - b) * (a - b)).mean(), rng = b.max() - b.min()
        eval(mse, rng)
        return 10 * log10f((rng.item(Float.self) * rng.item(Float.self)) / mse.item(Float.self))
    }
    /// Fraction of elements that match exactly (for face-id / coverage integer masks).
    static func matchFraction(_ got: MLXArray, _ exp: MLXArray) -> Float {
        let m = (got.asType(.int32) .== exp.asType(.int32)).asType(.float32).mean()
        eval(m); return m.item(Float.self)
    }
}
