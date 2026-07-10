import XCTest
import Foundation
import simd
import MLX
import Hy3DMLX

/// Fixture resolver for the parity tests. Resolves the fixtures directory from the `HY3D_FIXTURES`
/// environment variable (settable per test plan), defaulting to `./fixtures`. Any test whose
/// fixture is absent throws `XCTSkip`, so `swift test` is green with zero fixtures present.
struct FixtureStore {
    let dir: String
    init() { dir = ProcessInfo.processInfo.environment["HY3D_FIXTURES"] ?? "./fixtures" }

    func path(_ name: String) -> String { (dir as NSString).appendingPathComponent(name) }
    func exists(_ name: String) -> Bool { FileManager.default.fileExists(atPath: path(name)) }

    /// Load a safetensors fixture, or skip the test if it is not present.
    func require(_ name: String) throws -> [String: MLXArray] {
        guard exists(name) else { throw XCTSkip("fixture not present: \(path(name))") }
        return try loadArrays(url: URL(fileURLWithPath: path(name)))
    }

    /// Resolve a shape checkpoint directory for a slot ("small" = 2mini, "large" = 2.0-turbo).
    /// Uses `HY3D_SHAPE_SMALL` / `HY3D_SHAPE_LARGE`, else `<fixtures>/shape-<slot>`. Skips if absent.
    func shapeWeights(_ slot: String) throws -> URL {
        let env = ProcessInfo.processInfo.environment
        let key = slot == "small" ? "HY3D_SHAPE_SMALL" : "HY3D_SHAPE_LARGE"
        let base = env[key] ?? (dir as NSString).appendingPathComponent("shape-\(slot)")
        let file = (base as NSString).appendingPathComponent("model.fp16.safetensors")
        guard FileManager.default.fileExists(atPath: file) else {
            throw XCTSkip("shape \(slot) weights not present: \(file)")
        }
        return URL(fileURLWithPath: file)
    }
}

/// Flat or [n,3] float array -> SIMD3 vertex list (for Chamfer comparison).
func simd3Array(_ a: MLXArray) -> [SIMD3<Float>] {
    let flat = a.asType(.float32).reshaped([-1]).asArray(Float.self)
    var out = [SIMD3<Float>](); out.reserveCapacity(flat.count / 3)
    var i = 0
    while i + 2 < flat.count {
        out.append(SIMD3(flat[i], flat[i + 1], flat[i + 2])); i += 3
    }
    return out
}

/// Parity metrics (cosine / maxabs / PSNR) — ported from the v1 hy3d-cli parity harness.
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
}

/// Symmetric Chamfer distance normalized by the union's bounding-box diagonal. Vertices are
/// subsampled (deterministic stride) to keep the O(n·m) nearest-neighbour search tractable.
func chamferNormalized(_ a: [SIMD3<Float>], _ b: [SIMD3<Float>], sample: Int = 1500) -> Float {
    func stride(_ v: [SIMD3<Float>]) -> [SIMD3<Float>] {
        guard v.count > sample else { return v }
        let step = v.count / sample
        return (0 ..< sample).map { v[$0 * step] }
    }
    let sa = stride(a), sb = stride(b)
    func meanNN(_ from: [SIMD3<Float>], _ to: [SIMD3<Float>]) -> Float {
        var sum: Float = 0
        for p in from {
            var best = Float.greatestFiniteMagnitude
            for q in to { let d = simd_distance_squared(p, q); if d < best { best = d } }
            sum += best.squareRoot()
        }
        return sum / Float(max(from.count, 1))
    }
    let cd = 0.5 * (meanNN(sa, sb) + meanNN(sb, sa))
    var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
    var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
    for p in (a + b) { lo = simd_min(lo, p); hi = simd_max(hi, p) }
    let diag = simd_distance(lo, hi)
    return diag > 0 ? cd / diag : cd
}
