import Foundation
import CoreGraphics
import ImageIO
import MLX
import Hy3DMLX
import HunyuanPaintMLX

// MARK: - Argument parsing (hand-rolled; no external deps)

/// Minimal `--key value` / `-k value` / `--bool-flag` parser. `bools` names the value-less flags;
/// any option not in `bools` consumes the following token as its value. Bare tokens are positional.
struct Args {
    private(set) var positional: [String] = []
    private var options: [String: String] = [:]
    private var boolFlags: Set<String> = []

    init(_ argv: [String], bools: Set<String> = []) {
        var i = 0
        while i < argv.count {
            let a = argv[i]
            if a.hasPrefix("-"), a != "-", Double(a) == nil {
                let name = String(a.drop(while: { $0 == "-" }))
                if bools.contains(name) {
                    boolFlags.insert(name); i += 1
                } else if i + 1 < argv.count {
                    options[name] = argv[i + 1]; i += 2
                } else {
                    boolFlags.insert(name); i += 1        // trailing flag with no value
                }
            } else {
                positional.append(a); i += 1
            }
        }
    }

    func str(_ names: String...) -> String? { for n in names where options[n] != nil { return options[n] }; return nil }
    func int(_ names: String...) -> Int? { for n in names { if let v = options[n] { return Int(v) } }; return nil }
    func float(_ names: String...) -> Float? { for n in names { if let v = options[n] { return Float(v) } }; return nil }
    func flag(_ name: String) -> Bool { boolFlags.contains(name) }
}

/// CLI error with a clean message (no stack trace) for the top-level catch.
struct CLIError: Error, CustomStringConvertible {
    let message: String
    init(_ m: String) { message = m }
    var description: String { message }
}

func die(_ message: String) -> Never {
    FileHandle.standardError.write("error: \(message)\n".data(using: .utf8)!)
    exit(1)
}

// MARK: - Parity metrics (shared by the CLI print-panels)

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
}

// MARK: - Fixture directory (--fixtures <dir> or env HY3D_FIXTURES, default ./fixtures)

struct FixtureStore {
    let dir: String
    init(_ args: Args) {
        dir = args.str("fixtures")
            ?? ProcessInfo.processInfo.environment["HY3D_FIXTURES"]
            ?? "./fixtures"
    }
    func path(_ name: String) -> String { (dir as NSString).appendingPathComponent(name) }
    func exists(_ name: String) -> Bool { FileManager.default.fileExists(atPath: path(name)) }
    func anyExists(_ names: [String]) -> Bool { names.contains { exists($0) } }
    func load(_ name: String) throws -> [String: MLXArray] {
        try loadArrays(url: URL(fileURLWithPath: path(name)))
    }
}

// MARK: - Image / mesh helpers

func loadCGImage(_ path: String) -> CGImage? {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
    return img
}

/// Resolve a shape checkpoint file from a `--weights <dir>`: prefer `model.fp16.safetensors`,
/// else the first `*.safetensors` in the directory, else the path itself if it is a `.safetensors`.
func resolveShapeWeights(_ dir: String) -> URL? {
    let fm = FileManager.default
    if dir.hasSuffix(".safetensors"), fm.fileExists(atPath: dir) { return URL(fileURLWithPath: dir) }
    let base = URL(fileURLWithPath: dir, isDirectory: true)
    let preferred = base.appendingPathComponent("model.fp16.safetensors")
    if fm.fileExists(atPath: preferred.path) { return preferred }
    if let items = try? fm.contentsOfDirectory(at: base, includingPropertiesForKeys: nil),
       let st = items.first(where: { $0.pathExtension == "safetensors" }) { return st }
    return nil
}

/// Flatten a shape `Mesh` (SIMD3 verts + tuple faces) into the paint `LoadedMesh` layout.
func flatten(_ m: Mesh) -> LoadedMesh {
    var v = [Float](); v.reserveCapacity(m.vertices.count * 3)
    for p in m.vertices { v.append(p.x); v.append(p.y); v.append(p.z) }
    var f = [UInt32](); f.reserveCapacity(m.faces.count * 3)
    for t in m.faces { f.append(t.0); f.append(t.1); f.append(t.2) }
    return LoadedMesh(vertices: v, faces: f)
}

// MARK: - Usage

func printUsage() {
    print("""
    hy3d — Hunyuan3D shape + paint, native MLX-Swift.

    USAGE:
      hy3d shape    <image.png> -o <out.glb> --weights <dir>
                    [--steps N] [--guidance F] [--octree N] [--quantize 4|8] [--seed N]

      hy3d paint    <mesh.glb|obj> <image.png> -o <out.glb> --weights <dir>
                    [--model rgb|pbr] [--res N] [--steps N] [--tex N] [--no-superres] [--seed N]

      hy3d generate <image.png> -o <out.glb> --shape-weights <dir> --paint-weights <dir>
                    [--paint-model rgb|pbr] [--steps N] [--guidance F] [--octree N]
                    [--quantize 4|8] [--paint-steps N] [--res N] [--tex N] [--no-superres] [--seed N]

      hy3d parity-shape [--fixtures <dir>] [--weights <dir>] [--weights-turbo <dir>]
      hy3d parity-paint [--fixtures <dir>]

    Fixture directory for the parity panels resolves from --fixtures, else $HY3D_FIXTURES,
    else ./fixtures.

    Notes:
      • shape --weights points at a checkpoint directory (model.fp16.safetensors + config.yaml).
      • paint --weights points at a weights root containing hunyuan3d-paint-v2-0/,
        hunyuan3d-paintpbr-v2-1/, dinov2-giant/, realesrgan/.
      • --octree N is the SDF grid resolution (octree decode). Default 256.
      • paint --seed is accepted for interface parity; the paint pipeline currently re-seeds its
        RNG to 0 internally, so it has no effect there yet. shape/generate honor --seed.
    """)
}
