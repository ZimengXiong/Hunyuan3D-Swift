import Foundation
import MLX
import Hy3DMLX
import HunyuanPaintMLX

// MARK: - hy3d generate (chained shape -> paint; the flagship demo)

func cmdGenerate(_ args: Args) throws {
    guard let imagePath = args.positional.first else { throw CLIError("generate: missing <image.png>") }
    guard let out = args.str("o", "output") else { throw CLIError("generate: missing -o <out.glb>") }
    guard let shapeW = args.str("shape-weights") else { throw CLIError("generate: missing --shape-weights <dir>") }
    guard let paintW = args.str("paint-weights") else { throw CLIError("generate: missing --paint-weights <dir>") }
    guard let shapeURL = resolveShapeWeights(shapeW) else {
        throw CLIError("generate: no .safetensors checkpoint under \(shapeW)")
    }
    guard let cg = loadCGImage(imagePath) else { throw CLIError("generate: cannot read image \(imagePath)") }

    // shape knobs
    let steps = args.int("steps") ?? 30
    let guidance = args.float("guidance") ?? 5.0
    let resolution = args.int("octree") ?? 256
    let quantize = args.int("quantize") ?? 0
    let seed = UInt64(args.int("seed") ?? 0)
    // paint knobs
    let paintModel = (args.str("paint-model") ?? "rgb").lowercased()
    let paintSteps = args.int("paint-steps") ?? 15
    let res = args.int("res") ?? 512
    let tex = args.int("tex") ?? (paintModel == "pbr" ? 4096 : 2048)
    let superRes = !args.flag("no-superres")

    // ---- 1. shape ----
    print("generate[1/2] shape: \(shapeURL.lastPathComponent)  steps=\(steps) octree=\(resolution)")
    let gen = try ShapeGenerator(weightsURL: shapeURL, quantize: quantize)
    guard let mesh = gen.generate(image: cg, steps: steps, guidance: guidance, seed: seed,
                                  resolution: resolution, octree: true, onProgress: { p in
        print(String(format: "  [%3.0f%%] %@", p.fraction * 100, p.stage))
    }) else { throw CLIError("generate: shape stage failed (image preprocessing?)") }
    print("generate[1/2] shape mesh: \(mesh.vertices.count) verts, \(mesh.faces.count) faces")

    // ---- 2. paint ----
    print("generate[2/2] paint (\(paintModel)): res=\(res) steps=\(paintSteps) tex=\(tex) super-res=\(superRes)")
    let pipe = PaintPipeline(weightsRoot: paintW, res: res, steps: paintSteps, tex: tex, superRes: superRes)
    switch paintModel {
    case "pbr":
        // The PBR path writes the GLB itself; hand it the shape mesh via a temp .glb.
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("hy3d_shape_\(UUID().uuidString).glb")
        try GLB.write(mesh, to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }
        try pipe.run(meshPath: tmp.path, imagePath: imagePath, outGLB: out, fixturesDir: "")
    case "rgb":
        guard let r = try pipe.paintRGB(mesh: flatten(mesh), imagePath: imagePath, onProgress: { s, f in
            print(String(format: "  [%3.0f%%] %@", f * 100, s))
        }) else { throw CLIError("generate: paint stage returned no result (UV unwrap failed?)") }
        try writeGLB(path: out, vertices: r.vertices, faces: r.faces, uvs: r.uvs,
                     baseColorPNG: r.albedoPNG, metallicRoughnessPNG: nil)
    default:
        throw CLIError("generate: --paint-model must be rgb or pbr")
    }
    print("generate: wrote \(out)")
}
