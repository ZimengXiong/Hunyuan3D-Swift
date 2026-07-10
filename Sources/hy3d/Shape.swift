import Foundation
import MLX
import Hy3DMLX

/// Build DiT / VAE / DINOv2 from a checkpoint directory. Loads at fp32 by default (the parity
/// fixtures are fp32); the scale_factor audit fix flows through ShapeGenerator.
func loadShapeModels(_ wURL: URL, dtype: DType = .float32) throws -> (DiT, VAE, DINOv2) {
    let g = try ShapeGenerator(weightsURL: wURL, dtype: dtype)
    return (g.dit, g.vae, g.dino)
}

// MARK: - hy3d shape

func cmdShape(_ args: Args) throws {
    guard let imagePath = args.positional.first else { throw CLIError("shape: missing <image.png>") }
    guard let out = args.str("o", "output") else { throw CLIError("shape: missing -o <out.glb>") }
    guard let weightsDir = args.str("weights") else { throw CLIError("shape: missing --weights <dir>") }
    guard let wURL = resolveShapeWeights(weightsDir) else {
        throw CLIError("shape: no .safetensors checkpoint under \(weightsDir)")
    }
    guard let cg = loadCGImage(imagePath) else { throw CLIError("shape: cannot read image \(imagePath)") }

    let steps = args.int("steps") ?? 30
    let guidance = args.float("guidance") ?? 5.0
    let resolution = args.int("octree") ?? 256
    let quantize = args.int("quantize") ?? 0
    let seed = UInt64(args.int("seed") ?? 0)

    print("shape: \(wURL.lastPathComponent) (\(quantize == 0 ? "fp16" : "\(quantize)-bit"))  steps=\(steps) guidance=\(guidance) octree=\(resolution)")
    let gen = try ShapeGenerator(weightsURL: wURL, quantize: quantize)
    let t0 = Date()
    let (v, f) = try gen.generateGLB(image: cg, to: URL(fileURLWithPath: out),
                                     steps: steps, guidance: guidance, seed: seed,
                                     resolution: resolution, octree: true) { p in
        print(String(format: "  [%3.0f%%] %@", p.fraction * 100, p.stage))
    }
    print(String(format: "shape: %d verts, %d faces in %.1fs -> %@", v, f, -t0.timeIntervalSinceNow, out))
}

// MARK: - hy3d parity-shape (print-panel; ported from the v1 hy3d-cli parity harness)

func cmdParityShape(_ args: Args) throws {
    let fx = FixtureStore(args)
    // Fixture filenames — see parity/README.md (shape fixtures are namespaced `shape_*` so shape
    // and paint fixtures can share one directory). The 2.0-turbo DiT fixture does not exist yet;
    // its expected name is documented and the panel handles it when present.
    let shapeFixtures = ["shape_dit_fixture.safetensors", "shape_dit_fixture_turbo.safetensors",
                         "shape_vae_fixture.safetensors", "shape_dino_fixture.safetensors"]
    guard fx.anyExists(shapeFixtures) else {
        print("no fixtures found at \(fx.dir)")
        return
    }
    guard let weightsDir = args.str("weights") else {
        throw CLIError("shape parity fixtures present at \(fx.dir), but --weights <checkpoint dir> is required to run the Swift forwards")
    }
    guard let wURL = resolveShapeWeights(weightsDir) else {
        throw CLIError("no .safetensors checkpoint under \(weightsDir)")
    }
    print("shape parity — fixtures: \(fx.dir)")
    let (dit, vae, dino) = try loadShapeModels(wURL)

    func line(_ name: String, _ got: MLXArray, _ exp: MLXArray, gate: String) {
        print(String(format: "  %-16@ cos %.7f  maxabs %.3e   [%@]",
                     name, Metric.cosine(got, exp), Metric.maxabs(got, exp), gate))
    }

    if fx.exists("shape_dit_fixture.safetensors") {
        let f = try fx.load("shape_dit_fixture.safetensors")
        line("DiT (mini)", dit(f["x"]!, f["t"]!, f["cond"]!), f["v"]!, gate: "cos >= 0.99999")
    }
    if fx.exists("shape_vae_fixture.safetensors") {
        let f = try fx.load("shape_vae_fixture.safetensors")
        line("VAE geo-grid", vae.geoDecoder(f["q"]!, vae.decode(f["lat"]!)), f["sdf"]!, gate: "cos >= 0.9999")
    }
    if fx.exists("shape_dino_fixture.safetensors") {
        let f = try fx.load("shape_dino_fixture.safetensors")
        line("DINOv2", dino(f["pixels"]!), f["out"]!, gate: "cos >= 0.9999")
    }
    if fx.exists("shape_dit_fixture_turbo.safetensors") {
        if let tw = args.str("weights-turbo"), let twURL = resolveShapeWeights(tw) {
            let (tdit, _, _) = try loadShapeModels(twURL)
            let f = try fx.load("shape_dit_fixture_turbo.safetensors")
            line("DiT (2.0-turbo)", tdit(f["x"]!, f["t"]!, f["cond"]!, guidance: f["guidance"]),
                 f["v"]!, gate: "cos >= 0.99999")
        } else {
            print("  DiT (2.0-turbo)   [skipped: pass --weights-turbo <turbo checkpoint dir>]")
        }
    }
}
