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
    if fx.exists("shape_sigmas_fixture.safetensors") {
        let f = try fx.load("shape_sigmas_fixture.safetensors")
        let fm = Metric.maxabs(Sampler.flowMatchSigmas(f["flowmatch"]!.dim(0)), f["flowmatch"]!)
        let cs = Metric.maxabs(Sampler.consistencySigmas(f["consistency"]!.dim(0)), f["consistency"]!)
        print(String(format: "  %-16@ flow-match maxabs %.3e  consistency maxabs %.3e   [maxabs <= 1e-6]",
                     "Sigmas", fm, cs))
    }
    // ---- e2e mesh (fixed cond+noise -> denoise -> octree decode -> marching cubes) vs Python ----
    func e2e(_ name: String, runFixture: String, meshFixture: String, weights: URL, guidanceEmbed: Bool) throws {
        guard fx.exists(runFixture), fx.exists(meshFixture) else { return }
        let f = try fx.load(runFixture)
        let ref = try fx.load(meshFixture)
        let (d, v, _) = try loadShapeModels(weights)
        let pipe = Pipeline(dit: d, vae: v)
        let lat = pipe.denoise(cond: f["cond"]!, noise: f["noise"]!, sigmas: f["sigmas"]!,
                               guidance: f["guidance"]?.item(Float.self) ?? 5.0,
                               guidanceEmbed: guidanceEmbed)
        eval(lat)
        let grid = pipe.gridSDFOctree(latents: lat, resolution: 256); eval(grid)
        let mesh = MarchingCubes.extract(grid: grid, level: 0.0)
        let cd = chamferBBox(mesh.vertices.map { [$0.x, $0.y, $0.z] },
                             ref["V"]!.asType(.float32).asArray(Float.self))
        print(String(format: "  %-16@ %d verts  Chamfer/bbox %.5f   [<= 0.01]", name, mesh.vertices.count, cd))
    }
    try e2e("e2e mesh (mini)", runFixture: "shape_run_fixture.safetensors",
            meshFixture: "shape_mesh_python_mini.safetensors", weights: wURL, guidanceEmbed: false)
    if let tw = args.str("weights-turbo"), let twURL = resolveShapeWeights(tw) {
        try e2e("e2e mesh (turbo)", runFixture: "shape_run_fixture_turbo.safetensors",
                meshFixture: "shape_mesh_python_turbo.safetensors", weights: twURL, guidanceEmbed: true)
    }
}

/// Symmetric Chamfer distance / union-bbox diagonal (subsampled; mirrors the XCTest gate).
func chamferBBox(_ a: [[Float]], _ bFlat: [Float], sample: Int = 1500) -> Float {
    var b = [[Float]](); b.reserveCapacity(bFlat.count / 3)
    var i = 0
    while i + 2 < bFlat.count { b.append([bFlat[i], bFlat[i + 1], bFlat[i + 2]]); i += 3 }
    func stride(_ v: [[Float]]) -> [[Float]] {
        guard v.count > sample else { return v }
        let st = v.count / sample
        return (0 ..< sample).map { v[$0 * st] }
    }
    let sa = stride(a), sb = stride(b)
    func meanNN(_ from: [[Float]], _ to: [[Float]]) -> Float {
        var sum: Float = 0
        for p in from {
            var best = Float.greatestFiniteMagnitude
            for q in to {
                let dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2]
                let d = dx * dx + dy * dy + dz * dz
                if d < best { best = d }
            }
            sum += best.squareRoot()
        }
        return sum / Float(max(from.count, 1))
    }
    let cd = 0.5 * (meanNN(sa, sb) + meanNN(sb, sa))
    var lo: [Float] = [.greatestFiniteMagnitude, .greatestFiniteMagnitude, .greatestFiniteMagnitude]
    var hi: [Float] = [-.greatestFiniteMagnitude, -.greatestFiniteMagnitude, -.greatestFiniteMagnitude]
    for p in a + b {
        for k in 0 ..< 3 { lo[k] = min(lo[k], p[k]); hi[k] = max(hi[k], p[k]) }
    }
    let dx = hi[0] - lo[0], dy = hi[1] - lo[1], dz = hi[2] - lo[2]
    let diag = (dx * dx + dy * dy + dz * dz).squareRoot()
    return diag > 0 ? cd / diag : cd
}
