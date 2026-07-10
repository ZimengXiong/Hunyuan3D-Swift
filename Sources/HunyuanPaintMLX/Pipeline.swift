import Foundation
import MLX
import MLXRandom

/// Weight loading from the original torch safetensors (NCHW conv → NHWC transpose + substring renames).
public enum Weights {
    public static func loadTorch(_ path: String, renames: [(String, String)] = []) throws -> [String: MLXArray] {
        let sd = try loadArrays(url: URL(fileURLWithPath: path))
        var out = [String: MLXArray]()
        for (k0, v0) in sd {
            var k = k0
            for (a, b) in renames { k = k.replacingOccurrences(of: a, with: b) }
            out[k] = (v0.ndim == 4 ? v0.transposed(0, 2, 3, 1) : v0).asType(.float32)
        }
        return out
    }
    public static func splitPBR(_ all: [String: MLXArray]) -> (W, W) {
        var main = [String: MLXArray](), dual = [String: MLXArray]()
        for (k, v) in all {
            if k.hasPrefix("unet_dual.") { dual[String(k.dropFirst(10))] = v }
            else if k.hasPrefix("unet.") { main[String(k.dropFirst(5))] = v }
        }
        return (W(main), W(dual))
    }
}

/// Result geometry + baked texture from a paint run. Format-agnostic — the caller
/// serializes to whatever mesh format it uses (Modelr writes a `.tmesh` + PNG).
public struct PaintResult {
    public let vertices: [Float]   // flat xyz, unwrapped geometry
    public let faces: [UInt32]     // flat triangle indices
    public let uvs: [Float]        // flat uv, viewer convention (v-flipped to top-left)
    public let albedoPNG: Data     // baked base-color texture as PNG bytes
}

/// Paint pipeline in Swift: mesh + image → textured geometry. Port of run_paint*.py.
/// A class so loaded model weights stay resident across runs.
public final class PaintPipeline {
    let weightsRoot: String
    public var res: Int, steps: Int, tex: Int    // per-run knobs; do not affect which weights load
    // Audit fix (b): CFG guidance is per-model (RGB/2.0 uses 2.0, PBR/2.1 uses 3.0 — matches
    // scripts/run_paint.py and scripts/run_paint_pbr.py). It was previously hardcoded to 3.0 for
    // both paths; it is now a per-method parameter with the correct default per model.
    let sf: Float = 0.18215
    let superRes: Bool
    let elevs: [Float] = [0, 0, 0, 0, 90, -90]
    let azims: [Float] = [0, 90, 180, 270, 0, 180]
    let vw: [Float] = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

    // Resident RGB (2.0) models, loaded once on first paintRGB.
    private var rgb: (vae: PaintVAE, wrap: Paint20Wrapper, sr: RealESRGAN?, gen: MLXArray)?

    public init(weightsRoot: String, res: Int = 512, steps: Int = 15, tex: Int = 4096, superRes: Bool = true) {
        self.weightsRoot = weightsRoot; self.res = res; self.steps = steps; self.tex = tex; self.superRes = superRes
    }

    private func loadRGB() throws -> (vae: PaintVAE, wrap: Paint20Wrapper, sr: RealESRGAN?, gen: MLXArray) {
        if let r = rgb { return r }
        let vae = PaintVAE(W(try Weights.loadTorch("\(weightsRoot)/hunyuan3d-paint-v2-0/vae/diffusion_pytorch_model.safetensors",
                                               renames: [(".to_out.0.", ".to_out.")])))
        let (mainW, dualW) = Weights.splitPBR(try Weights.loadTorch("\(weightsRoot)/hunyuan3d-paint-v2-0/unet/diffusion_pytorch_model.safetensors",
                                                                renames: [("transformer_blocks.0.transformer.", "transformer_blocks.0.")]))
        let wrap = Paint20Wrapper(main: mainW, dual: dualW)
        // Super-res weights are a converted (non-HF) file; if absent, paint still works
        // without the x4 upscale rather than crashing.
        var sr: RealESRGAN? = nil
        if superRes,
           let arrs = try? loadArrays(url: URL(fileURLWithPath: "\(weightsRoot)/realesrgan/rrdbnet_mlx.safetensors")) {
            sr = RealESRGAN(W(arrs.mapValues { $0.asType(.float32) }))
        }
        let r = (vae, wrap, sr, mainW.a("learned_text_clip_gen"))
        rgb = r
        return r
    }

    public func run(meshPath: String, imagePath: String, outGLB: String, fixturesDir: String,
                    guidance: Float = 3.0) throws {
        let t0 = Date()
        func log(_ s: String) { print("[pipeline] \(s)  (\(Int(-t0.timeIntervalSinceNow))s)") }

        // ---- models ----
        let vae = PaintVAE(W(try Weights.loadTorch("\(weightsRoot)/hunyuan3d-paint-v2-0/vae/diffusion_pytorch_model.safetensors",
                                               renames: [(".to_out.0.", ".to_out.")])))
        let (mainW, dualW) = Weights.splitPBR(try Weights.loadTorch("\(weightsRoot)/hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.safetensors"))
        let wrap = PBRWrapper(main: mainW, dual: dualW, nPbr: 2)
        let dino = Dinov2(W(try Weights.loadTorch("\(weightsRoot)/dinov2-giant/model.safetensors")))
        let srModel: RealESRGAN? = superRes
            ? RealESRGAN(W((try loadArrays(url: URL(fileURLWithPath: "\(weightsRoot)/realesrgan/rrdbnet_mlx.safetensors"))).mapValues { $0.asType(.float32) }))
            : nil
        log("models loaded")

        // ---- mesh + unwrap ----
        let mesh = loadMesh(meshPath)
        guard let uw = xatlasUnwrap(vertices: mesh.vertices, vertexCount: mesh.vertexCount, faces: mesh.faces, faceCount: mesh.faceCount) else { return }
        var V = [Float](repeating: 0, count: uw.vertexCount * 3)               // original geometry gathered by vmapping
        for i in 0..<uw.vertexCount { let o = Int(uw.vmapping[i]) * 3; V[i*3] = mesh.vertices[o]; V[i*3+1] = mesh.vertices[o+1]; V[i*3+2] = mesh.vertices[o+2] }
        let R = MeshRender(); R.loadMesh(V, uw.indices); R.setUV(uw.uvs, flipV: true)
        log("unwrap: \(uw.vertexCount) verts \(uw.indices.count/3) faces")

        // ---- control maps + VAE encode + DINO ----
        let ctrl = zip(elevs, azims).map { R.renderControl($0.0, $0.1, res) }
        let normals = ctrl.map { $0.0 }, positions = ctrl.map { $0.1 }
        func enc(_ imgs: [MLXArray]) -> MLXArray { vae.encodeMean(stacked(imgs) * 2 - 1) * sf }
        let normalLat = enc(normals).expandedDimensions(axis: 0)               // [1,N,h,w,4]
        let positionLat = enc(positions).expandedDimensions(axis: 0)
        let refImg = prepRGB(imagePath, res)
        let refLat = enc([refImg]).expandedDimensions(axis: 0)                  // [1,1,h,w,4]
        let di = imagenetNorm(prepRGB(imagePath, 518)).expandedDimensions(axis: 0)
        let dinoHS = dino(di)                                                   // [1,1370,1536]
        let posmap = stacked(positions).expandedDimensions(axis: 0)            // [1,N,res,res,3]
        let N = elevs.count, h = res / 8
        eval(normalLat, positionLat, refLat, dinoHS); log("controls + dino ready")

        // ---- diffusion loop ----
        let (schedSigmas, schedTs) = uniPCSchedule(steps)
        let sched = UniPCScheduler(sigmas: schedSigmas, timesteps: schedTs)
        MLXRandom.seed(0)
        var latents = MLXRandom.normal([1, 2, N, h, h, 4])
        let (ced, dinoTok, rope) = wrap.prepare(refLat: refLat, dinoHidden: dinoHS, posmap: posmap, H: h, nGen: N)
        let dinoZero = zeros(dinoTok.shape)
        let nb = 1 * 2 * N
        for (i, t) in sched.timesteps.enumerated() {
            let tArr = MLXArray(Array(repeating: Float(t), count: nb))
            let vc = wrap.predict(latents, tArr, normalLat: normalLat, positionLat: positionLat, ced: ced, dino: dinoTok, rope: rope, mvaScale: 1, refScale: 1)
            let vu = wrap.predict(latents, tArr, normalLat: normalLat, positionLat: positionLat, ced: nil, dino: dinoZero, rope: rope, mvaScale: 1, refScale: 0)
            latents = sched.step(vu + guidance * (vc - vu), t, latents); eval(latents)
            log("step \(i+1)/\(steps)")
        }

        // ---- decode + super-res ----
        func decode(_ lat: MLXArray) -> [MLXArray] {
            let d = clip((vae.decode(lat / sf) + 1) / 2, min: 0, max: 1)        // [N,H,W,3]
            return (0..<N).map { d[$0] }
        }
        var alb = decode(latents[0, 0]), mr = decode(latents[0, 1])
        saveRGB(concatenated(alb, axis: 1), "\(outGLB).views.png")             // debug: decoded albedo views grid
        if let sr = srModel {
            func up(_ v: MLXArray) -> MLXArray { clip(sr(v.expandedDimensions(axis: 0))[0], min: 0, max: 1) }
            alb = alb.map(up); mr = mr.map(up)
            eval(alb[0]); log("super-res x4 → \(alb[0].dim(0))px")
        }

        // ---- bake + inpaint ----
        let (texs, covered) = R.bakeMulti([alb, mr], elevs, azims, textureSize: tex, weights: vw)
        let texA = MeshRender.inpaint(texs[0], covered), texM = MeshRender.inpaint(texs[1], covered)
        eval(texA, texM); log("baked")
        // debug: render the texture back onto the mesh (bypasses GLB) at 3 angles
        let dbg = [R.renderTextured(0, 20, 420, texA), R.renderTextured(0, 140, 420, texA), R.renderTextured(0, 260, 420, texA)]
        saveRGB(concatenated(dbg, axis: 1), "\(outGLB).rendercheck.png")

        // ---- export GLB ----
        let albPNG = "\(outGLB).albedo.png", mrPNG = "\(outGLB).mr.png"
        saveRGB(texA, albPNG); saveRGB(texM, mrPNG)
        var uvExport = uw.uvs                                                  // v-flip for the viewer (v-up)
        for i in 0..<(uvExport.count/2) { uvExport[i*2+1] = 1 - uvExport[i*2+1] }
        try writeGLB(path: outGLB, vertices: V, faces: uw.indices, uvs: uvExport,
                     baseColorPNG: try Data(contentsOf: URL(fileURLWithPath: albPNG)),
                     metallicRoughnessPNG: try Data(contentsOf: URL(fileURLWithPath: mrPNG)))
        log("DONE → \(outGLB)")
    }

    /// 2.0 RGB paint: geometry + reference image → unwrapped geometry + baked base-color texture.
    /// Polls `isCancelled` (returns nil if it fires); streams decoded view grids via `onViews`.
    public func paintRGB(mesh: LoadedMesh, imagePath: String, guidance: Float = 2.0,
                         onProgress: ((String, Float) -> Void)? = nil,
                         isCancelled: () -> Bool = { false },
                         onViews: ((Data) -> Void)? = nil) throws -> PaintResult? {
        onProgress?("Loading paint model", 0.02)
        let (vae, wrap, srModel, gen) = try loadRGB()
        if isCancelled() { return nil }

        onProgress?("Unwrapping UVs", 0.05)
        guard let uw = xatlasUnwrap(vertices: mesh.vertices, vertexCount: mesh.vertexCount,
                                    faces: mesh.faces, faceCount: mesh.faceCount) else { return nil }
        var V = [Float](repeating: 0, count: uw.vertexCount * 3)
        for i in 0..<uw.vertexCount { let o = Int(uw.vmapping[i]) * 3; V[i*3] = mesh.vertices[o]; V[i*3+1] = mesh.vertices[o+1]; V[i*3+2] = mesh.vertices[o+2] }
        let R = MeshRender(); R.loadMesh(V, uw.indices); R.setUV(uw.uvs, flipV: true)
        if isCancelled() { return nil }

        onProgress?("Rendering control maps", 0.1)
        let ctrl = zip(elevs, azims).map { R.renderControl($0.0, $0.1, res) }
        let normals = ctrl.map { $0.0 }, positions = ctrl.map { $0.1 }
        func enc(_ imgs: [MLXArray]) -> MLXArray { vae.encodeMean(stacked(imgs) * 2 - 1) * sf }
        let normalLat = enc(normals).expandedDimensions(axis: 0)
        let positionLat = enc(positions).expandedDimensions(axis: 0)
        let refLat = enc([prepRGB(imagePath, res)]).expandedDimensions(axis: 0)   // [1,1,h,w,4]
        let N = elevs.count, h = res / 8
        if isCancelled() { return nil }

        let (sig, ts) = uniPCSchedule(steps)
        let sched = UniPCScheduler(sigmas: sig, timesteps: ts)
        MLXRandom.seed(0)
        var latents = MLXRandom.normal([1, N, h, h, 4])
        let ced = wrap.prepare(refLat: refLat)
        let neg = zeros(gen.shape)
        let camGen = (0..<N).map { Int32($0) }
        for (i, t) in ts.enumerated() {
            if isCancelled() { return nil }
            let tArr = MLXArray(Array(repeating: Float(t), count: N))
            let vc = wrap.predict(latents, tArr, text: gen, normalLat: normalLat, positionLat: positionLat, camGen: camGen, ced: ced, mvaScale: 1, refScale: 1)
            let vu = wrap.predict(latents, tArr, text: neg, normalLat: normalLat, positionLat: positionLat, camGen: camGen, ced: nil, mvaScale: 1, refScale: 0)
            latents = sched.step(vu + guidance * (vc - vu), t, latents); eval(latents)
            onProgress?("Painting (\(i+1)/\(steps))", 0.15 + 0.6 * Float(i + 1) / Float(steps))
            if let onViews, i % 3 == 2 || i == steps - 1 {
                let prev = clip((vae.decode(latents[0] / sf) + 1) / 2, min: 0, max: 1)   // [N,H,W,3]
                let grid = concatenated((0..<N).map { prev[$0] }, axis: 1)
                if let d = pngData(grid) { onViews(d) }
            }
            MLX.Memory.clearCache()                    // release per-step UNet/decode buffers
        }
        if isCancelled() { return nil }

        onProgress?("Decoding views", 0.8)
        let dd = clip((vae.decode(latents[0] / sf) + 1) / 2, min: 0, max: 1)       // [N,H,W,3]
        var views = (0..<N).map { dd[$0] }
        if let sr = srModel {
            onProgress?("Super-resolving", 0.88)
            views = views.map { clip(sr($0.expandedDimensions(axis: 0))[0], min: 0, max: 1) }; eval(views[0])
        }
        if isCancelled() { return nil }

        onProgress?("Baking texture", 0.93)
        let (texs, covered) = R.bakeMulti([views], elevs, azims, textureSize: tex, weights: vw)
        let texC = MeshRender.inpaint(texs[0], covered); eval(texC)
        guard let albedoPNG = pngData(texC) else { return nil }
        var uvOut = uw.uvs
        for i in 0..<(uvOut.count / 2) { uvOut[i*2+1] = 1 - uvOut[i*2+1] }          // v-flip → viewer top-left
        onProgress?("Done", 1.0)
        return PaintResult(vertices: V, faces: uw.indices, uvs: uvOut, albedoPNG: albedoPNG)
    }
}
