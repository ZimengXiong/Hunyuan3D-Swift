import Foundation
import CoreGraphics
import MLX

/// High-level, app-facing entry point: load a 2mini checkpoint once, then turn a photo into a mesh
/// entirely on-device (MLX) — no Python, no fixtures, no network. Suitable for the iOS sandbox.
public final class ShapeGenerator {
    public let dit: DiT
    public let vae: VAE
    public let dino: DINOv2
    public let numLatents: Int

    /// Load a bundled 2mini checkpoint (single `.safetensors`) and build DiT + VAE + DINOv2.
    /// Weights stay in their on-disk dtype (fp16) — upcasting to fp32 would double resident memory
    /// (~3.6 GB -> ~7.6 GB) and OOM every iOS device. fp16 is also the reference activation dtype.
    /// `quantize`: 0 = fp16, or 4 / 8 for weights-only quantization of the bulk DiT + DINO linears
    /// (group size 64; VAE, norms, embedders + the projection/skip linears stay fp16). 4-bit cuts
    /// resident weights ~2.5× (≈3.6 GB -> ≈1.4 GB).
    public init(weightsURL: URL, dtype: DType = .float16, quantize: Int = 0,
                numLatents: Int = 512, cacheLimitMB: Int = 256) throws {
        MLX.Memory.cacheLimit = cacheLimitMB * 1024 * 1024    // keep the buffer cache off the jetsam limit
        let cfg = Self.readConfig(weightsURL)                 // per-model DiT shape (mini / 2.0 / turbo)
        let weights = try loadArrays(url: weightsURL)
        let condPrefix = "conditioner.main_image_encoder.model."
        var dw: [String: MLXArray] = [:], vw: [String: MLXArray] = [:], nw: [String: MLXArray] = [:]
        for (k, v) in weights {
            let w = v.dtype == dtype ? v : v.asType(dtype)
            if k.hasPrefix("model.") { dw[String(k.dropFirst(6))] = w }
            else if k.hasPrefix("vae.") { vw[String(k.dropFirst(4))] = w }
            else if k.hasPrefix(condPrefix) { nw[String(k.dropFirst(condPrefix.count))] = w }
        }
        let bits = (quantize == 4 || quantize == 8) ? quantize : 4, group = 64
        if quantize == 4 || quantize == 8 {
            dw = Self.quantizeWeights(dw, bits: quantize, group: group,
                skip: ["latent_in", "time_in", "cond_in", "guidance_in", "final_layer", "x_embedder"])
            nw = Self.quantizeWeights(nw, bits: quantize, group: group, skip: ["embeddings"])
            eval(Array(dw.values)); eval(Array(nw.values))   // materialize the packed 4-bit, drop fp16
        }
        self.dit = DiT(weights: dw, depth: cfg?.depth ?? 8, depthSingle: cfg?.depthSingle ?? 16,
                       guidanceEmbed: cfg?.guidanceEmbed ?? false, bits: bits, groupSize: group)
        // Audit fix (a): the ShapeVAE scale_factor is per-checkpoint (2mini 1.0188…, the whole 2.0
        // family 0.99909…, 2.1 1.00395…). It was previously hardcoded to the 2mini value, giving a
        // silent ~2% SDF-scale error on every non-mini model. Read it from config.yaml and pass it
        // through; fall back to the 2mini value when the config is absent.
        self.vae = VAE(weights: vw, scaleFactor: cfg?.scaleFactor ?? 1.0188137)
        self.dino = DINOv2(weights: nw, bits: bits, groupSize: group)
        self.numLatents = cfg?.numLatents ?? numLatents
    }

    struct ModelConfig { let depth: Int; let depthSingle: Int; let guidanceEmbed: Bool; let numLatents: Int; let scaleFactor: Float }

    /// Parse the per-model shape params from the checkpoint's sibling `config.yaml`
    /// (so mini / 2.0 / turbo variants all load correctly). Minimal key-scan — no YAML lib.
    static func readConfig(_ weightsURL: URL) -> ModelConfig? {
        let url = weightsURL.deletingLastPathComponent().appendingPathComponent("config.yaml")
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        func value(_ key: String) -> String? {
            for raw in text.split(separator: "\n") {
                let line = raw.trimmingCharacters(in: .whitespaces)
                if line.hasPrefix("\(key):") {
                    return line.dropFirst(key.count + 1).trimmingCharacters(in: .whitespaces)
                }
            }
            return nil
        }
        guard let d = value("depth").flatMap({ Int($0) }),
              let ds = value("depth_single_blocks").flatMap({ Int($0) }),
              let nl = value("num_latents").flatMap({ Int($0) }) else { return nil }
        // Audit fix (a): parse scale_factor; default to the 2mini value if the key is absent.
        let sf = value("scale_factor").flatMap { Float($0) } ?? 1.0188137142395404
        return ModelConfig(depth: d, depthSingle: ds,
                           guidanceEmbed: value("guidance_embed")?.lowercased() == "true",
                           numLatents: nl, scaleFactor: sf)
    }

    /// Quantize the 2-D linear weights of a sub-model in place, skipping any path in `skip` and any
    /// weight whose input dim isn't a multiple of the group size. Mirrors hy3dmlx.convert._quantize.
    static func quantizeWeights(_ d: [String: MLXArray], bits: Int, group: Int,
                                skip: [String]) -> [String: MLXArray] {
        var out = d
        for (k, v) in d where k.hasSuffix(".weight") && v.ndim == 2 && v.dim(1) % group == 0
            && !skip.contains(where: { k.contains($0) }) {
            let (wq, scales, biases) = quantized(v, groupSize: group, bits: bits)
            out[k] = wq
            out["\(k).scales"] = scales
            if let biases { out["\(k).biases"] = biases }
        }
        return out
    }

    public struct Progress: Sendable { public let stage: String; public let fraction: Float }

    /// Full on-device pipeline: photo -> watertight mesh.
    /// `isCancelled` is polled per denoise step and at each stage boundary; returns nil if it fires.
    public func generate(image: CGImage, steps: Int = 30, guidance: Float = 5.0,
                         seed: UInt64 = 0, resolution: Int = 256, octree: Bool = true,
                         isCancelled: () -> Bool = { false },
                         onPreview: ((Mesh) -> Void)? = nil,
                         onProgress: ((Progress) -> Void)? = nil) -> Mesh? {
        guard let pix = Preprocess.dinoPixels(cgImage: image) else { return nil }
        if isCancelled() { return nil }
        onProgress?(.init(stage: "Conditioning image", fraction: 0.05))
        // Turbo/distilled models embed guidance and skip CFG → conditional embedding only.
        let embed = dino(pix)
        let cond = dit.guidanceEmbed ? embed : concatenated([embed, dino.unconditional(1)], axis: 0)
        eval(cond)
        if isCancelled() { return nil }

        let pipe = Pipeline(dit: dit, vae: vae)
        let noise = Sampler.noise(numLatents: numLatents, seed: seed)
        let sigmas = dit.guidanceEmbed ? Sampler.consistencySigmas(steps) : Sampler.flowMatchSigmas(steps)
        // Decode a coarse preview from the in-progress latent at a few steps (a meshable
        // surface emerges partway through denoising; earlier steps are skipped when empty).
        let previewSteps: Set<Int> = onPreview == nil ? []
            : Set([0.4, 0.55, 0.7, 0.85].map { Int((Float(steps) * Float($0)).rounded()) })
        let lat = pipe.denoise(cond: cond, noise: noise, sigmas: sigmas, guidance: guidance,
                               guidanceEmbed: dit.guidanceEmbed, isCancelled: isCancelled) { i, n, curLat in
            onProgress?(.init(stage: "Denoising (\(i)/\(n))", fraction: 0.1 + 0.6 * Float(i) / Float(n)))
            if previewSteps.contains(i), let onPreview {
                let g = pipe.gridSDFOctree(latents: curLat, resolution: 48)
                let m = MarchingCubes.extract(grid: g, level: 0.0)
                MLX.Memory.clearCache()
                if !m.vertices.isEmpty && !m.faces.isEmpty { onPreview(m) }
            }
        }
        eval(lat); MLX.Memory.clearCache()        // release the 30-step denoise buffers before decode
        if isCancelled() { return nil }

        onProgress?(.init(stage: "Decoding shape", fraction: 0.72))
        let grid = octree ? pipe.gridSDFOctree(latents: lat, resolution: resolution)
                          : pipe.gridSDF(latents: lat, resolution: resolution)
        eval(grid); MLX.Memory.clearCache()       // release decode buffers before marching cubes
        if isCancelled() { return nil }

        onProgress?(.init(stage: "Building mesh", fraction: 0.9))
        let mesh = MarchingCubes.extract(grid: grid, level: 0.0)
        onProgress?(.init(stage: "Done", fraction: 1.0))
        return mesh
    }

    /// Generate and write a `.glb`; returns vertex/face counts.
    @discardableResult
    public func generateGLB(image: CGImage, to url: URL, steps: Int = 30, guidance: Float = 5.0,
                            seed: UInt64 = 0, resolution: Int = 256, octree: Bool = true,
                            onProgress: ((Progress) -> Void)? = nil) throws -> (verts: Int, faces: Int) {
        guard let mesh = generate(image: image, steps: steps, guidance: guidance, seed: seed,
                                  resolution: resolution, octree: octree, onProgress: onProgress) else {
            throw NSError(domain: "Hy3DMLX", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "image preprocessing failed"])
        }
        try GLB.write(mesh, to: url)
        return (mesh.vertices.count, mesh.faces.count)
    }
}
