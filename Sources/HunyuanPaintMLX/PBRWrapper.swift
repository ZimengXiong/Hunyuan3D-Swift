import Foundation
import MLX
import MLXFast

/// 2.1 PBR wrapper — functional port of unet2p5d_pbr.py: ImageProjModel (DINO projector),
/// 3D PoseRoPE (voxel indices → rotary tables), the dual-pass `prepare`, and the diffusion step.
public struct PBRWrapper {
    let w: W            // main self.unet weights (pbr) + learned_text_clip_* + image_proj_model_dino.*
    let wd: W           // unet_dual weights (base, pbr off)
    let nPbr: Int

    public init(main: W, dual: W, nPbr: Int = 2) { self.w = main; self.wd = dual; self.nPbr = nPbr }

    // ---- ImageProjModel: [B,N,1536] -> [B, N*4, 1024] ----
    public static func imageProj(_ embeds: MLXArray, _ w: W, _ p: String,
                                 extra: Int = 4, crossDim: Int = 1024) -> MLXArray {
        let numToken = embeds.dim(1)
        let flat = embeds.reshaped([-1, embeds.dim(2)])
        var tok = L2D.linear(flat, w, "\(p).proj").reshaped([-1, extra, crossDim])
        tok = MLXFast.layerNorm(tok, weight: w.a("\(p).norm.weight"), bias: w.a("\(p).norm.bias"), eps: 1e-5)
        let b = flat.dim(0) / numToken
        return tok.reshaped([b, numToken * extra, crossDim])
    }

    // ---- 3D PoseRoPE ----
    static func rotary1d(_ dim: Int, _ pos: MLXArray, theta: Float = 10000) -> (MLXArray, MLXArray) {
        let half = dim / 2
        let idx = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) })[0 ..< half]
        let freqs = 1.0 / pow(MLXArray(theta), idx / Float(dim))
        let f = outer(pos.asType(.float32), freqs)
        return (repeated(cos(f), count: 2, axis: 1), repeated(sin(f), count: 2, axis: 1))
    }

    public static func rotary3d(_ position: MLXArray, embedDim: Int, voxelRes: Int,
                                theta: Float = 10000) -> (MLXArray, MLXArray) {
        let dimXY = embedDim / 8 * 3, dimZ = embedDim / 8 * 2
        let grid = MLXArray(0 ..< voxelRes).asType(.float32)
        let (xyC, xyS) = rotary1d(dimXY, grid, theta: theta)
        let (zC, zS) = rotary1d(dimZ, grid, theta: theta)
        let shp = position.shape
        let flat = position.reshaped([-1, shp[shp.count - 1]]).asType(.int32)
        let f0 = flat[0..., 0], f1 = flat[0..., 1], f2 = flat[0..., 2]
        let cosO = concatenated([take(xyC, f0, axis: 0), take(xyC, f1, axis: 0), take(zC, f2, axis: 0)], axis: -1)
        let sinO = concatenated([take(xyS, f0, axis: 0), take(xyS, f1, axis: 0), take(zS, f2, axis: 0)], axis: -1)
        var outShape = Array(shp.dropLast()); outShape.append(embedDim)
        return (cosO.reshaped(outShape), sinO.reshaped(outShape))
    }

    /// compute_discrete_voxel_indice (fp16 window-average downsample + quantize). position [b,n,H,W,3] in [0,1].
    /// Exact CPU port of the reference numpy fp16 arithmetic: the input is cast to fp16 first
    /// (so `valid` tests the fp16 values), the window sum accumulates sequentially row-major in
    /// fp16 (numpy's strided multi-axis reduce), the count/threshold compare in integers, and the
    /// divide / clip / scale / round(half-even) all round to fp16 per op — voxel indices are
    /// bit-identical to Python, so the RoPE tables need no injection.
    public static func voxelIndices(_ position: MLXArray, gridRes: Int, voxelRes: Int) -> MLXArray {
        let b = position.dim(0), n = position.dim(1), H = position.dim(2), Wd = position.dim(3), c = position.dim(4)
        precondition(c == 3)
        let gh = H / gridRes, gw = Wd / gridRes
        let thres = (gh * gw) / 16
        let scale = Float16(voxelRes - 1)
        let p32 = position.asType(.float32).asArray(Float.self)
        let p16 = p32.map { Float16($0) }
        var out = [Int32](repeating: 0, count: b * n * gridRes * gridRes * 3)
        for bi in 0 ..< b {
            for ni in 0 ..< n {
                let imgBase = (bi * n + ni) * H * Wd * 3
                for g1 in 0 ..< gridRes {
                    for g2 in 0 ..< gridRes {
                        var acc: (Float16, Float16, Float16) = (0, 0, 0)
                        var count = 0
                        for hh in 0 ..< gh {                       // numpy reduce order: rows then cols
                            let row = imgBase + ((g1 * gh + hh) * Wd + g2 * gw) * 3
                            for ww in 0 ..< gw {
                                let p = row + ww * 3
                                let x = p16[p], y = p16[p + 1], z = p16[p + 2]
                                if x != 1, y != 1, z != 1 {        // valid: all channels != 1 (fp16)
                                    acc.0 += x; acc.1 += y; acc.2 += z
                                    count += 1
                                }
                            }
                        }
                        let o = (((bi * n + ni) * gridRes + g1) * gridRes + g2) * 3
                        if count < thres {
                            out[o] = 0; out[o + 1] = 0; out[o + 2] = 0
                        } else {
                            let denom = Float16(max(count, 1))
                            @inline(__always) func q(_ s: Float16) -> Int32 {
                                let gp = min(max(s / denom, 0), 1)
                                return Int32((gp * scale).rounded(.toNearestOrEven))
                            }
                            out[o] = q(acc.0); out[o + 1] = q(acc.1); out[o + 2] = q(acc.2)
                        }
                    }
                }
            }
        }
        return MLXArray(out, [b, n * gridRes * gridRes, 3])
    }

    /// rope tables keyed by multiview token count, from pixel position maps.
    public func ropeByTokens(_ posmap: MLXArray, hLat: Int, nGen: Int) -> [Int: (MLXArray, MLXArray)] {
        let grids = [hLat, hLat / 2, hLat / 4, hLat / 8]
        let vres = [hLat * 8, hLat * 4, hLat * 2, hLat]
        var out = [Int: (MLXArray, MLXArray)]()
        for (g, vr) in zip(grids, vres) {
            var vox = PBRWrapper.voxelIndices(posmap, gridRes: g, voxelRes: vr)   // [b, nGen*g*g, 3]
            vox = repeated(vox.expandedDimensions(axis: 1), count: nPbr, axis: 1)
                .reshaped([vox.dim(0) * nPbr, vox.dim(1), 3])
            out[nGen * g * g] = PBRWrapper.rotary3d(vox, embedDim: 64, voxelRes: vr)
        }
        return out
    }

    /// Dual-pass reference features + DINO projection + RoPE — constant across diffusion steps.
    public func prepare(refLat: MLXArray, dinoHidden: MLXArray, posmap: MLXArray,
                        H: Int, nGen: Int) -> (ced: [String: MLXArray], dino: MLXArray,
                                               rope: [Int: (MLXArray, MLXArray)]) {
        let B = refLat.dim(0), nRef = refLat.dim(1)
        let rl = refLat.reshaped([B * nRef, H, refLat.dim(3), 4])
        let refText = broadcast(w.a("learned_text_clip_ref").expandedDimensions(axis: 0), to: [B * nRef, 77, 1024])
        let xa = XAttn(mode: "w", numInBatch: nRef, nPbr: 1)
        let dual = PaintUNet(wd, pbr: true)   // dual ref unet uses PBR blocks with all use-flags OFF
        _ = dual(rl, zeros([B * nRef]), refText, xattn: xa)             // writes xa.conditionEmbed
        let dino = PBRWrapper.imageProj(dinoHidden, w, "image_proj_model_dino")
        return (xa.conditionEmbed, dino, ropeByTokens(posmap, hLat: H, nGen: nGen))
    }

    /// One PBR UNet velocity prediction. sample [B,nPbr,nGen,H,W,4].
    public func predict(_ sample: MLXArray, _ t: MLXArray, normalLat: MLXArray, positionLat: MLXArray,
                        ced: [String: MLXArray]?, dino: MLXArray, rope: [Int: (MLXArray, MLXArray)],
                        mvaScale: Float, refScale: Float) -> MLXArray {
        let B = sample.dim(0), Np = sample.dim(1), Ng = sample.dim(2), H = sample.dim(3), Wd = sample.dim(4)
        let nrep = broadcast(normalLat.expandedDimensions(axis: 1), to: [B, Np, Ng, H, Wd, 4])
        let prep = broadcast(positionLat.expandedDimensions(axis: 1), to: [B, Np, Ng, H, Wd, 4])
        let s = concatenated([sample, nrep, prep], axis: -1).reshaped([B * Np * Ng, H, Wd, 12])
        let alb = w.a("learned_text_clip_albedo"), mr = w.a("learned_text_clip_mr")
        var ehs = stacked([alb, mr], axis: 0).expandedDimensions(axis: 0)        // [1,2,77,1024]
        ehs = broadcast(ehs.expandedDimensions(axis: 2), to: [B, Np, Ng, 77, 1024]).reshaped([B * Np * Ng, 77, 1024])
        let xa = XAttn(mode: "r", numInBatch: Ng, conditionEmbed: ced ?? [:], refScale: refScale,
                       mvaScale: mvaScale, nPbr: Np, dino: dino, ropeByTokens: rope)
        let unet = PaintUNet(w, useMa: true, useRa: true, useMda: true, useDino: true, pbr: true)
        let out = unet(s, t, ehs, xattn: xa)
        return out.reshaped([B, Np, Ng, H, Wd, 4])
    }
}
