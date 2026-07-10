import Foundation
import MLX

/// Shape pipeline in MLX Swift: CFG flow-match denoise (DiT) -> VAE decode -> SDF grid (dense or
/// octree). All tensor math is MLX; only the per-level octree index bookkeeping is host-side.
public struct Pipeline {
    let dit: DiT
    let vae: VAE
    public init(dit: DiT, vae: VAE) { self.dit = dit; self.vae = vae }

    /// Flow-match Euler loop. Two modes:
    ///  - CFG (guidanceEmbed=false): `cond` is [2,Lc,Cc] (cond, uncond); one forward over a
    ///    duplicated batch, combined as v_uncond + guidance·(v_cond − v_uncond).
    ///  - guidance-embed (turbo/distilled): `cond` is [1,Lc,Cc] (conditional only); a single
    ///    forward per step with the guidance value embedded as a token (no CFG).
    public func denoise(cond: MLXArray, noise: MLXArray, sigmas: MLXArray,
                        guidance: Float = 5.0, guidanceEmbed: Bool = false,
                        isCancelled: () -> Bool = { false },
                        progress: ((Int, Int, MLXArray) -> Void)? = nil) -> MLXArray {
        let S = sigmas.dim(0)
        let sig = sigmas.asArray(Float.self)               // pull the fixed schedule once (not per step)
        var lat = noise
        let gvec = guidanceEmbed ? MLXArray([guidance]) : nil
        for i in 0 ..< S {
            if isCancelled() { break }                     // cooperative per-step cancellation
            let dt = (i + 1 < S ? sig[i + 1] : 1.0) - sig[i]
            if dt == 0 { continue }
            let si = sig[i]
            let v: MLXArray
            if guidanceEmbed {
                v = dit(lat, MLXArray([si]), cond, guidance: gvec)   // single forward, guidance token
            } else {
                let vv = dit(concatenated([lat, lat], axis: 0), MLXArray([si, si]), cond)
                let vp = split(vv, parts: 2, axis: 0)                // v_cond, v_uncond
                v = vp[1] + guidance * (vp[0] - vp[1])
            }
            lat = lat + dt * v
            eval(lat)
            progress?(i + 1, S, lat)
        }
        return lat
    }

    /// latents [1,N,64] -> dense (R+1)^3 SDF grid (MLXArray, float32).
    public func gridSDF(latents: MLXArray, resolution R: Int, bound: Float = 1.01,
                        numChunks: Int = 50000) -> MLXArray {
        let kv = vae.decode(latents / vae.scaleFactor)
        eval(kv)
        let n = R + 1
        let step = 2 * bound / Float(R)
        let ax = MLXArray((0 ... R).map { -bound + Float($0) * step })   // exact endpoints
        let xs = broadcast(ax.reshaped([n, 1, 1]), to: [n, n, n]).reshaped([-1])
        let ys = broadcast(ax.reshaped([1, n, 1]), to: [n, n, n]).reshaped([-1])
        let zs = broadcast(ax.reshaped([1, 1, n]), to: [n, n, n]).reshaped([-1])
        let xyz = stacked([xs, ys, zs], axis: -1)                       // [n^3, 3]
        let total = xyz.dim(0)
        var chunks: [MLXArray] = []
        var s = 0
        while s < total {
            let e = Swift.min(s + numChunks, total)
            let q = xyz[s ..< e].expandedDimensions(axis: 0)            // [1,P,3]
            let sdf = vae.geoDecoder(q, kv).reshaped([-1]).asType(.float32)
            eval(sdf)
            chunks.append(sdf)
            s = e
        }
        return concatenated(chunks, axis: 0).reshaped([n, n, n])
    }

    /// Evaluate the geo-decoder SDF at arbitrary query points (flat P*3 coords) -> [P].
    func decodePoints(kv: MLXArray, pts: [Float], count P: Int, numChunks: Int) -> [Float] {
        var out = [Float](); out.reserveCapacity(P)
        var s = 0
        while s < P {
            let e = Swift.min(s + numChunks, P)
            let q = MLXArray(Array(pts[(s * 3) ..< (e * 3)]), [1, e - s, 3])
            let sdf = vae.geoDecoder(q, kv).reshaped([-1]).asType(.float32)
            eval(sdf)
            out.append(contentsOf: sdf.asArray(Float.self))
            s = e
        }
        return out
    }

    /// Octree (FlashVDM near-surface) SDF grid: coarse dense decode, then refine only the
    /// near-surface band level-by-level (x2). Inactive cells are NaN (masked by marching cubes).
    /// Bit-for-bit the same algorithm as hy3dmlx query_grid_octree; only geoDecoder hits MLX.
    public func gridSDFOctree(latents: MLXArray, resolution R: Int = 256, bound: Float = 1.01,
                              minResolution: Int = 63, mcLevel: Float = 0.0,
                              numChunks: Int = 8000) -> MLXArray {
        let kv = vae.decode(latents / vae.scaleFactor); eval(kv)
        let bmin = -bound, bsize = 2 * bound
        var resList: [Int] = []
        var r = R
        while r >= minResolution { resList.append(r); r /= 2 }
        resList.reverse()                                                // [coarse ... fine]
        if resList.isEmpty { resList = [R] }

        // coarse dense decode (ij order: i=x slowest, matches MarchingCubes grid[i,j,k])
        let r0 = resList[0], n0 = r0 + 1
        let step0 = bsize / Float(r0)
        var coarse = [Float](repeating: 0, count: n0 * n0 * n0 * 3)
        var p = 0
        for i in 0 ..< n0 {
            let x = bmin + Float(i) * step0
            for j in 0 ..< n0 {
                let y = bmin + Float(j) * step0
                for k in 0 ..< n0 {
                    coarse[p] = x; coarse[p + 1] = y; coarse[p + 2] = bmin + Float(k) * step0; p += 3
                }
            }
        }
        var grid = decodePoints(kv: kv, pts: coarse, count: n0 * n0 * n0, numChunks: numChunks)
        var gn = n0

        for li in 1 ..< resList.count {
            let res = resList[li], gs = res + 1, step = bsize / Float(res)
            var near = Self.nearSurfaceMask(grid, gn, mcLevel)
            let expand = (res == resList.last!) ? 0 : 1
            for _ in 0 ..< expand { near = Self.dilate(near, gn) }
            var nxt = [Bool](repeating: false, count: gs * gs * gs)      // coarse near -> fine (x2)
            for ci in 0 ..< gn {
                for cj in 0 ..< gn {
                    for ck in 0 ..< gn where near[(ci * gn + cj) * gn + ck] {
                        let fi = Swift.min(ci * 2, gs - 1), fj = Swift.min(cj * 2, gs - 1), fk = Swift.min(ck * 2, gs - 1)
                        nxt[(fi * gs + fj) * gs + fk] = true
                    }
                }
            }
            for _ in 0 ..< (2 - expand) { nxt = Self.dilate(nxt, gs) }
            var nidx: [Int] = []                                        // flat indices of active voxels
            var pts: [Float] = []
            for fi in 0 ..< gs {
                let x = bmin + Float(fi) * step
                for fj in 0 ..< gs {
                    let y = bmin + Float(fj) * step
                    for fk in 0 ..< gs where nxt[(fi * gs + fj) * gs + fk] {
                        nidx.append((fi * gs + fj) * gs + fk)
                        pts.append(x); pts.append(y); pts.append(bmin + Float(fk) * step)
                    }
                }
            }
            let vals = decodePoints(kv: kv, pts: pts, count: nidx.count, numChunks: numChunks)
            var newGrid = [Float](repeating: Float.nan, count: gs * gs * gs)
            for m in 0 ..< nidx.count { newGrid[nidx[m]] = vals[m] }
            grid = newGrid; gn = gs
        }
        return MLXArray(grid, [gn, gn, gn])
    }

    /// Cells bracketing the iso-surface: sign change vs a 6-neighbor, or |value+alpha| < 0.95.
    /// NaN cells (inactive from a prior level) are excluded. Mirrors _near_surface_mask.
    static func nearSurfaceMask(_ g: [Float], _ n: Int, _ alpha: Float) -> [Bool] {
        @inline(__always) func id(_ i: Int, _ j: Int, _ k: Int) -> Int { (i * n + j) * n + k }
        @inline(__always) func sgn(_ v: Float) -> Float { let x = v + alpha; return x > 0 ? 1 : (x < 0 ? -1 : 0) }
        var near = [Bool](repeating: false, count: n * n * n)
        for t in 0 ..< n * n * n {                                       // |value| < 0.95
            let v = g[t]; if !v.isNaN && abs(v + alpha) < 0.95 { near[t] = true }
        }
        for i in 0 ..< n {                                              // sign changes along each axis
            for j in 0 ..< n {
                for k in 0 ..< n {
                    let a = g[id(i, j, k)]; if a.isNaN { continue }
                    let sa = sgn(a)
                    if i + 1 < n { let b = g[id(i + 1, j, k)]; if !b.isNaN && sa != sgn(b) { near[id(i, j, k)] = true; near[id(i + 1, j, k)] = true } }
                    if j + 1 < n { let b = g[id(i, j + 1, k)]; if !b.isNaN && sa != sgn(b) { near[id(i, j, k)] = true; near[id(i, j + 1, k)] = true } }
                    if k + 1 < n { let b = g[id(i, j, k + 1)]; if !b.isNaN && sa != sgn(b) { near[id(i, j, k)] = true; near[id(i, j, k + 1)] = true } }
                }
            }
        }
        for t in 0 ..< n * n * n where g[t].isNaN { near[t] = false }    // near &= valid
        return near
    }

    /// Binary dilation with a full 3x3x3 structuring element (26-neighbourhood + self).
    static func dilate(_ m: [Bool], _ n: Int) -> [Bool] {
        @inline(__always) func id(_ i: Int, _ j: Int, _ k: Int) -> Int { (i * n + j) * n + k }
        var out = [Bool](repeating: false, count: n * n * n)
        for i in 0 ..< n {
            for j in 0 ..< n {
                for k in 0 ..< n where m[id(i, j, k)] {
                    for di in -1 ... 1 where i + di >= 0 && i + di < n {
                        for dj in -1 ... 1 where j + dj >= 0 && j + dj < n {
                            for dk in -1 ... 1 where k + dk >= 0 && k + dk < n {
                                out[id(i + di, j + dj, k + dk)] = true
                            }
                        }
                    }
                }
            }
        }
        return out
    }
}
