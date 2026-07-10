import Foundation
import MLX

/// Mesh renderer — Swift port of mesh_render.py (cr-faithful rasterizer, AA off, NHWC).
/// Cameras + normalize in Swift Float; vertex transforms + bake in MLX; rasterize via SwiftRaster.
public final class MeshRender {
    let cameraDistance: Float
    let orthoScale: Float
    let scaleFactor: Float

    public private(set) var vtxPos: MLXArray = MLXArray([Float]())   // [V,3] normalized
    var posIdx: [Int32] = []                                          // [F*3]
    var faceCount = 0
    var vCount = 0
    var vtxUv: MLXArray = MLXArray([Float]())                         // [V,2] (v-flipped)
    var worldVN: MLXArray = MLXArray([Float]())                       // [V,3]
    private var projM: [Float]                                        // 4x4 row-major
    private var faceArr: MLXArray = MLXArray([Int32]())               // [F,3]

    public init(cameraDistance: Float = 1.45, orthoScale: Float = 1.2, scaleFactor: Float = 1.15) {
        self.cameraDistance = cameraDistance; self.orthoScale = orthoScale; self.scaleFactor = scaleFactor
        self.projM = MeshRender.orthoProj(scale: orthoScale)
    }

    // ---- camera matrices (computed in Double, cast to Float — matches numpy float64→float32) ----
    static func orthoProj(scale: Float, near: Double = 0, far: Double = 2) -> [Float] {
        let s = Double(scale)
        let l = -s*0.5, r = s*0.5, b = -s*0.5, t = s*0.5
        var m = [Double](repeating: 0, count: 16); m[15] = 1
        m[0] = 2/(r-l); m[5] = 2/(t-b); m[10] = -2/(far-near)
        m[3] = -(r+l)/(r-l); m[7] = -(t+b)/(t-b); m[11] = -(far+near)/(far-near)
        return m.map { Float($0) }
    }
    static func mvMatrix(elev: Float, azim: Float, dist: Float) -> [Float] {
        let e = -Double(elev), a = Double(azim) + 90, d = Double(dist)
        let er = e * .pi/180, ar = a * .pi/180
        let cam = [d*cos(er)*cos(ar), d*cos(er)*sin(ar), d*sin(er)]
        var lookat = [-cam[0], -cam[1], -cam[2]]
        let ln = norm(lookat); lookat = lookat.map { $0/ln }
        var up = [0.0, 0.0, 1.0]
        var right = cross(lookat, up); let rn = norm(right); right = right.map { $0/rn }
        up = cross(right, lookat); let un = norm(up); up = up.map { $0/un }
        let nl = [-lookat[0], -lookat[1], -lookat[2]]
        var w = [Double](repeating: 0, count: 16); w[15] = 1
        let rows = [right, up, nl]
        for i in 0..<3 {
            w[i*4+0] = rows[i][0]; w[i*4+1] = rows[i][1]; w[i*4+2] = rows[i][2]
            w[i*4+3] = -(rows[i][0]*cam[0] + rows[i][1]*cam[1] + rows[i][2]*cam[2])
        }
        return w.map { Float($0) }
    }
    static func cross(_ a: [Double], _ b: [Double]) -> [Double] {
        [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    }
    static func norm(_ a: [Double]) -> Double { (a[0]*a[0]+a[1]*a[1]+a[2]*a[2]).squareRoot() }

    // ---- mesh setup ----
    public func loadMesh(_ vertices: [Float], _ faces: [UInt32]) {
        vCount = vertices.count / 3; faceCount = faces.count / 3
        var v = vertices
        for i in 0..<vCount {                                   // flip X,Y ; swap Y,Z
            let x = -v[i*3+0], y = -v[i*3+1], z = v[i*3+2]
            v[i*3+0] = x; v[i*3+1] = z; v[i*3+2] = y
        }
        var mn = [Float](repeating: .greatestFiniteMagnitude, count: 3)
        var mx_ = [Float](repeating: -.greatestFiniteMagnitude, count: 3)
        for i in 0..<vCount { for k in 0..<3 { mn[k] = min(mn[k], v[i*3+k]); mx_[k] = max(mx_[k], v[i*3+k]) } }
        let ctr = [(mn[0]+mx_[0])/2, (mn[1]+mx_[1])/2, (mn[2]+mx_[2])/2]
        var maxr: Float = 0
        for i in 0..<vCount {
            let d = ((v[i*3]-ctr[0])*(v[i*3]-ctr[0]) + (v[i*3+1]-ctr[1])*(v[i*3+1]-ctr[1]) + (v[i*3+2]-ctr[2])*(v[i*3+2]-ctr[2])).squareRoot()
            maxr = max(maxr, d)
        }
        let scale = maxr * 2.0, s = scaleFactor / scale
        for i in 0..<vCount { for k in 0..<3 { v[i*3+k] = (v[i*3+k]-ctr[k]) * s } }
        vtxPos = MLXArray(v, [vCount, 3])
        posIdx = faces.map { Int32($0) }
        faceArr = MLXArray(posIdx, [faceCount, 3])
        worldVN = MeshRender.meanVertexNormals(v, posIdx, vCount, faceCount)
    }

    public func setUV(_ uv: [Float], flipV: Bool = true) {
        var u = uv
        if flipV { for i in 0..<(u.count/2) { u[i*2+1] = 1 - u[i*2+1] } }
        vtxUv = MLXArray(u, [u.count/2, 2])
    }

    static func meanVertexNormals(_ v: [Float], _ f: [Int32], _ nv: Int, _ nf: Int) -> MLXArray {
        var vn = [Double](repeating: 0, count: nv*3)
        for t in 0..<nf {
            let a = Int(f[t*3]), b = Int(f[t*3+1]), c = Int(f[t*3+2])
            let e1 = [v[b*3]-v[a*3], v[b*3+1]-v[a*3+1], v[b*3+2]-v[a*3+2]]
            let e2 = [v[c*3]-v[a*3], v[c*3+1]-v[a*3+1], v[c*3+2]-v[a*3+2]]
            var n = [Double(e1[1]*e2[2]-e1[2]*e2[1]), Double(e1[2]*e2[0]-e1[0]*e2[2]), Double(e1[0]*e2[1]-e1[1]*e2[0])]
            let ln = (n[0]*n[0]+n[1]*n[1]+n[2]*n[2]).squareRoot()
            let inv = ln < 1e-12 ? 0 : 1.0/ln; n = n.map { $0*inv }
            for k in 0..<3 { vn[a*3+k]+=n[k]; vn[b*3+k]+=n[k]; vn[c*3+k]+=n[k] }
        }
        var out = [Float](repeating: 0, count: nv*3)
        for i in 0..<nv {
            let ln = (vn[i*3]*vn[i*3]+vn[i*3+1]*vn[i*3+1]+vn[i*3+2]*vn[i*3+2]).squareRoot()
            let inv = ln < 1e-12 ? 0 : 1.0/ln
            for k in 0..<3 { out[i*3+k] = Float(vn[i*3+k]*inv) }
        }
        return MLXArray(out, [nv, 3])
    }

    // ---- transforms ----
    private func mvArray(_ elev: Float, _ azim: Float) -> MLXArray {
        MLXArray(MeshRender.mvMatrix(elev: elev, azim: azim, dist: cameraDistance), [4, 4])
    }
    func project(_ pos: MLXArray, _ elev: Float, _ azim: Float) -> (cam: MLXArray, clip: MLXArray) {
        let n = pos.dim(0)
        let posw = concatenated([pos, ones([n, 1])], axis: 1)
        let cam = matmul(posw, mvArray(elev, azim).transposed(1, 0))
        let pc = matmul(cam, MLXArray(projM, [4, 4]).transposed(1, 0))
        return (cam, pc)
    }

    /// normal (abs/world) + position control maps, NHWC [res,res,3] in [0,1].
    public func renderControl(_ elev: Float, _ azim: Float, _ res: Int, bg: Float = 1) -> (MLXArray, MLXArray) {
        let (_, pc) = project(vtxPos, elev, azim)
        let (fi, ba) = SwiftRaster.rasterize(pc, faceArr, res)
        let maskF = (fi .> 0).reshaped([res, res, 1]).asType(.float32)
        let bgv = MLXArray(bg)
        var normal = SwiftRaster.interpolate(worldVN, fi, ba, faceArr)
        normal = clip(((normal * maskF + bgv * (1 - maskF)) + 1) * 0.5, min: 0, max: 1)
        let texPos = (0.5 - vtxPos / scaleFactor).asType(.float32)
        var position = SwiftRaster.interpolate(texPos, fi, ba, faceArr)
        position = clip(position * maskF + bgv * (1 - maskF), min: 0, max: 1)
        return (normal, position)
    }

    /// UV-space rasterize → (tex_pos [T,T,3], tex_nrm [T,T,3], mask [T,T] bool).
    public func uvRasterize(_ texRes: Int) -> (MLXArray, MLXArray, MLXArray) {
        let nv = vtxUv.dim(0)
        let u = vtxUv[0..., 0].reshaped([nv, 1]), v = vtxUv[0..., 1].reshaped([nv, 1])
        let clipv = concatenated([u * 2 - 1, v * 2 - 1, zeros([nv, 1]), ones([nv, 1])], axis: 1)
        let (fi, ba) = SwiftRaster.rasterize(clipv, faceArr, texRes)
        let texPos = SwiftRaster.interpolate(vtxPos, fi, ba, faceArr)
        let texNrm = SwiftRaster.interpolate(worldVN, fi, ba, faceArr)
        return (texPos, texNrm, fi .> 0)
    }

    /// Debug: render a texture onto the mesh at (elev,azim) using vtxUv. → [res,res,3], white bg.
    public func renderTextured(_ elev: Float, _ azim: Float, _ res: Int, _ tex: MLXArray) -> MLXArray {
        let (_, pc) = project(vtxPos, elev, azim)
        let (fi, ba) = SwiftRaster.rasterize(pc, faceArr, res)
        let uvm = SwiftRaster.interpolate(vtxUv, fi, ba, faceArr)             // [res,res,2]
        let T = tex.dim(0)
        let rf = clip(uvm[0..., 0..., 1] * Float(T - 1), min: 0, max: Float(T - 1)).reshaped([res * res])
        let cf = clip(uvm[0..., 0..., 0] * Float(T - 1), min: 0, max: Float(T - 1)).reshaped([res * res])
        let col = MeshRender.bilinear(tex, rf, cf).reshaped([res, res, 3])
        let bgm = (fi .> 0).reshaped([res, res, 1]).asType(.float32)
        return col * bgm + (1 - bgm)
    }

    /// Bilinear gather: img [H,W,C] at (rowF,colF) [K] → [K,C].
    static func bilinear(_ img: MLXArray, _ rowF: MLXArray, _ colF: MLXArray) -> MLXArray {
        let H = img.dim(0), W = img.dim(1), C = img.dim(2)
        let r0 = clip(floor(rowF), min: 0, max: Float(H - 2)).asType(.int32)
        let c0 = clip(floor(colF), min: 0, max: Float(W - 2)).asType(.int32)
        let fr = (rowF - r0.asType(.float32)).reshaped([-1, 1])
        let fc = (colF - c0.asType(.float32)).reshaped([-1, 1])
        let imgF = img.reshaped([H * W, C])
        let i00 = r0 * Int32(W) + c0
        let g00 = take(imgF, i00, axis: 0), g01 = take(imgF, i00 + 1, axis: 0)
        let g10 = take(imgF, i00 + Int32(W), axis: 0), g11 = take(imgF, i00 + Int32(W) + 1, axis: 0)
        return g00 * (1 - fr) * (1 - fc) + g01 * (1 - fr) * fc + g10 * fr * (1 - fc) + g11 * fr * fc
    }

    /// Back-sample (gather) bake of several color sets sharing one geometry pass.
    /// viewSets: [set][view] = MLXArray [H,W,3]. Returns ([texture per set], covered mask).
    public func bakeMulti(_ viewSets: [[MLXArray]], _ elevs: [Float], _ azims: [Float],
                          textureSize: Int, exp: Float = 6, weights: [Float]? = nil, eps: Float = 0.05)
        -> (textures: [MLXArray], covered: MLXArray) {
        let w = weights ?? [Float](repeating: 1, count: elevs.count)
        let T = textureSize
        let (texPos3, texNrm3, _) = uvRasterize(T)
        let K = T * T
        let P = texPos3.reshaped([K, 3]), Nn = texNrm3.reshaped([K, 3])
        let Pw = concatenated([P, ones([K, 1])], axis: 1)            // [K,4]
        let nsets = viewSets.count
        var accs = (0..<nsets).map { _ in MLX.zeros([K, 3]) }
        var wsum = MLX.zeros([K, 1])
        let cosThr = cos(75.0 * Float.pi / 180)
        let proj = MLXArray(projM, [4, 4])
        let vertPosw = concatenated([vtxPos, ones([vtxPos.dim(0), 1])], axis: 1)
        for vi in 0..<elevs.count {
            let H = viewSets[0][vi].dim(0), Wd = viewSets[0][vi].dim(1)
            let mv = mvArray(elevs[vi], azims[vi])
            let posCamAll = matmul(vertPosw, mv.transposed(1, 0))
            let posClipAll = matmul(posCamAll, proj.transposed(1, 0))
            let (fiD, baD) = SwiftRaster.rasterize(posClipAll, faceArr, H)
            let depthMap = SwiftRaster.interpolate(posCamAll[0..., 2..<3], fiD, baD, faceArr).reshaped([H * Wd])
            let covd = (fiD .> 0).reshaped([H * Wd])
            let pc = matmul(Pw, mv.transposed(1, 0))                 // [K,4]
            let pcp = matmul(pc, proj.transposed(1, 0))
            let ndc0 = pcp[0..., 0] / pcp[0..., 3], ndc1 = pcp[0..., 1] / pcp[0..., 3]
            let zc = pc[0..., 2]
            let colF = (ndc0 * 0.5 + 0.5) * Float(Wd - 1) + 0.5
            let rowF = (0.5 + 0.5 * ndc1) * Float(H - 1) + 0.5
            let inside = (colF .>= 0) .&& (colF .<= Float(Wd - 1)) .&& (rowF .>= 0) .&& (rowF .<= Float(H - 1))
            let ri = clip(rowF.asType(.int32), min: 0, max: Int32(H - 1))
            let ci = clip(colF.asType(.int32), min: 0, max: Int32(Wd - 1))
            let flat = ri * Int32(Wd) + ci
            let covAt = take(covd, flat, axis: 0)
            let depthAt = take(depthMap, flat, axis: 0)
            let vis = (inside .&& covAt .&& (abs(zc - depthAt) .< eps)).asType(.float32)
            let camN = matmul(Nn, mv[0..<3, 0..<3].transposed(1, 0))
            let nrm = sqrt(sum(camN * camN, axis: 1))
            let cosv = -camN[0..., 2] / clip(nrm, min: 1e-8, max: Float.greatestFiniteMagnitude)
            let cosw = MLX.where(cosv .>= cosThr, pow(clip(cosv, min: 0, max: Float.greatestFiniteMagnitude), exp), MLXArray(Float(0))) * w[vi]
            let wgt = (vis * cosw).reshaped([K, 1])
            wsum = wsum + wgt
            let rfc = clip(rowF, min: 0, max: Float(H - 1)), cfc = clip(colF, min: 0, max: Float(Wd - 1))
            for si in 0..<nsets { accs[si] = accs[si] + MeshRender.bilinear(viewSets[si][vi], rfc, cfc) * wgt }
        }
        let wsafe = clip(wsum, min: 1e-8, max: Float.greatestFiniteMagnitude)
        let covered = (wsum[0..., 0] .> 1e-8).reshaped([T, T])
        let texs = accs.map { ($0 / wsafe).reshaped([T, T, 3]) }
        return (texs, covered)
    }

    /// Fill un-painted texels by iterative neighbor diffusion (covers holes; visible texels untouched).
    public static func inpaint(_ texture: MLXArray, _ mask: MLXArray, iters: Int = 64) -> MLXArray {
        let T = texture.dim(0)
        var img = clip(texture, min: 0, max: 1)
        var filled = mask.reshaped([T, T, 1]).asType(.float32)
        img = img * filled
        for _ in 0..<iters {
            // sum of 4-neighbor values + counts via shifts (zero-padded)
            func shift(_ a: MLXArray, _ dy: Int, _ dx: Int) -> MLXArray {
                let py = dy > 0 ? [dy, 0] : [0, -dy], px = dx > 0 ? [dx, 0] : [0, -dx]
                let p = padded(a, widths: [IntOrPair((py[0], py[1])), IntOrPair((px[0], px[1])), IntOrPair((0, 0))])
                return p[(dy > 0 ? 0 : -dy)..<(dy > 0 ? T : T - dy), (dx > 0 ? 0 : -dx)..<(dx > 0 ? T : T - dx), 0...]
            }
            let nbrSum = shift(img, 1, 0) + shift(img, -1, 0) + shift(img, 0, 1) + shift(img, 0, -1)
            let cntSum = shift(filled, 1, 0) + shift(filled, -1, 0) + shift(filled, 0, 1) + shift(filled, 0, -1)
            let canFill = (1 - filled) * (cntSum .> 0).asType(.float32)
            let avg = nbrSum / clip(cntSum, min: 1, max: Float(4))
            img = img + canFill * avg
            filled = clip(filled + canFill, min: 0, max: 1)
        }
        return clip(img, min: 0, max: 1)
    }
}
