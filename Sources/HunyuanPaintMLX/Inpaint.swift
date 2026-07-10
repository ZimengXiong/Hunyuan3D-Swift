import Foundation

/// CPU inpaint that matches the Python reference (`mesh_render.py inpaint`) exactly:
///   1. clip texture to [0,1]
///   2. EDT nearest-fill of uncovered texels — exact port of scipy.ndimage's Euclidean
///      feature transform (Maurer/Qi/Raghavan 2003), including its tie-breaking
///   3. uint8 round-trip + OpenCV `INPAINT_NS` on the holes — exact port of
///      OpenCV's `icvNSInpaintFMM` (fast-marching order, per-pixel estimate, rounding)
/// Both reference algorithms are deterministic; every arithmetic step below mirrors the
/// C/C++ originals (dtype widths, evaluation order, tie rules), so the output is
/// bit-identical to `np.clip → distance_transform_edt gather → cv2.inpaint(..., INPAINT_NS)`.
enum Inpaint {

    // MARK: - Exact scipy Euclidean feature transform (2D)

    /// Nearest-covered-texel index for every pixel (row-major `H*W` arrays of feature
    /// row / col). Exact port of scipy's `NI_EuclideanFeatureTransform` for rank 2:
    /// a per-column pass (axis 0) followed by a per-row Voronoi pass (axis 1) whose
    /// parabola envelope + monotone query scan reproduce scipy's tie-breaking.
    static func edtIndices(covered: [Bool], H: Int, W: Int) -> (rows: [Int32], cols: [Int32]) {
        // Stage 1 (axis 0, per column): nearest covered row within the column,
        // ties -> smaller row (scipy's envelope keeps the earlier site on ties).
        var fRow = [Int32](repeating: -1, count: H * W)
        for c in 0 ..< W {
            var last: Int32 = -1                    // nearest covered row at or above
            for r in 0 ..< H {
                if covered[r * W + c] { last = Int32(r) }
                fRow[r * W + c] = last
            }
            var next: Int32 = -1                    // nearest covered row below
            for r in stride(from: H - 1, through: 0, by: -1) {
                if covered[r * W + c] { next = Int32(r) }
                let up = fRow[r * W + c]
                if up < 0 { fRow[r * W + c] = next }
                else if next >= 0 {
                    let dU = Int32(r) - up, dD = next - Int32(r)
                    if dD < dU { fRow[r * W + c] = next }   // tie (dD == dU) keeps `up`
                }
            }
        }

        // Stage 2 (axis 1, per row): full 2D Voronoi over the per-column sites.
        var outR = [Int32](repeating: 0, count: H * W)
        var outC = [Int32](repeating: 0, count: H * W)
        var g = [Int](repeating: 0, count: W)       // envelope: site column indices
        for r in 0 ..< H {
            var l = -1
            for c in 0 ..< W {                      // build parabola lower envelope
                let fr = fRow[r * W + c]
                guard fr >= 0 else { continue }
                let wR = Double(Int(fr) - r) * Double(Int(fr) - r)
                while l >= 1 {
                    let i1 = g[l], i2 = g[l - 1]
                    let a = Double(i1 - i2)
                    let b = Double(c - i1)
                    let cc = a + b
                    let tu = Double(Int(fRow[r * W + i2]) - r), uR = tu * tu
                    let tv = Double(Int(fRow[r * W + i1]) - r), vR = tv * tv
                    if cc * vR - b * uR - a * wR - a * b * cc <= 0.0 { break }
                    l -= 1
                }
                l += 1
                g[l] = c
            }
            let maxl = l
            guard maxl >= 0 else { continue }       // no covered texel anywhere (caller guards)
            l = 0
            for c in 0 ..< W {                      // monotone query scan; ties stay (earlier site)
                func dist2(_ site: Int) -> Double {
                    let dr = Double(Int(fRow[r * W + site]) - r)
                    let dc = Double(site - c)
                    return dr * dr + dc * dc
                }
                var d1 = dist2(g[l])
                while l < maxl {
                    let d2 = dist2(g[l + 1])
                    if d1 <= d2 { break }
                    d1 = d2
                    l += 1
                }
                outR[r * W + c] = fRow[r * W + g[l]]
                outC[r * W + c] = Int32(g[l])
            }
        }
        return (outR, outC)
    }

    // MARK: - Exact OpenCV Navier-Stokes inpaint (8UC3, inpaintRange 3)

    private struct Heap {                            // min-heap on (T, insertion order)
        var t: [Float] = [], ord: [Int32] = [], pos: [Int32] = []
        var count = 0
        private var nextOrder: Int32 = 0
        mutating func push(_ p: Int32, _ T: Float) {
            t.append(T); ord.append(nextOrder); pos.append(p); nextOrder += 1
            var i = count; count += 1
            while i > 0 {
                let up = (i - 1) >> 1
                if less(i, up) { swapAt(i, up); i = up } else { break }
            }
        }
        mutating func pop() -> Int32? {
            guard count > 0 else { return nil }
            let out = pos[0]
            count -= 1
            if count > 0 {
                t[0] = t[count]; ord[0] = ord[count]; pos[0] = pos[count]
                var i = 0
                while true {
                    let a = 2 * i + 1, b = 2 * i + 2
                    var m = i
                    if a < count, less(a, m) { m = a }
                    if b < count, less(b, m) { m = b }
                    if m == i { break }
                    swapAt(i, m); i = m
                }
            }
            t.removeLast(); ord.removeLast(); pos.removeLast()
            return out
        }
        private func less(_ a: Int, _ b: Int) -> Bool {
            t[a] != t[b] ? t[a] < t[b] : ord[a] < ord[b]
        }
        private mutating func swapAt(_ a: Int, _ b: Int) {
            t.swapAt(a, b); ord.swapAt(a, b); pos.swapAt(a, b)
        }
    }

    private static let KNOWN: UInt8 = 0, BAND: UInt8 = 1, INSIDE: UInt8 = 2

    /// `cv2.inpaint(img, holes, 3, cv2.INPAINT_NS)` for an 8-bit 3-channel image.
    /// `img` is H*W*3 row-major uint8 (modified in place); `holes[r*W+c]` marks texels to fill.
    static func navierStokes(_ img: inout [UInt8], holes: [Bool], H: Int, W: Int, range: Int = 3) {
        let eW = W + 2, eH = H + 2                   // padded state/T grids (OpenCV border)
        var state = [UInt8](repeating: KNOWN, count: eH * eW)
        var tval = [Float](repeating: 1.0e6, count: eH * eW)
        for r in 0 ..< H {
            for c in 0 ..< W where holes[r * W + c] { state[(r + 1) * eW + (c + 1)] = INSIDE }
        }
        var heap = Heap()
        // band = cross-dilate(mask) - mask, border cleared; pushed row-major with T=0
        for pi in 1 ... H {
            for pj in 1 ... W {
                let p = pi * eW + pj
                guard state[p] != INSIDE else { continue }
                if state[p - 1] == INSIDE || state[p + 1] == INSIDE
                    || state[p - eW] == INSIDE || state[p + eW] == INSIDE {
                    tval[p] = 0
                    heap.push(Int32(p), 0)
                }
            }
        }

        // FastMarching_solve: quadratic update from two known-ish neighbours (double math)
        func solve(_ i1: Int, _ j1: Int, _ i2: Int, _ j2: Int) -> Float {
            let p1 = i1 * eW + j1, p2 = i2 * eW + j2
            let a11 = Double(tval[p1]), a22 = Double(tval[p2])
            let m12 = min(a11, a22)
            var sol: Double
            if state[p1] != INSIDE {
                if state[p2] != INSIDE {
                    sol = abs(a11 - a22) >= 1.0
                        ? 1 + m12
                        : (a11 + a22 + (2 - (a11 - a22) * (a11 - a22)).squareRoot()) * 0.5
                } else { sol = 1 + a11 }
            } else if state[p2] != INSIDE { sol = 1 + a22 }
            else { sol = 1 + m12 }
            return Float(sol)
        }

        while let popped = heap.pop() {
            let ii = Int(popped) / eW, jj = Int(popped) % eW
            state[Int(popped)] = KNOWN
            for q in 0 ..< 4 {
                var i = ii, j = jj
                switch q {                            // N, W, S, E — OpenCV's order
                case 0: i = ii - 1
                case 1: j = jj - 1
                case 2: i = ii + 1
                default: j = jj + 1
                }
                if i <= 0 || j <= 0 || i > eH - 1 || j > eW - 1 { continue }
                let p = i * eW + j
                guard state[p] == INSIDE else { continue }

                let dist = min(min(solve(i - 1, j, i, j - 1), solve(i + 1, j, i, j - 1)),
                               min(solve(i - 1, j, i, j + 1), solve(i + 1, j, i, j + 1)))
                tval[p] = dist

                var Ia: (Float, Float, Float) = (0, 0, 0)
                var s: (Float, Float, Float) = (1.0e-20, 1.0e-20, 1.0e-20)
                for k in (i - range) ... (i + range) {
                    let km = k - 1 + (k == 1 ? 1 : 0)
                    let kp = k - 1 - (k == eH - 2 ? 1 : 0)
                    for l in (j - range) ... (j + range) {
                        let lm = l - 1 + (l == 1 ? 1 : 0)
                        let lp = l - 1 - (l == eW - 2 ? 1 : 0)
                        guard k > 0, l > 0, k < eH - 1, l < eW - 1 else { continue }
                        guard state[k * eW + l] != INSIDE,
                              (l - j) * (l - j) + (k - i) * (k - i) <= range * range else { continue }
                        let ry = Float(k - i), rx = Float(l - j)
                        let rLen = rx * rx + ry * ry              // VectorLength = squared length
                        let dst = 1 / (rLen * rLen + 1)
                        let kD = state[(k + 1) * eW + l] != INSIDE   // down not-INSIDE
                        let kU = state[(k - 1) * eW + l] != INSIDE   // up
                        let lR = state[k * eW + (l + 1)] != INSIDE   // right
                        let lL = state[k * eW + (l - 1)] != INSIDE   // left
                        for ch in 0 ..< 3 {
                            @inline(__always) func px(_ r: Int, _ c: Int) -> Int {
                                Int(img[(r * W + c) * 3 + ch])
                            }
                            var gx: Float
                            if kD {
                                gx = kU ? Float(abs(px(kp + 1, lm) - px(kp, lm)) + abs(px(kp, lm) - px(km - 1, lm)))
                                        : Float(abs(px(kp + 1, lm) - px(kp, lm))) * 2
                            } else {
                                gx = kU ? Float(abs(px(kp, lm) - px(km - 1, lm))) * 2 : 0
                            }
                            var gy: Float
                            if lR {
                                gy = lL ? Float(abs(px(km, lp + 1) - px(km, lm)) + abs(px(km, lm) - px(km, lm - 1)))
                                        : Float(abs(px(km, lp + 1) - px(km, lm))) * 2
                            } else {
                                gy = lL ? Float(abs(px(km, lm) - px(km, lm - 1))) * 2 : 0
                            }
                            gx = -gx
                            var dir = rx * gx + ry * gy
                            if abs(Double(dir)) <= 0.01 {
                                dir = 0.000001
                            } else {
                                let gLen = gx * gx + gy * gy
                                dir = Float(abs(Double(rx * gx + ry * gy) / Double(rLen * gLen).squareRoot()))
                            }
                            let w = dst * dir
                            switch ch {
                            case 0: Ia.0 += w * Float(px(k - 1, l - 1)); s.0 += w
                            case 1: Ia.1 += w * Float(px(k - 1, l - 1)); s.1 += w
                            default: Ia.2 += w * Float(px(k - 1, l - 1)); s.2 += w
                            }
                        }
                    }
                }
                let base = ((i - 1) * W + (j - 1)) * 3
                img[base] = satCastU8(Double(Ia.0) / Double(s.0))
                img[base + 1] = satCastU8(Double(Ia.1) / Double(s.1))
                img[base + 2] = satCastU8(Double(Ia.2) / Double(s.2))
                state[p] = BAND
                heap.push(Int32(p), dist)
            }
        }
    }

    /// cv::saturate_cast<uchar>(double): round half to even, clamp to 0...255.
    @inline(__always) private static func satCastU8(_ v: Double) -> UInt8 {
        let r = v.rounded(.toNearestOrEven)
        return UInt8(min(max(r, 0), 255))
    }

    // MARK: - Full reference pipeline (clip -> EDT fill -> u8 round-trip + NS)

    /// `texture` is H*W*3 row-major float32; returns the filled texture (same layout, [0,1]).
    static func fill(texture: [Float], covered: [Bool], H: Int, W: Int) -> [Float] {
        var out = texture.map { min(max($0, 0), 1) }
        let holes = covered.map { !$0 }
        let anyHole = holes.contains(true), anyCovered = covered.contains(true)
        guard anyHole, anyCovered else { return out }

        let (fr, fc) = edtIndices(covered: covered, H: H, W: W)
        var gathered = [Float](repeating: 0, count: H * W * 3)
        for p in 0 ..< H * W {
            let src = (Int(fr[p]) * W + Int(fc[p])) * 3
            gathered[p * 3] = out[src]; gathered[p * 3 + 1] = out[src + 1]; gathered[p * 3 + 2] = out[src + 2]
        }
        // uint8 round-trip exactly as Python: (out * 255).astype(np.uint8) truncates toward zero
        var u8 = [UInt8](repeating: 0, count: H * W * 3)
        for p in 0 ..< u8.count { u8[p] = UInt8(gathered[p] * 255) }
        navierStokes(&u8, holes: holes, H: H, W: W)
        for p in 0 ..< u8.count { out[p] = Float(u8[p]) / 255 }
        return out
    }
}
