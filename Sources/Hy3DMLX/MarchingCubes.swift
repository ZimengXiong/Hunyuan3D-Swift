import Foundation
import MLX

/// A triangle mesh in world space.
public struct Mesh {
    public var vertices: [SIMD3<Float>]
    public var faces: [(UInt32, UInt32, UInt32)]
}

/// Marching cubes over an (n,n,n) SDF grid using lewiner-derived tables (MCTables, generated from
/// skimage). Vertices interpolated at `level`, deduped by grid edge => watertight, smooth surface
/// matching the Python reference. Pure Swift (the table is a baked constant; no runtime Python).
public enum MarchingCubes {
    private static func corner(_ c: Int) -> (Int, Int, Int) { (c & 1, (c >> 1) & 1, (c >> 2) & 1) }

    public static func extract(grid: MLXArray, level: Float = 0.0,
                               bboxMin: Float = -1.01, bboxMax: Float = 1.01) -> Mesh {
        let n = grid.dim(0)
        let g = grid.asType(.float32).asArray(Float.self)
        @inline(__always) func idx(_ i: Int, _ j: Int, _ k: Int) -> Int { (i * n + j) * n + k }
        @inline(__always) func val(_ i: Int, _ j: Int, _ k: Int) -> Float { g[idx(i, j, k)] }
        let edgeCorners = MCTables.edgeCorners
        let triTable = MCTables.triTable

        var verts: [SIMD3<Float>] = []
        var faces: [(UInt32, UInt32, UInt32)] = []
        var edgeCache: [Int64: UInt32] = [:]
        let nnn = Int64(n * n * n)

        func edgeVertex(_ ai: (Int, Int, Int), _ bi: (Int, Int, Int), _ fa: Float, _ fb: Float) -> UInt32 {
            let ga = Int64(idx(ai.0, ai.1, ai.2)), gb = Int64(idx(bi.0, bi.1, bi.2))
            let key = ga < gb ? ga * nnn + gb : gb * nnn + ga
            if let v = edgeCache[key] { return v }
            let t = (level - fa) / (fb - fa)
            let p = SIMD3<Float>(Float(ai.0) + t * Float(bi.0 - ai.0),
                                 Float(ai.1) + t * Float(bi.1 - ai.1),
                                 Float(ai.2) + t * Float(bi.2 - ai.2))
            let v = UInt32(verts.count); verts.append(p); edgeCache[key] = v
            return v
        }

        var cc = [(Int, Int, Int)](repeating: (0, 0, 0), count: 8)
        var cv = [Float](repeating: 0, count: 8)
        for i in 0 ..< (n - 1) {
            for j in 0 ..< (n - 1) {
                for k in 0 ..< (n - 1) {
                    var config = 0
                    var hasNaN = false
                    for c in 0 ..< 8 {
                        let (dx, dy, dz) = corner(c)
                        cc[c] = (i + dx, j + dy, k + dz)
                        let v = val(i + dx, j + dy, k + dz); cv[c] = v
                        if v.isNaN { hasNaN = true }
                        if v < level { config |= 1 << c }
                    }
                    if hasNaN { continue }   // octree band edge: NaN edges don't cross the iso-level
                    let tris = triTable[config]
                    if tris.isEmpty { continue }
                    var m = 0
                    while m < tris.count {
                        var tri = [UInt32](repeating: 0, count: 3)
                        for t in 0 ..< 3 {
                            let (a, b) = edgeCorners[tris[m + t]]
                            tri[t] = edgeVertex(cc[a], cc[b], cv[a], cv[b])
                        }
                        faces.append((tri[0], tri[1], tri[2]))
                        m += 3
                    }
                }
            }
        }
        let s = (bboxMax - bboxMin) / Float(n - 1)
        let world = verts.map { SIMD3<Float>($0.x * s + bboxMin, $0.y * s + bboxMin, $0.z * s + bboxMin) }
        return Mesh(vertices: world, faces: faces)
    }
}
