import Foundation
import simd

/// Minimal binary glTF (.glb) writer + reader: one mesh with POSITION + indices.
public enum GLB {
    public enum GLBError: Error { case malformed }

    /// Read back a `.glb` written by `write` (POSITION accessor 1, indices accessor 0).
    /// Throws `GLBError.malformed` rather than trapping on a truncated/unexpected file.
    public static func read(_ url: URL) throws -> Mesh {
        let data = try Data(contentsOf: url)
        guard data.count >= 12 else { throw GLBError.malformed }
        func u32(_ off: Int) -> UInt32 { data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: off, as: UInt32.self) } }
        var off = 12, jsonData = Data(), bin = Data()        // skip 12-byte header
        while off + 8 <= data.count {
            let len = Int(u32(off)); let type = u32(off + 4); off += 8
            guard len >= 0, off + len <= data.count else { throw GLBError.malformed }
            let chunk = data.subdata(in: off ..< off + len); off += len
            if type == 0x4E4F_534A { jsonData = chunk }       // "JSON"
            else if type == 0x004E_4942 { bin = chunk }       // "BIN\0"
        }
        guard let json = (try? JSONSerialization.jsonObject(with: jsonData)) as? [String: Any],
              let accessors = json["accessors"] as? [[String: Any]],
              let views = json["bufferViews"] as? [[String: Any]], accessors.count >= 2
        else { throw GLBError.malformed }
        func loc(_ a: [String: Any]) -> (off: Int, count: Int)? {
            guard let bv = a["bufferView"] as? Int, bv < views.count, let count = a["count"] as? Int
            else { return nil }
            return ((views[bv]["byteOffset"] as? Int) ?? 0, count)
        }
        guard let (iOff, iCount) = loc(accessors[0]), let (pOff, pCount) = loc(accessors[1]),
              iOff >= 0, pOff >= 0, iOff + iCount * 4 <= bin.count, pOff + pCount * 12 <= bin.count
        else { throw GLBError.malformed }
        let idx: [UInt32] = bin.subdata(in: iOff ..< iOff + iCount * 4)
            .withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
        let pos: [Float] = bin.subdata(in: pOff ..< pOff + pCount * 12)
            .withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        var verts = [SIMD3<Float>](); verts.reserveCapacity(pCount)
        for i in 0 ..< pCount { verts.append(SIMD3(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2])) }
        var faces = [(UInt32, UInt32, UInt32)](); faces.reserveCapacity(iCount / 3)
        for i in stride(from: 0, to: iCount, by: 3) { faces.append((idx[i], idx[i + 1], idx[i + 2])) }
        return Mesh(vertices: verts, faces: faces)
    }

    public static func write(_ mesh: Mesh, to url: URL) throws {
        var indices = [UInt32](); indices.reserveCapacity(mesh.faces.count * 3)
        for f in mesh.faces { indices.append(f.0); indices.append(f.1); indices.append(f.2) }
        var positions = [Float](); positions.reserveCapacity(mesh.vertices.count * 3)
        var mn = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var mx = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for v in mesh.vertices {
            positions.append(v.x); positions.append(v.y); positions.append(v.z)
            mn = simd_min(mn, v); mx = simd_max(mx, v)
        }
        let idxData = indices.withUnsafeBytes { Data($0) }      // UInt32 LE
        let posData = positions.withUnsafeBytes { Data($0) }    // Float LE (offset 4-aligned: idx is *4)
        var bin = Data(); bin.append(idxData); bin.append(posData)
        while bin.count % 4 != 0 { bin.append(0) }

        let json = """
        {"asset":{"version":"2.0"},"buffers":[{"byteLength":\(bin.count)}],\
        "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":\(idxData.count),"target":34963},\
        {"buffer":0,"byteOffset":\(idxData.count),"byteLength":\(posData.count),"target":34962}],\
        "accessors":[{"bufferView":0,"componentType":5125,"count":\(indices.count),"type":"SCALAR"},\
        {"bufferView":1,"componentType":5126,"count":\(mesh.vertices.count),"type":"VEC3",\
        "min":[\(mn.x),\(mn.y),\(mn.z)],"max":[\(mx.x),\(mx.y),\(mx.z)]}],\
        "meshes":[{"primitives":[{"attributes":{"POSITION":1},"indices":0}]}],\
        "nodes":[{"mesh":0}],"scenes":[{"nodes":[0]}],"scene":0}
        """
        var jsonData = json.data(using: .utf8)!
        while jsonData.count % 4 != 0 { jsonData.append(0x20) }   // pad JSON with spaces

        func u32(_ v: UInt32) -> Data { var x = v.littleEndian; return Data(bytes: &x, count: 4) }
        var glb = Data()
        glb.append("glTF".data(using: .ascii)!); glb.append(u32(2))
        glb.append(u32(UInt32(12 + 8 + jsonData.count + 8 + bin.count)))
        glb.append(u32(UInt32(jsonData.count))); glb.append("JSON".data(using: .ascii)!); glb.append(jsonData)
        glb.append(u32(UInt32(bin.count))); glb.append(Data([0x42, 0x49, 0x4E, 0x00])); glb.append(bin)
        try glb.write(to: url)
    }
}
