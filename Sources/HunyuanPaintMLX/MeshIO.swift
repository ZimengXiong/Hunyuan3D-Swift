import Foundation
import MLX
import ModelIO
import Metal

/// Lightweight mesh loading for the Hunyuan paint pipeline.
///
/// Supports Wavefront `.obj` (parsed manually with Foundation) and
/// `.glb` / `.gltf` (loaded via ModelIO's `MDLAsset`). Geometry is returned
/// as flat float/index arrays suitable for handing directly to MLX or a
/// rasterizer. Vertices are `[x, y, z]` triplets; faces are 0-based triangle
/// index triplets. Polygons are triangulated with a simple triangle fan.
public struct LoadedMesh {
    public let vertices: [Float]   // flat, [vertexCount*3], xyz
    public let faces: [UInt32]     // flat, [faceCount*3], triangle indices
    public let vertexCount: Int
    public let faceCount: Int

    public init(vertices: [Float], faces: [UInt32]) {
        self.vertices = vertices
        self.faces = faces
        self.vertexCount = vertices.count / 3
        self.faceCount = faces.count / 3
    }
}

/// Load a mesh from `path`. Dispatches on file extension: `.obj` is parsed
/// directly, `.glb`/`.gltf` go through ModelIO. Unknown extensions are
/// attempted via ModelIO as a fallback.
public func loadMesh(_ path: String) -> LoadedMesh {
    let ext = (path as NSString).pathExtension.lowercased()
    switch ext {
    case "obj":
        return loadOBJ(path)
    case "glb":
        return loadGLB(path)
    case "gltf":
        return loadModelIO(path)   // external-buffer glTF: fall back to ModelIO
    default:
        return loadModelIO(path)
    }
}

// MARK: - GLB (binary glTF 2.0) — self-contained parser, no ModelIO

/// Parse a binary `.glb`: JSON chunk + BIN chunk, merging all mesh primitives
/// (POSITION VEC3 f32 + triangle indices, widened to UInt32, with vertex-offset).
private func loadGLB(_ path: String) -> LoadedMesh {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)), data.count > 12 else {
        return LoadedMesh(vertices: [], faces: [])
    }
    func u32(_ off: Int) -> Int { Int(data[off]) | Int(data[off+1])<<8 | Int(data[off+2])<<16 | Int(data[off+3])<<24 }
    guard u32(0) == 0x4654_6C67 else { return loadModelIO(path) }       // 'glTF'
    var o = 12
    var json: [String: Any]?
    var bin: Data?
    while o + 8 <= data.count {
        let clen = u32(o), ctype = u32(o + 4); o += 8
        let chunk = data.subdata(in: o ..< min(o + clen, data.count)); o += clen
        if ctype == 0x4E4F_534A { json = try? JSONSerialization.jsonObject(with: chunk) as? [String: Any] }
        else if ctype == 0x004E_4942 { bin = chunk }
    }
    guard let g = json, let buf = bin,
          let accessors = g["accessors"] as? [[String: Any]],
          let bufferViews = g["bufferViews"] as? [[String: Any]],
          let meshes = g["meshes"] as? [[String: Any]] else { return LoadedMesh(vertices: [], faces: []) }

    func readAccessor(_ ai: Int) -> (data: [Double], comps: Int, ct: Int) {
        let a = accessors[ai]; let bv = bufferViews[a["bufferView"] as! Int]
        let off = (bv["byteOffset"] as? Int ?? 0) + (a["byteOffset"] as? Int ?? 0)
        let count = a["count"] as! Int, ct = a["componentType"] as! Int
        let comps = ["SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4][a["type"] as! String]!
        let csize = [5120:1, 5121:1, 5122:2, 5123:2, 5125:4, 5126:4][ct]!
        let stride = (bv["byteStride"] as? Int) ?? (comps * csize)
        var out = [Double](); out.reserveCapacity(count * comps)
        buf.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let base = raw.baseAddress!
            for i in 0 ..< count {
                let row = off + i * stride
                for c in 0 ..< comps {
                    let p = base + row + c * csize
                    switch ct {
                    case 5126: out.append(Double(p.loadUnaligned(as: Float.self)))
                    case 5125: out.append(Double(p.loadUnaligned(as: UInt32.self)))
                    case 5123: out.append(Double(p.loadUnaligned(as: UInt16.self)))
                    case 5121: out.append(Double(p.load(as: UInt8.self)))
                    default: out.append(0)
                    }
                }
            }
        }
        return (out, comps, ct)
    }

    var vertices = [Float](), faces = [UInt32]()
    for mesh in meshes {
        guard let prims = mesh["primitives"] as? [[String: Any]] else { continue }
        for prim in prims {
            guard let attrs = prim["attributes"] as? [String: Any], let posAcc = attrs["POSITION"] as? Int else { continue }
            let vbase = UInt32(vertices.count / 3)
            let pos = readAccessor(posAcc)
            for d in pos.data { vertices.append(Float(d)) }
            if let idxAcc = prim["indices"] as? Int {
                let idx = readAccessor(idxAcc)
                for d in idx.data { faces.append(vbase + UInt32(d)) }
            } else {                                                    // non-indexed: sequential triangles
                let n = pos.data.count / 3
                for i in 0 ..< n { faces.append(vbase + UInt32(i)) }
            }
        }
    }
    return LoadedMesh(vertices: vertices, faces: faces)
}

// MARK: - OBJ

/// Parse a Wavefront `.obj` file using only Foundation string handling.
private func loadOBJ(_ path: String) -> LoadedMesh {
    guard let text = try? String(contentsOfFile: path, encoding: .utf8) else {
        return LoadedMesh(vertices: [], faces: [])
    }

    var vertices: [Float] = []
    var faces: [UInt32] = []

    // Split on any newline variant.
    let lines = text.split(whereSeparator: { $0 == "\n" || $0 == "\r" })

    for rawLine in lines {
        // Trim leading/trailing whitespace.
        let line = rawLine.trimmingCharacters(in: .whitespaces)
        if line.isEmpty || line.hasPrefix("#") { continue }

        // Tokenize on any run of whitespace.
        let tokens = line.split(whereSeparator: { $0 == " " || $0 == "\t" })
        guard let keyword = tokens.first else { continue }

        if keyword == "v" {
            // Vertex: take the first three numeric components, ignore extras
            // (e.g. vertex colors). Missing components default to 0.
            var xyz: [Float] = [0, 0, 0]
            var i = 0
            for t in tokens.dropFirst() {
                if i >= 3 { break }
                xyz[i] = Float(t) ?? 0
                i += 1
            }
            vertices.append(contentsOf: xyz)
        } else if keyword == "f" {
            // Face: each token is one of "v", "v/vt", "v//vn", "v/vt/vn".
            // Take the integer before the first "/" as the 1-based vertex index.
            var idx: [Int] = []
            for t in tokens.dropFirst() {
                let vertField = t.split(separator: "/", maxSplits: 1,
                                        omittingEmptySubsequences: false).first ?? Substring("")
                guard let oneBased = Int(vertField) else { continue }
                // OBJ indices may be negative (relative to end of vertex list).
                let zeroBased: Int
                if oneBased < 0 {
                    zeroBased = (vertices.count / 3) + oneBased
                } else {
                    zeroBased = oneBased - 1
                }
                if zeroBased >= 0 {
                    idx.append(zeroBased)
                }
            }
            // Triangulate polygon with a triangle fan: (0, i, i+1).
            if idx.count >= 3 {
                for i in 1..<(idx.count - 1) {
                    faces.append(UInt32(idx[0]))
                    faces.append(UInt32(idx[i]))
                    faces.append(UInt32(idx[i + 1]))
                }
            }
        }
        // Ignore vt / vn / g / o / s / usemtl / mtllib / everything else.
    }

    return LoadedMesh(vertices: vertices, faces: faces)
}

// MARK: - ModelIO (GLB / glTF)

/// Load a mesh via ModelIO. Reads the first `MDLMesh`, extracts the POSITION
/// attribute (stride-aware) and the triangle index buffers (widening UInt16
/// to UInt32 as needed).
private func loadModelIO(_ path: String) -> LoadedMesh {
    let url = URL(fileURLWithPath: path)

    // Prefer a Metal-backed allocator so buffers are readable; nil also works
    // but an allocator makes ModelIO's behavior more predictable.
    let allocator: MDLMeshBufferAllocator?
    if let device = MTLCreateSystemDefaultDevice() {
        allocator = MTKMeshBufferAllocatorBox.make(device: device)
    } else {
        allocator = nil
    }

    let asset = MDLAsset(url: url,
                         vertexDescriptor: nil,
                         bufferAllocator: allocator)
    asset.loadTextures()

    // Find the first MDLMesh anywhere in the object hierarchy.
    guard let mesh = firstMesh(in: asset) else {
        return LoadedMesh(vertices: [], faces: [])
    }

    // Ensure geometry is triangulated. ModelIO can re-tessellate non-triangle
    // submeshes for us; ignore failures and proceed with whatever we have.
    if let submeshes = mesh.submeshes as? [MDLSubmesh] {
        for submesh in submeshes where submesh.geometryType != .triangles {
            // Audit fix (e): `makeVerticesUnique` is deprecated (macOS 10.13); its supported
            // replacement is the throwing `makeVerticesUniqueAndReturnError()`. (The original also
            // dropped the call's parentheses, making it a no-op; this actually runs it.)
            _ = try? mesh.makeVerticesUniqueAndReturnError()
            break
        }
    }

    let vertices = readPositions(from: mesh)
    let faces = readIndices(from: mesh)
    return LoadedMesh(vertices: vertices, faces: faces)
}

/// Depth-first search for the first `MDLMesh` in an asset's object tree.
private func firstMesh(in asset: MDLAsset) -> MDLMesh? {
    for i in 0..<asset.count {
        if let found = firstMesh(in: asset.object(at: i)) {
            return found
        }
    }
    return nil
}

private func firstMesh(in object: MDLObject) -> MDLMesh? {
    if let mesh = object as? MDLMesh {
        return mesh
    }
    for i in 0..<object.children.count {
        if let found = firstMesh(in: object.children[i]) {
            return found
        }
    }
    return nil
}

/// Read the POSITION vertex attribute into a flat `[x, y, z, ...]` array,
/// honoring the attribute's byte offset and the layout stride.
private func readPositions(from mesh: MDLMesh) -> [Float] {
    guard let vertexDescriptor = mesh.vertexDescriptor as MDLVertexDescriptor? else {
        return []
    }

    // Locate the POSITION attribute.
    var positionAttr: MDLVertexAttribute?
    if let attrs = vertexDescriptor.attributes as? [MDLVertexAttribute] {
        for attr in attrs where attr.name == MDLVertexAttributePosition {
            positionAttr = attr
            break
        }
    }
    guard let attr = positionAttr, attr.format != .invalid else { return [] }

    let layoutIndex = attr.bufferIndex
    guard layoutIndex < mesh.vertexBuffers.count else { return [] }

    // Stride for this buffer's layout.
    var stride = 0
    if let layouts = vertexDescriptor.layouts as? [MDLVertexBufferLayout],
       layoutIndex < layouts.count {
        stride = layouts[layoutIndex].stride
    }
    if stride == 0 {
        // Fall back to a tightly packed 3-float stride.
        stride = MemoryLayout<Float>.size * 3
    }

    let buffer = mesh.vertexBuffers[layoutIndex]
    let map = buffer.map()
    let base = map.bytes
    let count = mesh.vertexCount
    let offset = attr.offset

    var out = [Float]()
    out.reserveCapacity(count * 3)
    let bufLength = buffer.length

    for v in 0..<count {
        let rowOffset = v * stride + offset
        // Bounds guard: need 3 floats available.
        if rowOffset + MemoryLayout<Float>.size * 3 > bufLength { break }
        let ptr = base.advanced(by: rowOffset).assumingMemoryBound(to: Float.self)
        out.append(ptr[0])
        out.append(ptr[1])
        out.append(ptr[2])
    }
    return out
}

/// Read all submesh triangle index buffers into a single flat `[UInt32]`,
/// widening 8/16-bit index types to 32-bit.
private func readIndices(from mesh: MDLMesh) -> [UInt32] {
    guard let submeshes = mesh.submeshes as? [MDLSubmesh] else { return [] }

    var out = [UInt32]()
    for submesh in submeshes {
        guard submesh.geometryType == .triangles else { continue }
        let indexBuffer = submesh.indexBuffer(asIndexType: submesh.indexType)
        let map = indexBuffer.map()
        let base = map.bytes
        let length = indexBuffer.length

        switch submesh.indexType {
        case .uInt8:
            let n = length / MemoryLayout<UInt8>.size
            let ptr = base.assumingMemoryBound(to: UInt8.self)
            out.reserveCapacity(out.count + n)
            for i in 0..<n { out.append(UInt32(ptr[i])) }
        case .uInt16:
            let n = length / MemoryLayout<UInt16>.size
            let ptr = base.assumingMemoryBound(to: UInt16.self)
            out.reserveCapacity(out.count + n)
            for i in 0..<n { out.append(UInt32(ptr[i])) }
        case .uInt32:
            let n = length / MemoryLayout<UInt32>.size
            let ptr = base.assumingMemoryBound(to: UInt32.self)
            out.reserveCapacity(out.count + n)
            for i in 0..<n { out.append(ptr[i]) }
        case .invalid:
            continue
        @unknown default:
            continue
        }
    }
    return out
}

// MARK: - MetalKit allocator helper

import MetalKit

/// Tiny indirection so importing MetalKit only affects this helper.
private enum MTKMeshBufferAllocatorBox {
    static func make(device: MTLDevice) -> MDLMeshBufferAllocator {
        return MTKMeshBufferAllocator(device: device)
    }
}
