import Foundation
import CXatlas

/// xatlas UV unwrap result. `vmapping[i]` is the original vertex index for unwrapped vertex i;
/// `uvs` are [vertexCount*2] in [0,1]; `indices` are [faceCount*3] into the unwrapped vertices.
public struct UVUnwrap {
    public let vmapping: [UInt32]
    public let uvs: [Float]
    public let indices: [UInt32]
    public let vertexCount: Int
}

/// Unwrap a mesh with xatlas (vendored C++). vertices: flat [vertexCount*3]; faces: flat [faceCount*3].
/// Returns nil on empty/degenerate/non-manifold input (xatlas rejects out-of-range indices etc.)
/// so the caller can fail the job gracefully instead of crashing.
public func xatlasUnwrap(vertices: [Float], vertexCount: Int, faces: [UInt32], faceCount: Int) -> UVUnwrap? {
    guard vertexCount > 0, faceCount > 0,
          vertices.count >= vertexCount * 3, faces.count >= faceCount * 3 else { return nil }
    let res = vertices.withUnsafeBufferPointer { vp in
        faces.withUnsafeBufferPointer { fp in
            xatlas_unwrap(vp.baseAddress, UInt32(vertexCount), fp.baseAddress, UInt32(faceCount))
        }
    }
    guard let r = res else { return nil }
    let vc = Int(r.pointee.vertexCount), ic = Int(r.pointee.indexCount)
    let uvs = Array(UnsafeBufferPointer(start: r.pointee.uv, count: vc * 2))
    let xref = Array(UnsafeBufferPointer(start: r.pointee.xref, count: vc))
    let idx = Array(UnsafeBufferPointer(start: r.pointee.indices, count: ic))
    xatlas_free(r)
    return UVUnwrap(vmapping: xref, uvs: uvs, indices: idx, vertexCount: vc)
}
