import Foundation

// Minimal, dependency-free glTF 2.0 binary (.glb) writer.
//
// Produces a single-buffer GLB containing:
//   - indices (UInt32, SCALAR, componentType 5125)
//   - POSITION (float32 VEC3, componentType 5126, with min/max)
//   - TEXCOORD_0 (float32 VEC2, componentType 5126)
//   - one (or two) PNG image(s) embedded in the buffer
//
// One mesh / one primitive (TRIANGLES) / one material (PBR metallic-roughness,
// doubleSided), one node, one scene.

// MARK: - Little-endian byte helpers

private extension Data {
    mutating func appendLE(_ value: UInt32) {
        var v = value.littleEndian
        Swift.withUnsafeBytes(of: &v) { append(contentsOf: $0) }
    }

    mutating func appendLE(_ value: Float) {
        var v = value.bitPattern.littleEndian
        Swift.withUnsafeBytes(of: &v) { append(contentsOf: $0) }
    }
}

/// Pad `data` to a 4-byte boundary in place using the given pad byte.
private func pad4(_ data: inout Data, with pad: UInt8) {
    let rem = data.count % 4
    if rem != 0 {
        data.append(contentsOf: [UInt8](repeating: pad, count: 4 - rem))
    }
}

/// Round `n` up to the next multiple of 4.
private func align4(_ n: Int) -> Int {
    let rem = n % 4
    return rem == 0 ? n : n + (4 - rem)
}

// MARK: - Public API

/// Write a `.glb` (binary glTF 2.0) file.
///
/// - Parameters:
///   - path: Destination file path.
///   - vertices: Flat `[V*3]` xyz positions (the final mesh geometry).
///   - faces: Flat `[F*3]` triangle indices.
///   - uvs: Flat `[V*2]` TEXCOORD_0 in `[0, 1]`.
///   - baseColorPNG: PNG bytes used for the baseColorTexture.
///   - metallicRoughnessPNG: Optional PNG bytes for the metallicRoughnessTexture.
public func writeGLB(path: String,
                     vertices: [Float],
                     faces: [UInt32],
                     uvs: [Float],
                     baseColorPNG: Data,
                     metallicRoughnessPNG: Data?) throws {

    let vertexCount = vertices.count / 3
    let indexCount = faces.count

    // MARK: Build the binary buffer (concatenated, each section 4-byte aligned).

    var bin = Data()

    // --- indices (UInt32) ---
    let indicesOffset = bin.count
    for idx in faces { bin.appendLE(idx) }
    let indicesLength = bin.count - indicesOffset
    pad4(&bin, with: 0)

    // --- POSITION (float32 vec3) + min/max ---
    let positionOffset = bin.count
    var minPos: [Float] = [Float.greatestFiniteMagnitude,
                           Float.greatestFiniteMagnitude,
                           Float.greatestFiniteMagnitude]
    var maxPos: [Float] = [-Float.greatestFiniteMagnitude,
                           -Float.greatestFiniteMagnitude,
                           -Float.greatestFiniteMagnitude]
    if vertexCount > 0 {
        for v in 0..<vertexCount {
            for c in 0..<3 {
                let value = vertices[v * 3 + c]
                bin.appendLE(value)
                if value < minPos[c] { minPos[c] = value }
                if value > maxPos[c] { maxPos[c] = value }
            }
        }
    } else {
        minPos = [0, 0, 0]
        maxPos = [0, 0, 0]
    }
    let positionLength = bin.count - positionOffset
    pad4(&bin, with: 0)

    // --- TEXCOORD_0 (float32 vec2) ---
    let uvOffset = bin.count
    for u in uvs { bin.appendLE(u) }
    let uvLength = bin.count - uvOffset
    pad4(&bin, with: 0)

    // --- baseColor PNG ---
    let baseColorImageOffset = bin.count
    bin.append(baseColorPNG)
    let baseColorImageLength = baseColorPNG.count
    pad4(&bin, with: 0)

    // --- optional metallicRoughness PNG ---
    var mrImageOffset = 0
    var mrImageLength = 0
    if let mrPNG = metallicRoughnessPNG {
        mrImageOffset = bin.count
        bin.append(mrPNG)
        mrImageLength = mrPNG.count
        pad4(&bin, with: 0)
    }

    let totalBufferLength = bin.count

    // MARK: Build bufferViews.

    // bufferView indices:
    //   0: indices
    //   1: POSITION
    //   2: TEXCOORD_0
    //   3: baseColor image
    //   4: metallicRoughness image (optional)
    var bufferViews: [[String: Any]] = []

    bufferViews.append([
        "buffer": 0,
        "byteOffset": indicesOffset,
        "byteLength": indicesLength,
        "target": 34963  // ELEMENT_ARRAY_BUFFER
    ])
    bufferViews.append([
        "buffer": 0,
        "byteOffset": positionOffset,
        "byteLength": positionLength,
        "byteStride": 12,
        "target": 34962  // ARRAY_BUFFER
    ])
    bufferViews.append([
        "buffer": 0,
        "byteOffset": uvOffset,
        "byteLength": uvLength,
        "byteStride": 8,
        "target": 34962  // ARRAY_BUFFER
    ])
    let baseColorBufferViewIndex = bufferViews.count
    bufferViews.append([
        "buffer": 0,
        "byteOffset": baseColorImageOffset,
        "byteLength": baseColorImageLength
    ])
    var mrBufferViewIndex = -1
    if metallicRoughnessPNG != nil {
        mrBufferViewIndex = bufferViews.count
        bufferViews.append([
            "buffer": 0,
            "byteOffset": mrImageOffset,
            "byteLength": mrImageLength
        ])
    }

    // MARK: Build accessors.

    // accessor indices:
    //   0: indices (SCALAR / UNSIGNED_INT)
    //   1: POSITION (VEC3 / FLOAT) with min/max
    //   2: TEXCOORD_0 (VEC2 / FLOAT)
    let accessors: [[String: Any]] = [
        [
            "bufferView": 0,
            "byteOffset": 0,
            "componentType": 5125,  // UNSIGNED_INT
            "count": indexCount,
            "type": "SCALAR"
        ],
        [
            "bufferView": 1,
            "byteOffset": 0,
            "componentType": 5126,  // FLOAT
            "count": vertexCount,
            "type": "VEC3",
            "min": minPos,
            "max": maxPos
        ],
        [
            "bufferView": 2,
            "byteOffset": 0,
            "componentType": 5126,  // FLOAT
            "count": vertexCount,
            "type": "VEC2"
        ]
    ]

    // MARK: Build images / samplers / textures.

    var images: [[String: Any]] = [
        ["bufferView": baseColorBufferViewIndex, "mimeType": "image/png"]
    ]
    if mrBufferViewIndex >= 0 {
        images.append(["bufferView": mrBufferViewIndex, "mimeType": "image/png"])
    }

    // A single linear sampler reused by all textures.
    let samplers: [[String: Any]] = [
        [
            "magFilter": 9729,  // LINEAR
            "minFilter": 9987,  // LINEAR_MIPMAP_LINEAR
            "wrapS": 10497,     // REPEAT
            "wrapT": 10497      // REPEAT
        ]
    ]

    var textures: [[String: Any]] = [
        ["sampler": 0, "source": 0]  // texture 0 -> baseColor image
    ]
    var mrTextureIndex = -1
    if mrBufferViewIndex >= 0 {
        mrTextureIndex = textures.count
        textures.append(["sampler": 0, "source": 1])  // texture 1 -> MR image
    }

    // MARK: Build material.

    var pbr: [String: Any] = [
        "baseColorTexture": ["index": 0, "texCoord": 0],
        "metallicFactor": 1.0,
        "roughnessFactor": 1.0
    ]
    if mrTextureIndex >= 0 {
        pbr["metallicRoughnessTexture"] = ["index": mrTextureIndex, "texCoord": 0]
    }
    let materials: [[String: Any]] = [
        [
            "pbrMetallicRoughness": pbr,
            "doubleSided": true
        ]
    ]

    // MARK: Build mesh / node / scene.

    let meshes: [[String: Any]] = [
        [
            "primitives": [
                [
                    "attributes": [
                        "POSITION": 1,
                        "TEXCOORD_0": 2
                    ],
                    "indices": 0,
                    "material": 0,
                    "mode": 4  // TRIANGLES
                ]
            ]
        ]
    ]
    let nodes: [[String: Any]] = [["mesh": 0]]
    let scenes: [[String: Any]] = [["nodes": [0]]]

    let buffers: [[String: Any]] = [["byteLength": totalBufferLength]]

    // MARK: Assemble glTF JSON root.

    let gltf: [String: Any] = [
        "asset": ["version": "2.0", "generator": "HunyuanPaintMLX.GLB"],
        "scene": 0,
        "scenes": scenes,
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "textures": textures,
        "images": images,
        "samplers": samplers,
        "accessors": accessors,
        "bufferViews": bufferViews,
        "buffers": buffers
    ]

    var jsonData = try JSONSerialization.data(
        withJSONObject: gltf,
        options: [.sortedKeys]
    )
    // JSON chunk must be padded to a 4-byte boundary with spaces (0x20).
    pad4(&jsonData, with: 0x20)

    // BIN chunk must be padded to a 4-byte boundary with zeros.
    var binChunk = bin
    pad4(&binChunk, with: 0x00)

    // MARK: GLB container.

    let headerLength = 12
    let chunkHeaderLength = 8
    let totalLength = headerLength
        + chunkHeaderLength + jsonData.count
        + chunkHeaderLength + binChunk.count

    var glb = Data()
    // 12-byte header.
    glb.appendLE(UInt32(0x46546C67))      // magic "glTF"
    glb.appendLE(UInt32(2))               // version
    glb.appendLE(UInt32(totalLength))     // total length

    // JSON chunk.
    glb.appendLE(UInt32(jsonData.count))  // chunk length
    glb.appendLE(UInt32(0x4E4F534A))      // chunk type "JSON"
    glb.append(jsonData)

    // BIN chunk.
    glb.appendLE(UInt32(binChunk.count))  // chunk length
    glb.appendLE(UInt32(0x004E4942))      // chunk type "BIN\0"
    glb.append(binChunk)

    try glb.write(to: URL(fileURLWithPath: path))

    _ = align4  // keep helper available; silences unused warning if not used elsewhere
}
