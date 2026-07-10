#include "xatlas.h"
#include "CXatlas.h"
#include <cstdlib>
#include <cstring>

extern "C" XatlasResult* xatlas_unwrap(const float* positions, uint32_t vertexCount,
                                       const uint32_t* indices, uint32_t faceCount) {
    xatlas::Atlas* atlas = xatlas::Create();
    xatlas::MeshDecl decl;
    decl.vertexCount = vertexCount;
    decl.vertexPositionData = positions;
    decl.vertexPositionStride = sizeof(float) * 3;
    decl.indexCount = faceCount * 3;
    decl.indexData = indices;
    decl.indexFormat = xatlas::IndexFormat::UInt32;
    if (xatlas::AddMesh(atlas, decl) != xatlas::AddMeshError::Success) { xatlas::Destroy(atlas); return nullptr; }
    xatlas::Generate(atlas);
    const xatlas::Mesh& m = atlas->meshes[0];
    XatlasResult* r = (XatlasResult*)malloc(sizeof(XatlasResult));
    r->vertexCount = m.vertexCount; r->indexCount = m.indexCount;
    r->uv = (float*)malloc(sizeof(float) * m.vertexCount * 2);
    r->xref = (uint32_t*)malloc(sizeof(uint32_t) * m.vertexCount);
    r->indices = (uint32_t*)malloc(sizeof(uint32_t) * m.indexCount);
    float w = (float)atlas->width, h = (float)atlas->height;
    for (uint32_t i = 0; i < m.vertexCount; i++) {
        r->uv[i*2] = m.vertexArray[i].uv[0] / w;
        r->uv[i*2+1] = m.vertexArray[i].uv[1] / h;
        r->xref[i] = m.vertexArray[i].xref;
    }
    memcpy(r->indices, m.indexArray, sizeof(uint32_t) * m.indexCount);
    xatlas::Destroy(atlas);
    return r;
}
extern "C" void xatlas_free(XatlasResult* r) { if (r) { free(r->uv); free(r->xref); free(r->indices); free(r); } }
