#ifndef CXATLAS_H
#define CXATLAS_H
#include <stdint.h>
typedef struct {
    uint32_t vertexCount;   // unwrapped vertex count
    uint32_t indexCount;    // faceCount*3
    float*    uv;           // [vertexCount*2], normalized [0,1]
    uint32_t* xref;         // [vertexCount] original vertex index (vmapping)
    uint32_t* indices;      // [indexCount] unwrapped triangle indices
} XatlasResult;
#ifdef __cplusplus
extern "C" {
#endif
XatlasResult* xatlas_unwrap(const float* positions, uint32_t vertexCount,
                            const uint32_t* indices, uint32_t faceCount);
void xatlas_free(XatlasResult* r);
#ifdef __cplusplus
}
#endif
#endif
