// Torch-free pybind11 build of Hunyuan3D custom_rasterizer's CPU path.
//
// The CUDA kernel (rasterizer_gpu.cu) and this CPU path (rasterize_image_cpu in
// rasterizer.cpp) are algorithmically identical by construction — the CPU twin is
// auto-selected by the original code when tensors live on CPU. We extract the exact
// scalar functions (they already operate on raw pointers, no ATen/torch needed) and
// expose them over numpy. This is the BIT-EXACT parity oracle for the MLX rasterizer:
// matching it == matching the reference CUDA pipeline.
//
// Source: reference/Hunyuan3D-MLX/hy3dgen/texgen/custom_rasterizer/lib/
//         custom_rasterizer_kernel/{rasterizer.h,rasterizer.cpp}
// Transcribed verbatim (only the torch::Tensor I/O replaced by a plain C ABI, so it
// loads via ctypes with zero CPython-ABI coupling — no pybind11, no libpython link).

#include <vector>
#include <algorithm>
#include <cmath>

#define INT64 unsigned long long
#define MAXINT 2147483647

static inline float calculateSignedArea2(float* a, float* b, float* c) {
    return ((c[0] - a[0]) * (b[1] - a[1]) - (b[0] - a[0]) * (c[1] - a[1]));
}

static inline void calculateBarycentricCoordinate(float* a, float* b, float* c, float* p,
    float* barycentric)
{
    float beta_tri = calculateSignedArea2(a, p, c);
    float gamma_tri = calculateSignedArea2(a, b, p);
    float area = calculateSignedArea2(a, b, c);
    if (area == 0) {
        barycentric[0] = -1.0;
        barycentric[1] = -1.0;
        barycentric[2] = -1.0;
        return;
    }
    float tri_inv = 1.0 / area;
    float beta = beta_tri * tri_inv;
    float gamma = gamma_tri * tri_inv;
    float alpha = 1.0 - beta - gamma;
    barycentric[0] = alpha;
    barycentric[1] = beta;
    barycentric[2] = gamma;
}

static inline bool isBarycentricCoordInBounds(float* barycentricCoord) {
    return barycentricCoord[0] >= 0.0 && barycentricCoord[0] <= 1.0 &&
           barycentricCoord[1] >= 0.0 && barycentricCoord[1] <= 1.0 &&
           barycentricCoord[2] >= 0.0 && barycentricCoord[2] <= 1.0;
}

static void rasterizeTriangleCPU(int idx, float* vt0, float* vt1, float* vt2, int width, int height,
    INT64* zbuffer, float* d, float occlusion_truncation) {
    float x_min = std::min(vt0[0], std::min(vt1[0], vt2[0]));
    float x_max = std::max(vt0[0], std::max(vt1[0], vt2[0]));
    float y_min = std::min(vt0[1], std::min(vt1[1], vt2[1]));
    float y_max = std::max(vt0[1], std::max(vt1[1], vt2[1]));

    for (int px = x_min; px < x_max + 1; ++px) {
        if (px < 0 || px >= width) continue;
        for (int py = y_min; py < y_max + 1; ++py) {
            if (py < 0 || py >= height) continue;
            float vt[2] = {px + 0.5f, py + 0.5f};
            float baryCentricCoordinate[3];
            calculateBarycentricCoordinate(vt0, vt1, vt2, vt, baryCentricCoordinate);
            if (isBarycentricCoordInBounds(baryCentricCoordinate)) {
                int pixel = py * width + px;
                if (zbuffer == 0) {
                    zbuffer[pixel] = (INT64)(idx + 1);
                    continue;
                }
                float depth = baryCentricCoordinate[0] * vt0[2] + baryCentricCoordinate[1] * vt1[2] + baryCentricCoordinate[2] * vt2[2];
                float depth_thres = 0;
                if (d) {
                    depth_thres = d[pixel] * 0.49999f + 0.5f + occlusion_truncation;
                }
                int z_quantize = depth * (2 << 17);
                INT64 token = (INT64)z_quantize * MAXINT + (INT64)(idx + 1);
                if (depth < depth_thres) continue;
                zbuffer[pixel] = std::min(zbuffer[pixel], token);
            }
        }
    }
}

static void barycentricFromImgcoordCPU(float* V, int* F, int* findices, INT64* zbuffer, int width, int height,
    int num_vertices, int num_faces, float* barycentric_map, int pix)
{
    INT64 f = zbuffer[pix] % MAXINT;
    if (f == (MAXINT - 1)) {
        findices[pix] = 0;
        barycentric_map[pix * 3] = 0;
        barycentric_map[pix * 3 + 1] = 0;
        barycentric_map[pix * 3 + 2] = 0;
        return;
    }
    findices[pix] = f;
    f -= 1;
    float barycentric[3] = {0, 0, 0};
    if (f >= 0) {
        float vt[2] = {float(pix % width) + 0.5f, float(pix / width) + 0.5f};
        float* vt0_ptr = V + (F[f * 3] * 4);
        float* vt1_ptr = V + (F[f * 3 + 1] * 4);
        float* vt2_ptr = V + (F[f * 3 + 2] * 4);

        float vt0[2] = {(vt0_ptr[0] / vt0_ptr[3] * 0.5f + 0.5f) * (width - 1) + 0.5f, (0.5f + 0.5f * vt0_ptr[1] / vt0_ptr[3]) * (height - 1) + 0.5f};
        float vt1[2] = {(vt1_ptr[0] / vt1_ptr[3] * 0.5f + 0.5f) * (width - 1) + 0.5f, (0.5f + 0.5f * vt1_ptr[1] / vt1_ptr[3]) * (height - 1) + 0.5f};
        float vt2[2] = {(vt2_ptr[0] / vt2_ptr[3] * 0.5f + 0.5f) * (width - 1) + 0.5f, (0.5f + 0.5f * vt2_ptr[1] / vt2_ptr[3]) * (height - 1) + 0.5f};

        calculateBarycentricCoordinate(vt0, vt1, vt2, vt, barycentric);

        barycentric[0] = barycentric[0] / vt0_ptr[3];
        barycentric[1] = barycentric[1] / vt1_ptr[3];
        barycentric[2] = barycentric[2] / vt2_ptr[3];
        float w = 1.0f / (barycentric[0] + barycentric[1] + barycentric[2]);
        barycentric[0] *= w;
        barycentric[1] *= w;
        barycentric[2] *= w;
    }
    barycentric_map[pix * 3] = barycentric[0];
    barycentric_map[pix * 3 + 1] = barycentric[1];
    barycentric_map[pix * 3 + 2] = barycentric[2];
}

static void rasterizeImagecoordsKernelCPU(float* V, int* F, float* d, INT64* zbuffer, float occlusion_trunc,
    int width, int height, int num_vertices, int num_faces, int f)
{
    float* vt0_ptr = V + (F[f * 3] * 4);
    float* vt1_ptr = V + (F[f * 3 + 1] * 4);
    float* vt2_ptr = V + (F[f * 3 + 2] * 4);

    float vt0[3] = {(vt0_ptr[0] / vt0_ptr[3] * 0.5f + 0.5f) * (width - 1) + 0.5f, (0.5f + 0.5f * vt0_ptr[1] / vt0_ptr[3]) * (height - 1) + 0.5f, vt0_ptr[2] / vt0_ptr[3] * 0.49999f + 0.5f};
    float vt1[3] = {(vt1_ptr[0] / vt1_ptr[3] * 0.5f + 0.5f) * (width - 1) + 0.5f, (0.5f + 0.5f * vt1_ptr[1] / vt1_ptr[3]) * (height - 1) + 0.5f, vt1_ptr[2] / vt1_ptr[3] * 0.49999f + 0.5f};
    float vt2[3] = {(vt2_ptr[0] / vt2_ptr[3] * 0.5f + 0.5f) * (width - 1) + 0.5f, (0.5f + 0.5f * vt2_ptr[1] / vt2_ptr[3]) * (height - 1) + 0.5f, vt2_ptr[2] / vt2_ptr[3] * 0.49999f + 0.5f};

    rasterizeTriangleCPU(f, vt0, vt1, vt2, width, height, zbuffer, d, occlusion_trunc);
}

// Plain C ABI entry (called from Python via ctypes).
//   V            : [num_vertices, 4] float32 clip-space homogeneous (x, y, z, w), row-major
//   F            : [num_faces, 3]   int32 vertex indices, row-major
//   findices_out : [height*width]   int32   (0 = background, else 1-based face id)
//   bary_out     : [height*width*3] float32 (perspective-correct barycentric)
// No depth prior (matches the production call use_depth_prior=0).
extern "C" void cr_rasterize(const float* V, const int* F, int num_vertices, int num_faces,
                             int width, int height, int* findices_out, float* bary_out)
{
    float* Vp = const_cast<float*>(V);
    int* Fp = const_cast<int*>(F);

    std::vector<INT64> zbuffer(static_cast<size_t>(width) * height);
    INT64 maxint = (INT64)MAXINT * (INT64)MAXINT + (MAXINT - 1);
    for (auto& z : zbuffer) z = maxint;

    for (int i = 0; i < num_faces; ++i)
        rasterizeImagecoordsKernelCPU(Vp, Fp, 0, zbuffer.data(), 0.0f, width, height, num_vertices, num_faces, i);

    for (int i = 0; i < width * height; ++i)
        barycentricFromImgcoordCPU(Vp, Fp, findices_out, zbuffer.data(), width, height, num_vertices, num_faces, bary_out, i);
}
