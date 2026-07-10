import Foundation
import MLX
import MLXFast

/// GPU custom_rasterizer (Apple Silicon Metal kernel) — Swift port of hy3dpaint_mlx/raster/gpu_raster.py.
/// Same Metal kernel source → bit-identical to the Python GPU rasterizer (and the cr_cpu oracle).
/// findices [H,W] int32 (0=bg, else 1-based), barycentric [H,W,3] float32 (perspective-correct).
public enum SwiftRaster {
    static let sentinel: Float = 2147483648.0   // 2^31, float-representable

    static let header = #"""
    inline float signed_area2(float ax, float ay, float bx, float by, float cx, float cy) {
        float p1 = (cx - ax) * (by - ay);
        float p2 = (bx - ax) * (cy - ay);
        return p1 - p2;
    }
    inline void screen_xyz(const device float* V, int v, int W, int H,
                           thread float& sx, thread float& sy, thread float& sz) {
        float x = V[v * 4 + 0];
        float y = V[v * 4 + 1];
        float z = V[v * 4 + 2];
        float w = V[v * 4 + 3];
        sx = (x / w * 0.5f + 0.5f) * (float)(W - 1) + 0.5f;
        sy = (0.5f + 0.5f * y / w) * (float)(H - 1) + 0.5f;
        sz = z / w * 0.49999f + 0.5f;
    }
    """#

    static let pass1a = #"""
        uint f = thread_position_in_grid.x;
        if (f >= (uint)num_faces[0]) return;
        int W = res[0]; int H = res[1];
        int i0 = F[f*3+0], i1 = F[f*3+1], i2 = F[f*3+2];
        float x0,y0,z0,x1,y1,z1,x2,y2,z2;
        screen_xyz(V,i0,W,H,x0,y0,z0); screen_xyz(V,i1,W,H,x1,y1,z1); screen_xyz(V,i2,W,H,x2,y2,z2);
        float area = signed_area2(x0,y0,x1,y1,x2,y2);
        if (area == 0.0f) return;
        float inv = 1.0f / area;
        float xminf=metal::min(x0,metal::min(x1,x2)), xmaxf=metal::max(x0,metal::max(x1,x2));
        float yminf=metal::min(y0,metal::min(y1,y2)), ymaxf=metal::max(y0,metal::max(y1,y2));
        int px0=(int)xminf, px1=(int)(xmaxf+1.0f), py0=(int)yminf, py1=(int)(ymaxf+1.0f);
        for (int px=px0; px<px1; ++px) { if (px<0||px>=W) continue; float vx=(float)px+0.5f;
            for (int py=py0; py<py1; ++py) { if (py<0||py>=H) continue; float vy=(float)py+0.5f;
                float beta=signed_area2(x0,y0,vx,vy,x2,y2)*inv;
                float gamma=signed_area2(x0,y0,x1,y1,vx,vy)*inv;
                float alpha=1.0f-beta-gamma;
                if (alpha<0.0f||alpha>1.0f) continue;
                if (beta<0.0f||beta>1.0f) continue;
                if (gamma<0.0f||gamma>1.0f) continue;
                float depth=alpha*z0+beta*z1+gamma*z2;
                int zq=(int)(depth*262144.0f);
                uint pix=(uint)(py*W+px);
                atomic_fetch_min_explicit(&zbuf_zq[pix],(uint)zq,memory_order_relaxed);
            } }
    """#

    static let pass1b = #"""
        uint f = thread_position_in_grid.x;
        if (f >= (uint)num_faces[0]) return;
        int W = res[0]; int H = res[1];
        int i0 = F[f*3+0], i1 = F[f*3+1], i2 = F[f*3+2];
        float x0,y0,z0,x1,y1,z1,x2,y2,z2;
        screen_xyz(V,i0,W,H,x0,y0,z0); screen_xyz(V,i1,W,H,x1,y1,z1); screen_xyz(V,i2,W,H,x2,y2,z2);
        float area = signed_area2(x0,y0,x1,y1,x2,y2);
        if (area == 0.0f) return;
        float inv = 1.0f / area;
        float xminf=metal::min(x0,metal::min(x1,x2)), xmaxf=metal::max(x0,metal::max(x1,x2));
        float yminf=metal::min(y0,metal::min(y1,y2)), ymaxf=metal::max(y0,metal::max(y1,y2));
        int px0=(int)xminf, px1=(int)(xmaxf+1.0f), py0=(int)yminf, py1=(int)(ymaxf+1.0f);
        for (int px=px0; px<px1; ++px) { if (px<0||px>=W) continue; float vx=(float)px+0.5f;
            for (int py=py0; py<py1; ++py) { if (py<0||py>=H) continue; float vy=(float)py+0.5f;
                float beta=signed_area2(x0,y0,vx,vy,x2,y2)*inv;
                float gamma=signed_area2(x0,y0,x1,y1,vx,vy)*inv;
                float alpha=1.0f-beta-gamma;
                if (alpha<0.0f||alpha>1.0f) continue;
                if (beta<0.0f||beta>1.0f) continue;
                if (gamma<0.0f||gamma>1.0f) continue;
                float depth=alpha*z0+beta*z1+gamma*z2;
                int zq=(int)(depth*262144.0f);
                uint pix=(uint)(py*W+px);
                if ((uint)zq==zbuf_zq[pix]) atomic_fetch_min_explicit(&zbuf_fid[pix],f+1u,memory_order_relaxed);
            } }
    """#

    static let pass2 = #"""
        uint pix = thread_position_in_grid.x;
        if (pix >= (uint)(res[0]*res[1])) return;
        int W = res[0]; int H = res[1];
        uint fid = zbuf_fid[pix];
        if (fid >= (uint)SENTINEL_C || fid == 0u) {
            findices[pix]=0; barycentric[pix*3+0]=0.0f; barycentric[pix*3+1]=0.0f; barycentric[pix*3+2]=0.0f; return;
        }
        findices[pix]=(int)fid;
        int f=(int)fid-1;
        int i0=F[f*3+0], i1=F[f*3+1], i2=F[f*3+2];
        float x0,y0,z0,x1,y1,z1,x2,y2,z2;
        screen_xyz(V,i0,W,H,x0,y0,z0); screen_xyz(V,i1,W,H,x1,y1,z1); screen_xyz(V,i2,W,H,x2,y2,z2);
        float w0=V[i0*4+3], w1=V[i1*4+3], w2=V[i2*4+3];
        float vx=(float)(pix%(uint)W)+0.5f, vy=(float)(pix/(uint)W)+0.5f;
        float area=signed_area2(x0,y0,x1,y1,x2,y2);
        float inv=1.0f/area;
        float beta=signed_area2(x0,y0,vx,vy,x2,y2)*inv;
        float gamma=signed_area2(x0,y0,x1,y1,vx,vy)*inv;
        float alpha=1.0f-beta-gamma;
        float a=alpha/w0, b=beta/w1, c=gamma/w2;
        float s=1.0f/(a+b+c);
        barycentric[pix*3+0]=a*s; barycentric[pix*3+1]=b*s; barycentric[pix*3+2]=c*s;
    """#

    static let k1a = MLXFast.metalKernel(name: "cr_pass1a_zq", inputNames: ["V", "F", "num_faces", "res"],
                                         outputNames: ["zbuf_zq"], source: pass1a, header: header, atomicOutputs: true)
    static let k1b = MLXFast.metalKernel(name: "cr_pass1b_fid", inputNames: ["V", "F", "num_faces", "res", "zbuf_zq"],
                                         outputNames: ["zbuf_fid"], source: pass1b, header: header, atomicOutputs: true)
    static let k2 = MLXFast.metalKernel(name: "cr_pass2_bary", inputNames: ["V", "F", "res", "zbuf_fid"],
                                        outputNames: ["findices", "barycentric"], source: pass2,
                                        header: header + "\n#define SENTINEL_C 2147483648u\n")

    /// pos: [V,4] or [1,V,4] clip-space homogeneous; tri: [F,3] int. → (findices [H,W] int32, bary [H,W,3]).
    public static func rasterize(_ pos: MLXArray, _ tri: MLXArray, _ resolution: Int) -> (MLXArray, MLXArray) {
        let p = pos.ndim == 3 ? pos[0] : pos
        let H = resolution, W = resolution, npix = H * W
        let V = p.asType(.float32), F = tri.asType(.int32)
        let numFaces = F.dim(0)
        if numFaces == 0 { return (MLX.zeros([H, W]).asType(.int32), MLX.zeros([H, W, 3])) }
        let res = MLXArray([Int32(W), Int32(H)]), nf = MLXArray([Int32(numFaces)])
        let tg = min(256, numFaces)
        let zq = k1a([V, F, nf, res], grid: (numFaces, 1, 1), threadGroup: (tg, 1, 1),
                     outputShapes: [[npix]], outputDTypes: [.uint32], initValue: sentinel)[0]
        let fid = k1b([V, F, nf, res, zq], grid: (numFaces, 1, 1), threadGroup: (tg, 1, 1),
                      outputShapes: [[npix]], outputDTypes: [.uint32], initValue: sentinel)[0]
        let out = k2([V, F, res, fid], grid: (npix, 1, 1), threadGroup: (min(256, npix), 1, 1),
                     outputShapes: [[npix], [npix * 3]], outputDTypes: [.int32, .float32])
        return (out[0].reshaped([H, W]), out[1].reshaped([H, W, 3]))
    }

    /// Vertex-attribute interpolation (matches cr_raster.interpolate). col [V,C] → [H,W,C], 0 at bg.
    public static func interpolate(_ col: MLXArray, _ findices: MLXArray, _ bary: MLXArray, _ tri: MLXArray) -> MLXArray {
        let H = findices.dim(0), W = findices.dim(1), C = col.dim(1)
        let fi = findices.reshaped([H * W])
        let bg = (fi .== 0)
        let f = maximum(fi - 1, 0)
        let vidx = take(tri.asType(.int32), f, axis: 0)                      // [HW,3]
        let vcol = take(col, vidx.reshaped([H * W * 3]), axis: 0).reshaped([H * W, 3, C])
        let out = (bary.reshaped([H * W, 3, 1]) * vcol).sum(axis: 1).reshaped([H, W, C])
        return MLX.where(bg.reshaped([H, W, 1]), MLX.zeros(out.shape), out)
    }
}
