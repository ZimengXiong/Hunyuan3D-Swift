import Foundation
import MLX

/// Weight dictionary keyed by the Python MLX module path (e.g. "decoder.conv_in.weight").
public struct W {
    public let d: [String: MLXArray]
    public init(_ d: [String: MLXArray]) { self.d = d }
    @inline(__always) public func a(_ k: String) -> MLXArray { d[k]! }
    @inline(__always) public func has(_ k: String) -> Bool { d[k] != nil }
}

/// Shared 2D primitives for the SD2.1 VAE + UNet (NHWC), ported 1:1 from hy3dpaint_mlx/layers2d.py.
/// fp32-internal norms; conv weights are already MLX-layout [O,kH,kW,I] (transposed at dump time).
public enum L2D {
    public static func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

    /// fp32-internal GroupNorm for NHWC, groups = consecutive channels (matches torch GroupNorm).
    public static func groupNorm(_ x: MLXArray, _ w: W, _ p: String,
                                 groups: Int = 32, eps: Float = 1e-6) -> MLXArray {
        let N = x.dim(0), H = x.dim(1), Wd = x.dim(2), C = x.dim(3)
        let dt = x.dtype
        let xf = x.asType(.float32).reshaped([N, H, Wd, groups, C / groups])
        let mu = xf.mean(axes: [1, 2, 4], keepDims: true)
        let dd = xf - mu
        let v = (dd * dd).mean(axes: [1, 2, 4], keepDims: true)   // population variance
        var xn = (dd * rsqrt(v + eps)).reshaped([N, H, Wd, C])
        xn = xn * w.a("\(p).weight").asType(.float32) + w.a("\(p).bias").asType(.float32)
        return xn.asType(dt)
    }

    /// Conv2d, weight [O,kH,kW,I]; bias [O] broadcasts over NHW.
    public static func conv(_ x: MLXArray, _ w: W, _ p: String,
                            stride: Int = 1, padding: Int = 1) -> MLXArray {
        var y = conv2d(x, w.a("\(p).weight"), stride: IntOrPair(stride), padding: IntOrPair(padding))
        if let b = w.d["\(p).bias"] { y = y + b }
        return y
    }

    /// nn.Linear: weight [out,in]; y = x @ wᵀ + b.
    public static func linear(_ x: MLXArray, _ w: W, _ p: String) -> MLXArray {
        var y = matmul(x, w.a("\(p).weight").transposed(1, 0))
        if let b = w.d["\(p).bias"] { y = y + b }
        return y
    }

    /// diffusers ResnetBlock2D: norm1→silu→conv1 (+temb), norm2→silu→conv2, + (conv_shortcut?) skip.
    public static func resnet(_ x: MLXArray, _ w: W, _ p: String, temb: MLXArray? = nil,
                              groups: Int = 32, eps: Float = 1e-6) -> MLXArray {
        var h = conv(silu(groupNorm(x, w, "\(p).norm1", groups: groups, eps: eps)), w, "\(p).conv1")
        if let temb, w.has("\(p).time_emb_proj.weight") {
            let t = linear(silu(temb), w, "\(p).time_emb_proj")          // [N, out]
            h = h + t.reshaped([t.dim(0), 1, 1, t.dim(1)])
        }
        h = conv(silu(groupNorm(h, w, "\(p).norm2", groups: groups, eps: eps)), w, "\(p).conv2")
        let skip = w.has("\(p).conv_shortcut.weight")
            ? conv(x, w, "\(p).conv_shortcut", padding: 0) : x
        return h + skip
    }

    /// VAE downsample: stride-2 3x3 conv. padding 0 → manual asymmetric (0,1) pad first (VAE).
    public static func downsample(_ x: MLXArray, _ w: W, _ p: String, padding: Int) -> MLXArray {
        var xx = x
        if padding == 0 {
            xx = padded(x, widths: [IntOrPair((0, 0)), IntOrPair((0, 1)), IntOrPair((0, 1)), IntOrPair((0, 0))])
        }
        return conv(xx, w, "\(p).conv", stride: 2, padding: padding)
    }

    /// Upsample: NHWC nearest-neighbor x2 then 3x3 conv.
    public static func upsample(_ x: MLXArray, _ w: W, _ p: String) -> MLXArray {
        let N = x.dim(0), H = x.dim(1), Wd = x.dim(2), C = x.dim(3)
        let up = broadcast(x.reshaped([N, H, 1, Wd, 1, C]), to: [N, H, 2, Wd, 2, C])
            .reshaped([N, 2 * H, 2 * Wd, C])
        return conv(up, w, "\(p).conv")
    }

    /// VAE mid-block single-head spatial self-attention.
    public static func vaeAttn(_ x: MLXArray, _ w: W, _ p: String,
                               groups: Int = 32, eps: Float = 1e-6) -> MLXArray {
        let N = x.dim(0), H = x.dim(1), Wd = x.dim(2), C = x.dim(3)
        let h = groupNorm(x, w, "\(p).group_norm", groups: groups, eps: eps).reshaped([N, H * Wd, C])
        let q = linear(h, w, "\(p).to_q"), k = linear(h, w, "\(p).to_k"), v = linear(h, w, "\(p).to_v")
        let scale = 1.0 / Float(C).squareRoot()
        let attn = softmax(matmul(q, k.transposed(0, 2, 1)) * scale, axis: -1)
        let out = linear(matmul(attn, v), w, "\(p).to_out").reshaped([N, H, Wd, C])
        return x + out
    }
}
