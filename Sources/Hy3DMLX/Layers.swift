import Foundation
import MLX
import MLXFast

/// Shared primitives, ported 1:1 from the Python hy3dmlx.layers (fp32-internal norms).
public enum Layers {
    /// nn.Linear: weight is [out, in]; y = x @ w.T + b.
    public static func linear(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray? = nil) -> MLXArray {
        var y = matmul(x, w.transposed(1, 0))
        if let b { y = y + b }
        return y
    }

    /// Linear against a quantized weight (packed `wq` + `scales`/`biases`), via quantizedMatmul.
    public static func qlinear(_ x: MLXArray, _ wq: MLXArray, scales: MLXArray, biases: MLXArray,
                               bias: MLXArray?, bits: Int, groupSize: Int) -> MLXArray {
        var y = quantizedMatmul(x, wq, scales: scales, biases: biases, transpose: true,
                                groupSize: groupSize, bits: bits)
        if let bias { y = y + bias }
        return y
    }

    /// fp32-internal LayerNorm; population variance (matches torch). affine via w/b.
    public static func layerNorm(_ x: MLXArray, _ w: MLXArray? = nil, _ b: MLXArray? = nil,
                                 eps: Float = 1e-6) -> MLXArray {
        let dt = x.dtype
        let xf = x.asType(.float32)
        let mu = xf.mean(axis: -1, keepDims: true)
        let d = xf - mu
        let v = (d * d).mean(axis: -1, keepDims: true)
        var y = d * rsqrt(v + eps)
        if let w { y = y * w.asType(.float32) + b!.asType(.float32) }
        return y.asType(dt)
    }

    /// torch RMSNorm: scale applied after cast-back (hunyuan3ddit.py).
    public static func rmsNorm(_ x: MLXArray, _ scale: MLXArray, eps: Float = 1e-6) -> MLXArray {
        let dt = x.dtype
        let xf = x.asType(.float32)
        let r = rsqrt((xf * xf).mean(axis: -1, keepDims: true) + eps)
        return (xf * r).asType(dt) * scale
    }

    /// torch nn.GELU(approximate="tanh") — DiT MLPs.
    public static func geluTanh(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let inner = Float(0.7978845608028654) * (xf + 0.044715 * xf * xf * xf)
        return (0.5 * xf * (1.0 + tanh(inner))).asType(x.dtype)
    }

    /// torch nn.GELU() exact (erf) — VAE MLPs.
    public static func geluErf(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        return (xf * 0.5 * (1.0 + erf(xf / Float(2.0).squareRoot()))).asType(x.dtype)
    }

    public static func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

    /// SDPA, q,k,v: [B,H,L,D] -> [B,H,L,D]. scale defaults to 1/sqrt(D).
    public static func sdpa(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, scale: Float? = nil) -> MLXArray {
        let s = scale ?? (1.0 / Float(q.dim(-1)).squareRoot())
        return MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: s, mask: .none)
    }

    /// Matches Python timestep_embedding(t, 256, max_period=time_factor=1000): t*1000, [cos,sin].
    public static func timestepEmbedding(_ t: MLXArray, dim: Int = 256, maxPeriod: Float = 1000) -> MLXArray {
        let half = dim / 2
        let tScaled = maxPeriod * t.asType(.float32)
        let freqs = exp(-Float(log(Double(maxPeriod))) * MLXArray(0 ..< half).asType(.float32) / Float(half))
        let args = tScaled.reshaped([t.dim(0), 1]) * freqs.reshaped([1, half])
        return concatenated([cos(args), sin(args)], axis: -1)
    }
}
