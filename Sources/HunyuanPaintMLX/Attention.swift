import Foundation
import MLX
import MLXFast
import MLXNN

/// Cross-attention kwargs for the 2.5D paint blocks (reference + multiview attention).
/// `conditionEmbed` is mutated in "w" (write) mode and read in "r" mode — a reference class.
public final class XAttn {
    public var mode: String
    public var numInBatch: Int
    public var conditionEmbed: [String: MLXArray]
    public var refScale: Float
    public var mvaScale: Float
    // PBR (2.1) extras
    public var nPbr: Int
    public var dino: MLXArray?
    public var ropeByTokens: [Int: (MLXArray, MLXArray)]
    public init(mode: String, numInBatch: Int, conditionEmbed: [String: MLXArray] = [:],
                refScale: Float = 1, mvaScale: Float = 1, nPbr: Int = 1,
                dino: MLXArray? = nil, ropeByTokens: [Int: (MLXArray, MLXArray)] = [:]) {
        self.mode = mode; self.numInBatch = numInBatch; self.conditionEmbed = conditionEmbed
        self.refScale = refScale; self.mvaScale = mvaScale
        self.nPbr = nPbr; self.dino = dino; self.ropeByTokens = ropeByTokens
    }
}

/// Attention / Transformer2D / timestep primitives for the SD2.1 UNet — port of attention.py.
public enum Attn {
    /// diffusers get_timestep_embedding (max_period=10000, flip_sin_to_cos): t [N] → [N,dim].
    public static func timestepEmbedding(_ t: MLXArray, dim: Int, maxPeriod: Float = 10000) -> MLXArray {
        let half = dim / 2
        let exponent = -Foundation.log(maxPeriod) * MLXArray(0 ..< half).asType(.float32) / Float(half)
        let emb = exp(exponent)
        let e = t.asType(.float32).reshaped([t.dim(0), 1]) * emb.reshaped([1, half])
        return concatenated([cos(e), sin(e)], axis: -1)
    }

    public static func layerNorm(_ x: MLXArray, _ w: W, _ p: String, eps: Float = 1e-5) -> MLXArray {
        MLXFast.layerNorm(x, weight: w.a("\(p).weight"), bias: w.a("\(p).bias"), eps: eps)
    }

    /// diffusers Attention; self if context nil, else cross. to_q/k/v bias-less, to_out.0 has bias.
    public static func attention(_ x: MLXArray, _ w: W, _ p: String, heads: Int, dimHead: Int,
                                 context: MLXArray? = nil) -> MLXArray {
        let ctx = context ?? x
        let B = x.dim(0), N = x.dim(1), M = ctx.dim(1)
        let scale = Float(pow(Double(dimHead), -0.5))
        let q = L2D.linear(x, w, "\(p).to_q").reshaped([B, N, heads, dimHead]).transposed(0, 2, 1, 3)
        let k = L2D.linear(ctx, w, "\(p).to_k").reshaped([B, M, heads, dimHead]).transposed(0, 2, 1, 3)
        let v = L2D.linear(ctx, w, "\(p).to_v").reshaped([B, M, heads, dimHead]).transposed(0, 2, 1, 3)
        let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        return L2D.linear(o.transposed(0, 2, 1, 3).reshaped([B, N, heads * dimHead]), w, "\(p).to_out.0")
    }

    public static func feedForward(_ x: MLXArray, _ w: W, _ p: String) -> MLXArray {
        let proj = L2D.linear(x, w, "\(p).net.0.proj")          // GEGLU
        let parts = split(proj, parts: 2, axis: -1)
        return L2D.linear(parts[0] * gelu(parts[1]), w, "\(p).net.2")
    }

    /// SD2.1 BasicTransformerBlock + optional 2.5D reference (RA) / multiview (MA) attention.
    public static func basicBlock(_ x: MLXArray, _ w: W, _ p: String, heads: Int, dimHead: Int,
                                  context: MLXArray?, useMa: Bool, useRa: Bool,
                                  layerName: String?, xattn: XAttn?) -> MLXArray {
        let normH = layerNorm(x, w, "\(p).norm1")
        var h = attention(normH, w, "\(p).attn1", heads: heads, dimHead: dimHead) + x
        if let xa = xattn, let ln = layerName {
            let n = xa.numInBatch
            let B = x.dim(0) / n, L = x.dim(1), C = x.dim(2)
            if xa.mode.contains("w") {
                xa.conditionEmbed[ln] = normH.reshaped([B, n * L, C])
            }
            if xa.mode.contains("r"), useRa, let ce0 = xa.conditionEmbed[ln] {
                let Lr = ce0.dim(1)
                let ce = broadcast(ce0.reshaped([B, 1, Lr, C]), to: [B, n, Lr, C]).reshaped([B * n, Lr, C])
                h = xa.refScale * attention(normH, w, "\(p).attn_refview", heads: heads, dimHead: dimHead, context: ce) + h
            }
            if n > 1, useMa {
                let mv = normH.reshaped([B, n * L, C])
                let out = attention(mv, w, "\(p).attn_multiview", heads: heads, dimHead: dimHead, context: mv)
                    .reshaped([B * n, L, C])
                h = xa.mvaScale * out + h
            }
        }
        h = attention(layerNorm(h, w, "\(p).norm2"), w, "\(p).attn2", heads: heads, dimHead: dimHead, context: context) + h
        h = feedForward(layerNorm(h, w, "\(p).norm3"), w, "\(p).ff") + h
        return h
    }

    /// SD2.1 Transformer2DModel (use_linear_projection). NHWC in/out.
    public static func transformer2d(_ x: MLXArray, _ w: W, _ p: String, heads: Int, dimHead: Int,
                                     depth: Int = 1, context: MLXArray?, useMa: Bool, useRa: Bool,
                                     layerName: String?, xattn: XAttn?, pbr: Bool = false,
                                     useMda: Bool = false, useDino: Bool = false) -> MLXArray {
        let N = x.dim(0), H = x.dim(1), Wd = x.dim(2), C = x.dim(3)
        var h = L2D.groupNorm(x, w, "\(p).norm", groups: 32, eps: 1e-6).reshaped([N, H * Wd, C])
        h = L2D.linear(h, w, "\(p).proj_in")
        for d in 0 ..< depth {
            let bp = "\(p).transformer_blocks.\(d)"
            if pbr {
                h = PBR.block(h, w, bp, heads: heads, dimHead: dimHead, context: context,
                              layerName: layerName ?? "", xattn: xattn!,
                              useMa: useMa, useRa: useRa, useMda: useMda, useDino: useDino)
            } else {
                h = basicBlock(h, w, bp, heads: heads, dimHead: dimHead, context: context,
                               useMa: useMa, useRa: useRa, layerName: layerName, xattn: xattn)
            }
        }
        return L2D.linear(h, w, "\(p).proj_out").reshaped([N, H, Wd, C]) + x
    }
}
