import Foundation
import MLX

/// ShapeVAE decode path (post_kl -> transformer -> geo_decoder) in MLX Swift.
/// Functional port of hy3dmlx/models/shape_vae.py. Head-major qkv split (differs from the DiT's
/// K-outermost layout); geo_decoder.ln_post uses eps 1e-5, all other norms 1e-6.
public struct VAE {
    let w: [String: MLXArray]
    let heads: Int
    let width: Int
    let layers: Int
    public let scaleFactor: Float
    // 2^arange(8); include_pi=false. fourier out_dim = 3*(8*2+1) = 51.
    private let freqs = MLXArray([1, 2, 4, 8, 16, 32, 64, 128] as [Float])

    public init(weights: [String: MLXArray], heads: Int = 16, width: Int = 1024,
                layers: Int = 16, scaleFactor: Float = 1.0188137) {
        self.w = weights; self.heads = heads; self.width = width
        self.layers = layers; self.scaleFactor = scaleFactor
    }

    private func g(_ k: String) -> MLXArray { w[k]! }
    private func lin(_ x: MLXArray, _ p: String) -> MLXArray { Layers.linear(x, g("\(p).weight"), w["\(p).bias"]) }
    private func ln(_ x: MLXArray, _ p: String, _ eps: Float = 1e-6) -> MLXArray {
        Layers.layerNorm(x, g("\(p).weight"), g("\(p).bias"), eps: eps)
    }
    private func qkNorm(_ x: MLXArray, _ p: String) -> MLXArray {
        Layers.layerNorm(x, g("\(p).weight"), g("\(p).bias"), eps: 1e-6)
    }
    private func mlp(_ x: MLXArray, _ p: String) -> MLXArray {  // c_fc -> gelu(erf) -> c_proj
        lin(Layers.geluErf(lin(x, "\(p).c_fc")), "\(p).c_proj")
    }

    // self-attention: head-major split of c_qkv into q,k,v (each head_dim), qk_norm, sdpa
    private func selfAttn(_ x: MLXArray, _ p: String) -> MLXArray {
        let B = x.dim(0), N = x.dim(1)
        let r = lin(x, "\(p).c_qkv").reshaped([B, N, heads, -1])   // [B,N,H,3*hd]
        let parts = split(r, parts: 3, axis: -1)
        var q = qkNorm(parts[0], "\(p).attention.q_norm")
        var k = qkNorm(parts[1], "\(p).attention.k_norm")
        let v = parts[2].transposed(0, 2, 1, 3)
        q = q.transposed(0, 2, 1, 3); k = k.transposed(0, 2, 1, 3)
        let o = Layers.sdpa(q, k, v).transposed(0, 2, 1, 3).reshaped([B, N, -1])
        return lin(o, "\(p).c_proj")
    }

    // cross-attention: q from x, k/v from data (fused c_kv), qk_norm, sdpa
    private func crossAttn(_ x: MLXArray, _ data: MLXArray, _ p: String) -> MLXArray {
        let B = x.dim(0), nCtx = x.dim(1), nData = data.dim(1)
        let qf = lin(x, "\(p).c_q").reshaped([B, nCtx, heads, -1])
        let kv = lin(data, "\(p).c_kv").reshaped([B, nData, heads, -1])  // [B,nData,H,2*hd]
        let kvp = split(kv, parts: 2, axis: -1)
        var q = qkNorm(qf, "\(p).attention.q_norm").transposed(0, 2, 1, 3)
        var k = qkNorm(kvp[0], "\(p).attention.k_norm").transposed(0, 2, 1, 3)
        let v = kvp[1].transposed(0, 2, 1, 3)
        let o = Layers.sdpa(q, k, v).transposed(0, 2, 1, 3).reshaped([B, nCtx, -1])
        return lin(o, "\(p).c_proj")
    }

    /// latents [B,512,64] (already divided by scale_factor) -> kv [B,512,width]
    public func decode(_ latents: MLXArray) -> MLXArray {
        var x = lin(latents, "post_kl")
        for i in 0 ..< layers {
            let p = "transformer.resblocks.\(i)"
            x = x + selfAttn(ln(x, "\(p).ln_1"), "\(p).attn")
            x = x + mlp(ln(x, "\(p).ln_2"), "\(p).mlp")
        }
        return x
    }

    private func fourier(_ x: MLXArray) -> MLXArray {
        let embed = (x.expandedDimensions(axis: -1) * freqs)
            .reshaped([x.dim(0), x.dim(1), x.dim(2) * 8])
        return concatenated([x, sin(embed), cos(embed)], axis: -1)  // [B,P,51]
    }

    /// geo_decoder: query points [B,P,3] + kv latents -> SDF logits [B,P,1]
    public func geoDecoder(_ queries: MLXArray, _ kv: MLXArray) -> MLXArray {
        let pre = "geo_decoder"
        var qe = lin(fourier(queries), "\(pre).query_proj")
        let cp = "\(pre).cross_attn_decoder"
        qe = qe + crossAttn(ln(qe, "\(cp).ln_1"), ln(kv, "\(cp).ln_2"), "\(cp).attn")
        qe = qe + mlp(ln(qe, "\(cp).ln_3"), "\(cp).mlp")
        qe = ln(qe, "\(pre).ln_post", 1e-5)   // CRITICAL: torch-default eps
        return lin(qe, "\(pre).output_proj")
    }
}
