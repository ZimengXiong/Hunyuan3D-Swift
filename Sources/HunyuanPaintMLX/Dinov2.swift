import Foundation
import MLX
import MLXFast

/// DINOv2-giant (ViT + SwiGLU FFN) — functional port of dinov2.py. hidden 1536, 24 heads, 40 layers,
/// patch 14, 518px → 1370 tokens. Input is NHWC, ImageNet-normalized.
public struct Dinov2 {
    let w: W
    let layers = 40
    let heads = 24
    public init(_ weights: W) { self.w = weights }

    private func ln(_ x: MLXArray, _ p: String) -> MLXArray {
        MLXFast.layerNorm(x, weight: w.a("\(p).weight"), bias: w.a("\(p).bias"), eps: 1e-6)
    }
    private func headsT(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), N = x.dim(1), HD = x.dim(2)
        return x.reshaped([B, N, heads, HD / heads]).transposed(0, 2, 1, 3)
    }

    private func layer(_ x0: MLXArray, _ p: String) -> MLXArray {
        var x = x0
        let n1 = ln(x, "\(p).norm1")
        let q = headsT(L2D.linear(n1, w, "\(p).attention.attention.query"))
        let k = headsT(L2D.linear(n1, w, "\(p).attention.attention.key"))
        let v = headsT(L2D.linear(n1, w, "\(p).attention.attention.value"))
        let scale = Float(pow(Double(q.dim(3)), -0.5))
        let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        let attn = o.transposed(0, 2, 1, 3).reshaped([x.dim(0), x.dim(1), heads * q.dim(3)])
        x = x + L2D.linear(attn, w, "\(p).attention.output.dense") * w.a("\(p).layer_scale1.lambda1")
        let n2 = ln(x, "\(p).norm2")
        let halves = split(L2D.linear(n2, w, "\(p).mlp.weights_in"), parts: 2, axis: -1)
        let mlp = L2D.linear(L2D.silu(halves[0]) * halves[1], w, "\(p).mlp.weights_out")
        return x + mlp * w.a("\(p).layer_scale2.lambda1")
    }

    /// pixel: [B,518,518,3] → last_hidden_state [B,1370,1536].
    public func callAsFunction(_ pixel: MLXArray) -> MLXArray {
        let p = L2D.conv(pixel, w, "embeddings.patch_embeddings.projection", stride: 14, padding: 0)
        let B = p.dim(0), C = p.dim(3)
        let patches = p.reshaped([B, p.dim(1) * p.dim(2), C])
        let cls = broadcast(w.a("embeddings.cls_token"), to: [B, 1, C])
        var x = concatenated([cls, patches], axis: 1) + w.a("embeddings.position_embeddings")
        for i in 0 ..< layers { x = layer(x, "encoder.layer.\(i)") }
        return ln(x, "layernorm")
    }
}
