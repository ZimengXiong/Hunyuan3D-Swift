import Foundation
import MLX

/// DINOv2-giant conditioner (ViT + SwiGLU) in MLX Swift. Functional port of hy3dmlx/models/dinov2.py.
/// Input: NHWC pixel values [1,518,518,3], already resized + ImageNet-normalized.
/// Weights are the `conditioner.main_image_encoder.model.` subtree (prefix stripped to `model.`...
/// here keys are stripped to the bare `embeddings.* / encoder.layer.* / layernorm.*`).
public struct DINOv2 {
    let w: [String: MLXArray]
    let heads: Int
    let layers: Int
    let patchW: MLXArray   // [out, kh, kw, in] for MLX conv2d
    let patch: Int
    let bits: Int
    let groupSize: Int

    public init(weights: [String: MLXArray], heads: Int = 24, layers: Int = 40, patch: Int = 14,
                bits: Int = 4, groupSize: Int = 64) {
        self.w = weights; self.heads = heads; self.layers = layers; self.patch = patch
        self.bits = bits; self.groupSize = groupSize
        // safetensors stores NCHW [out,in,kh,kw]; MLX conv2d wants [out,kh,kw,in]
        self.patchW = weights["embeddings.patch_embeddings.projection.weight"]!.transposed(0, 2, 3, 1)
    }

    private func g(_ k: String) -> MLXArray { w[k]! }
    private func lin(_ x: MLXArray, _ p: String) -> MLXArray {
        let wk = "\(p).weight"
        if let s = w["\(wk).scales"], let b = w["\(wk).biases"] {
            return Layers.qlinear(x, g(wk), scales: s, biases: b, bias: w["\(p).bias"],
                                  bits: bits, groupSize: groupSize)
        }
        return Layers.linear(x, g(wk), w["\(p).bias"])
    }
    private func ln(_ x: MLXArray, _ p: String) -> MLXArray {
        Layers.layerNorm(x, g("\(p).weight"), g("\(p).bias"), eps: 1e-6)
    }

    private func attn(_ x: MLXArray, _ p: String) -> MLXArray {
        let B = x.dim(0), N = x.dim(1)
        func h(_ t: MLXArray) -> MLXArray { t.reshaped([B, N, heads, -1]).transposed(0, 2, 1, 3) }
        let q = h(lin(x, "\(p).attention.query"))
        let k = h(lin(x, "\(p).attention.key"))
        let v = h(lin(x, "\(p).attention.value"))
        let o = Layers.sdpa(q, k, v).transposed(0, 2, 1, 3).reshaped([B, N, -1])
        return lin(o, "\(p).output.dense")
    }

    private func swiglu(_ x: MLXArray, _ p: String) -> MLXArray {
        let hpair = split(lin(x, "\(p).weights_in"), parts: 2, axis: -1)
        return lin(Layers.silu(hpair[0]) * hpair[1], "\(p).weights_out")
    }

    /// pixels NHWC [B,518,518,3] -> last_hidden_state [B,1370,1536]
    public func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        let B = pixels.dim(0)
        var p = conv2d(pixels, patchW, stride: IntOrPair(patch))             // [B,37,37,hidden]
        p = p + g("embeddings.patch_embeddings.projection.bias")
        let hidden = p.dim(-1)
        p = p.reshaped([B, p.dim(1) * p.dim(2), hidden])                     // [B,1369,hidden]
        let cls = broadcast(g("embeddings.cls_token"), to: [B, 1, hidden])
        var x = concatenated([cls, p], axis: 1) + g("embeddings.position_embeddings")
        for i in 0 ..< layers {
            let lp = "encoder.layer.\(i)"
            x = x + g("\(lp).layer_scale1.lambda1") * attn(ln(x, "\(lp).norm1"), "\(lp).attention")
            x = x + g("\(lp).layer_scale2.lambda1") * swiglu(ln(x, "\(lp).norm2"), "\(lp).mlp")
        }
        return ln(x, "layernorm")
    }

    public func unconditional(_ batch: Int, tokens: Int = 1370, hidden: Int = 1536) -> MLXArray {
        MLXArray.zeros([batch, tokens, hidden])
    }
}
