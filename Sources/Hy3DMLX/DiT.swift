import Foundation
import MLX

/// Hunyuan3DDiT (FLUX double/single stream, no RoPE) in MLX Swift.
/// Functional port of hy3dmlx/models/dit.py — indexes a weights dict by torch key
/// (the `model.` prefix already stripped). Proof that the MLX Swift path matches Python.
public struct DiT {
    let w: [String: MLXArray]
    let heads: Int
    let hidden: Int
    let depth: Int
    let depthSingle: Int
    let timeFactor: Float
    let guidanceEmbed: Bool
    let bits: Int
    let groupSize: Int

    public init(weights: [String: MLXArray], heads: Int = 16, hidden: Int = 1024,
                depth: Int = 8, depthSingle: Int = 16, timeFactor: Float = 1000,
                guidanceEmbed: Bool = false, bits: Int = 4, groupSize: Int = 64) {
        self.w = weights; self.heads = heads; self.hidden = hidden
        self.depth = depth; self.depthSingle = depthSingle
        self.timeFactor = timeFactor; self.guidanceEmbed = guidanceEmbed
        self.bits = bits; self.groupSize = groupSize
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

    private func splitQKV(_ qkv: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let B = qkv.dim(0), Lq = qkv.dim(1)
        let r = qkv.reshaped([B, Lq, 3, heads, -1]).transposed(2, 0, 3, 1, 4)  // [3,B,H,L,D]
        return (r[0], r[1], r[2])
    }
    private func mergeHeads(_ x: MLXArray) -> MLXArray {
        x.transposed(0, 2, 1, 3).reshaped([x.dim(0), x.dim(2), x.dim(1) * x.dim(3)])
    }
    private func qknorm(_ q: MLXArray, _ k: MLXArray, _ p: String) -> (MLXArray, MLXArray) {
        (Layers.rmsNorm(q, g("\(p).query_norm.scale")), Layers.rmsNorm(k, g("\(p).key_norm.scale")))
    }
    private func modulation(_ vec: MLXArray, _ p: String, _ mult: Int) -> [MLXArray] {
        let m = lin(Layers.silu(vec), "\(p).lin")
        return split(m.expandedDimensions(axis: 1), parts: mult, axis: -1)  // mult x [B,1,hidden]
    }
    private func ln(_ x: MLXArray) -> MLXArray { Layers.layerNorm(x, eps: 1e-6) }  // no-affine
    private func mlp(_ x: MLXArray, _ p: String) -> MLXArray {  // Linear -> gelu(tanh) -> Linear
        lin(Layers.geluTanh(lin(x, "\(p).0")), "\(p).2")
    }

    private func doubleBlock(_ img0: MLXArray, _ cond0: MLXArray, _ vec: MLXArray, _ i: Int)
        -> (MLXArray, MLXArray) {
        let p = "double_blocks.\(i)"
        let im = modulation(vec, "\(p).img_mod", 6)  // shift,scale,gate, shift2,scale2,gate2
        let tm = modulation(vec, "\(p).txt_mod", 6)
        var img = img0, cond = cond0

        let imgMod = (1 + im[1]) * ln(img) + im[0]
        var (iq, ik, iv) = splitQKV(lin(imgMod, "\(p).img_attn.qkv"))
        (iq, ik) = qknorm(iq, ik, "\(p).img_attn.norm")
        let txtMod = (1 + tm[1]) * ln(cond) + tm[0]
        var (tq, tk, tv) = splitQKV(lin(txtMod, "\(p).txt_attn.qkv"))
        (tq, tk) = qknorm(tq, tk, "\(p).txt_attn.norm")

        let q = concatenated([tq, iq], axis: 2)
        let k = concatenated([tk, ik], axis: 2)
        let v = concatenated([tv, iv], axis: 2)
        let attn = mergeHeads(Layers.sdpa(q, k, v))
        let parts = split(attn, indices: [cond.dim(1)], axis: 1)
        let txtAttn = parts[0], imgAttn = parts[1]

        img = img + im[2] * lin(imgAttn, "\(p).img_attn.proj")
        img = img + im[5] * mlp((1 + im[4]) * ln(img) + im[3], "\(p).img_mlp")
        cond = cond + tm[2] * lin(txtAttn, "\(p).txt_attn.proj")
        cond = cond + tm[5] * mlp((1 + tm[4]) * ln(cond) + tm[3], "\(p).txt_mlp")
        return (img, cond)
    }

    private func singleBlock(_ x: MLXArray, _ vec: MLXArray, _ i: Int) -> MLXArray {
        let p = "single_blocks.\(i)"
        let m = modulation(vec, "\(p).modulation", 3)  // shift,scale,gate
        let xMod = (1 + m[1]) * ln(x) + m[0]
        let out = lin(xMod, "\(p).linear1")
        let parts = split(out, indices: [3 * hidden], axis: -1)
        var (q, k, v) = splitQKV(parts[0])
        (q, k) = qknorm(q, k, "\(p).norm")
        let attn = mergeHeads(Layers.sdpa(q, k, v))
        let o = lin(concatenated([attn, Layers.geluTanh(parts[1])], axis: 2), "\(p).linear2")
        return x + m[2] * o
    }

    private func finalLayer(_ x: MLXArray, _ vec: MLXArray) -> MLXArray {
        let mod = lin(Layers.silu(vec), "final_layer.adaLN_modulation.1")
        let sc = split(mod, parts: 2, axis: 1)
        let shift = sc[0].expandedDimensions(axis: 1), scale = sc[1].expandedDimensions(axis: 1)
        return lin((1 + scale) * ln(x) + shift, "final_layer.linear")
    }

    public func callAsFunction(_ x: MLXArray, _ t: MLXArray, _ cond: MLXArray,
                               guidance: MLXArray? = nil) -> MLXArray {
        var latent = lin(x, "latent_in")
        let te = Layers.timestepEmbedding(t, dim: 256, maxPeriod: timeFactor)
        var vec = lin(Layers.silu(lin(te, "time_in.in_layer")), "time_in.out_layer")
        if guidanceEmbed, let guidance {
            let ge = Layers.timestepEmbedding(guidance, dim: 256, maxPeriod: timeFactor)
            vec = vec + lin(Layers.silu(lin(ge, "guidance_in.in_layer")), "guidance_in.out_layer")
        }
        var cond = lin(cond, "cond_in")
        for i in 0 ..< depth { (latent, cond) = doubleBlock(latent, cond, vec, i) }
        latent = concatenated([cond, latent], axis: 1)
        for i in 0 ..< depthSingle { latent = singleBlock(latent, vec, i) }
        latent = split(latent, indices: [cond.dim(1)], axis: 1)[1]
        return finalLayer(latent, vec)
    }
}
