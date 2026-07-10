import Foundation
import MLX
import MLXFast

/// 2.1 PBR attention: 3D PoseRoPE, material self-attn (MDA), reference attn (RA, per-material V),
/// multiview attn (MA, RoPE), DINO cross-attn. Functional port of attention_pbr.py PBRTransformerBlock.
/// Base parts live under "<p>.transformer." (key match); material weights under ".processor.*_mr".
public enum PBR {
    /// SDPA on flat [B,L,inner]; v may carry a wider head dim (per-material concat) → [B,Lq,heads*hdv].
    static func sdpaFlat(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, heads: Int, scale: Float) -> MLXArray {
        let B = q.dim(0), Lq = q.dim(1), Lk = k.dim(1)
        let hd = q.dim(2) / heads, hdv = v.dim(2) / heads
        let qh = q.reshaped([B, Lq, heads, hd]).transposed(0, 2, 1, 3)
        let kh = k.reshaped([B, Lk, heads, hd]).transposed(0, 2, 1, 3)
        let vh = v.reshaped([B, Lk, heads, hdv]).transposed(0, 2, 1, 3)
        let o = MLXFast.scaledDotProductAttention(queries: qh, keys: kh, values: vh, scale: scale, mask: .none)
        return o.transposed(0, 2, 1, 3).reshaped([B, Lq, heads * hdv])
    }

    /// apply_rotary_emb: x [B,heads,L,hd]; cos/sin [B,L,hd].
    static func applyRotary(_ x: MLXArray, _ cos: MLXArray, _ sin: MLXArray) -> MLXArray {
        let c = cos.expandedDimensions(axis: 1), s = sin.expandedDimensions(axis: 1)
        let shp = x.shape
        let x2 = x.reshaped([shp[0], shp[1], shp[2], shp[3] / 2, 2])
        let parts = split(x2, parts: 2, axis: -1)               // real, imag (each [...,1])
        let rot = concatenated([-parts[1], parts[0]], axis: -1).reshaped(shp)
        return x * c + rot * s
    }

    /// MDA: per-material self-attn. normH [(b nPbr n), l, c]; albedo→base weights, mr→processor weights.
    static func selfAttnMaterial(_ normH: MLXArray, _ w: W, _ t: String,
                                 heads: Int, scale: Float, n: Int, nPbr: Int) -> MLXArray {
        let Bt = normH.dim(0), l = normH.dim(1), c = normH.dim(2), b = Bt / (nPbr * n)
        let xs = split(normH.reshaped([b, nPbr, n, l, c]), parts: nPbr, axis: 1)
        var outs = [MLXArray]()
        for mi in 0 ..< nPbr {
            let hs = xs[mi].reshaped([b * n, l, c])
            let base = mi == 0 ? "\(t).attn1" : "\(t).attn1.processor"
            let suf = mi == 0 ? "" : "_mr"
            let q = L2D.linear(hs, w, "\(base).to_q\(suf)")
            let k = L2D.linear(hs, w, "\(base).to_k\(suf)")
            let v = L2D.linear(hs, w, "\(base).to_v\(suf)")
            let outKey = mi == 0 ? "\(t).attn1.to_out.0" : "\(t).attn1.processor.to_out_mr.0"
            let o = L2D.linear(sdpaFlat(q, k, v, heads: heads, scale: scale), w, outKey)
            outs.append(o.reshaped([b, 1, n, l, c]))
        }
        return concatenated(outs, axis: 1).reshaped([Bt, l, c])
    }

    /// RA: query albedo ref_norm [b,(n l),c]; k/v from cond; per-material V → stack [b,nPbr,(n l),c].
    static func refAttn(_ refNorm: MLXArray, _ cond: MLXArray, _ w: W, _ p: String,
                        heads: Int, dimHead: Int, scale: Float) -> MLXArray {
        let b = refNorm.dim(0)
        let q = L2D.linear(refNorm, w, "\(p).attn_refview.to_q")
        let k = L2D.linear(cond, w, "\(p).attn_refview.to_k")
        let vAlb = L2D.linear(cond, w, "\(p).attn_refview.to_v")
        let vMr = L2D.linear(cond, w, "\(p).attn_refview.processor.to_v_mr")
        let o = sdpaFlat(q, k, concatenated([vAlb, vMr], axis: -1), heads: heads, scale: scale)
        let Lq = o.dim(1)
        let halves = split(o.reshaped([b, Lq, heads, 2 * dimHead]), parts: 2, axis: -1)
        let outAlb = L2D.linear(halves[0].reshaped([b, Lq, heads * dimHead]), w, "\(p).attn_refview.to_out.0")
        let outMr = L2D.linear(halves[1].reshaped([b, Lq, heads * dimHead]), w, "\(p).attn_refview.processor.to_out_mr.0")
        return stacked([outAlb, outMr], axis: 1)
    }

    /// MA: multiview self-attn with 3D RoPE on q,k. mv [(b nPbr),(n l),c].
    static func mvAttn(_ mv: MLXArray, _ w: W, _ p: String, heads: Int, dimHead: Int, scale: Float,
                       cos: MLXArray?, sin: MLXArray?) -> MLXArray {
        let B = mv.dim(0), L = mv.dim(1)
        var q = L2D.linear(mv, w, "\(p).attn_multiview.to_q").reshaped([B, L, heads, dimHead]).transposed(0, 2, 1, 3)
        var k = L2D.linear(mv, w, "\(p).attn_multiview.to_k").reshaped([B, L, heads, dimHead]).transposed(0, 2, 1, 3)
        let v = L2D.linear(mv, w, "\(p).attn_multiview.to_v").reshaped([B, L, heads, dimHead]).transposed(0, 2, 1, 3)
        if let cos, let sin { q = applyRotary(q, cos, sin); k = applyRotary(k, cos, sin) }
        let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        return L2D.linear(o.transposed(0, 2, 1, 3).reshaped([B, L, heads * dimHead]), w, "\(p).attn_multiview.to_out.0")
    }

    /// PBR block forward. Flags select features: main unet = all on; dual reference unet = all off
    /// (but still keyed under "<p>.transformer." — that's why the dual uses this, not basicBlock).
    static func block(_ hidden: MLXArray, _ w: W, _ p: String, heads: Int, dimHead: Int,
                      context: MLXArray?, layerName: String, xattn: XAttn,
                      useMa: Bool, useRa: Bool, useMda: Bool, useDino: Bool) -> MLXArray {
        let t = "\(p).transformer"
        let scale = Float(pow(Double(dimHead), -0.5))
        let n = xattn.numInBatch, nPbr = xattn.nPbr
        let normH = Attn.layerNorm(hidden, w, "\(t).norm1")
        var h = useMda ? (selfAttnMaterial(normH, w, t, heads: heads, scale: scale, n: n, nPbr: nPbr) + hidden)
                       : (Attn.attention(normH, w, "\(t).attn1", heads: heads, dimHead: dimHead) + hidden)
        let Bt = normH.dim(0), l = normH.dim(1), c = normH.dim(2), b = Bt / (nPbr * n)
        if xattn.mode.contains("w") { xattn.conditionEmbed[layerName] = normH.reshaped([b, n * l, c]) }
        if xattn.mode.contains("r"), useRa, let cond = xattn.conditionEmbed[layerName] {
            let refNorm0 = split(normH.reshaped([b, nPbr, n * l, c]), parts: nPbr, axis: 1)[0].reshaped([b, n * l, c])
            let ra = refAttn(refNorm0, cond, w, p, heads: heads, dimHead: dimHead, scale: scale)
            h = xattn.refScale * ra.reshaped([b, nPbr, n, l, c]).reshaped([Bt, l, c]) + h
        }
        if n > 1, useMa {
            let mv = normH.reshaped([b, nPbr, n, l, c]).reshaped([b * nPbr, n * l, c])
            let (cos, sin) = xattn.ropeByTokens[mv.dim(1)] ?? (nil, nil)
            let o = mvAttn(mv, w, p, heads: heads, dimHead: dimHead, scale: scale, cos: cos, sin: sin)
            h = xattn.mvaScale * o.reshaped([b, nPbr, n, l, c]).reshaped([Bt, l, c]) + h
        }
        let normH2 = Attn.layerNorm(h, w, "\(t).norm2")
        h = Attn.attention(normH2, w, "\(t).attn2", heads: heads, dimHead: dimHead, context: context) + h
        if useDino, let dino = xattn.dino {
            let d0 = dino.dim(0), d1 = dino.dim(1), d2 = dino.dim(2)
            let dinoR = broadcast(dino.reshaped([d0, 1, d1, d2]), to: [d0, nPbr * n, d1, d2]).reshaped([d0 * nPbr * n, d1, d2])
            let q = L2D.linear(normH2, w, "\(p).attn_dino.to_q")
            let k = L2D.linear(dinoR, w, "\(p).attn_dino.to_k")
            let v = L2D.linear(dinoR, w, "\(p).attn_dino.to_v")
            h = L2D.linear(sdpaFlat(q, k, v, heads: heads, scale: scale), w, "\(p).attn_dino.to_out.0") + h
        }
        return Attn.feedForward(Attn.layerNorm(h, w, "\(t).norm3"), w, "\(t).ff") + h
    }
}
