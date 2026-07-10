import Foundation
import MLX

/// SD2.1 UNet2DConditionModel (NHWC) — functional port of unet.py. The base backbone the paint
/// 2.5D wrapper extends (use_ma/use_ra add reference + multiview attention; class_embedding = camera).
public struct PaintUNet {
    let w: W
    let boc = [320, 640, 1280, 1280]
    let heads = [5, 10, 20, 20]
    let layersPerBlock = 2
    let groups = 32
    let useMa: Bool
    let useRa: Bool
    let useMda: Bool
    let useDino: Bool
    let hasClassEmbed: Bool
    let pbr: Bool

    public init(_ weights: W, useMa: Bool = false, useRa: Bool = false,
                useMda: Bool = false, useDino: Bool = false,
                hasClassEmbed: Bool = false, pbr: Bool = false) {
        self.w = weights; self.useMa = useMa; self.useRa = useRa
        self.useMda = useMda; self.useDino = useDino
        self.hasClassEmbed = hasClassEmbed; self.pbr = pbr
    }

    public func callAsFunction(_ sample: MLXArray, _ timestep: MLXArray, _ ctx: MLXArray,
                               classLabels: MLXArray? = nil, xattn: XAttn? = nil) -> MLXArray {
        let tEmb = Attn.timestepEmbedding(timestep, dim: boc[0])
        var temb = L2D.linear(L2D.silu(L2D.linear(tEmb, w, "time_embedding.linear_1")), w, "time_embedding.linear_2")
        if hasClassEmbed, let cl = classLabels {
            temb = temb + take(w.a("class_embedding.weight"), cl, axis: 0)
        }

        var h = L2D.conv(sample, w, "conv_in")
        var res = [h]
        for i in 0 ..< boc.count {
            let bp = "down_blocks.\(i)", isCross = i < boc.count - 1
            for l in 0 ..< layersPerBlock {
                h = L2D.resnet(h, w, "\(bp).resnets.\(l)", temb: temb, groups: groups)
                if isCross {
                    h = Attn.transformer2d(h, w, "\(bp).attentions.\(l)", heads: heads[i], dimHead: boc[i] / heads[i],
                                           context: ctx, useMa: useMa, useRa: useRa,
                                           layerName: "down_\(i)_\(l)_0", xattn: xattn, pbr: pbr, useMda: useMda, useDino: useDino)
                }
                res.append(h)
            }
            if i != boc.count - 1 {
                h = L2D.downsample(h, w, "\(bp).downsamplers.0", padding: 1); res.append(h)
            }
        }

        h = L2D.resnet(h, w, "mid_block.resnets.0", temb: temb, groups: groups)
        h = Attn.transformer2d(h, w, "mid_block.attentions.0", heads: heads.last!, dimHead: boc.last! / heads.last!,
                               context: ctx, useMa: useMa, useRa: useRa, layerName: "mid_0_0", xattn: xattn, pbr: pbr, useMda: useMda, useDino: useDino)
        h = L2D.resnet(h, w, "mid_block.resnets.1", temb: temb, groups: groups)

        let revBoc = Array(boc.reversed()), revHeads = Array(heads.reversed())
        for i in 0 ..< revBoc.count {
            let bp = "up_blocks.\(i)", isCross = i != 0
            for l in 0 ..< (layersPerBlock + 1) {
                h = concatenated([h, res.removeLast()], axis: -1)
                h = L2D.resnet(h, w, "\(bp).resnets.\(l)", temb: temb, groups: groups)
                if isCross {
                    h = Attn.transformer2d(h, w, "\(bp).attentions.\(l)", heads: revHeads[i], dimHead: revBoc[i] / revHeads[i],
                                           context: ctx, useMa: useMa, useRa: useRa,
                                           layerName: "up_\(i)_\(l)_0", xattn: xattn, pbr: pbr, useMda: useMda, useDino: useDino)
                }
            }
            if i != revBoc.count - 1 { h = L2D.upsample(h, w, "\(bp).upsamplers.0") }
        }
        return L2D.conv(L2D.silu(L2D.groupNorm(h, w, "conv_norm_out", groups: groups, eps: 1e-5)), w, "conv_out")
    }
}
