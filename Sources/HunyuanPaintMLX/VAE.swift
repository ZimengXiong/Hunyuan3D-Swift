import Foundation
import MLX

/// AutoencoderKL (SD2.1 VAE) in MLX Swift — functional port of hy3dpaint_mlx/vae.py.
/// Mirrors the diffusers module tree so the dumped Python params map 1:1 by key.
public struct PaintVAE {
    let w: W
    let blockOut = [128, 256, 512, 512]
    let layersPerBlock = 2
    let groups = 32
    let eps: Float = 1e-6

    public init(_ weights: W) { self.w = weights }

    private func midBlock(_ x: MLXArray, _ p: String) -> MLXArray {
        var h = L2D.resnet(x, w, "\(p).resnets.0", groups: groups, eps: eps)
        h = L2D.vaeAttn(h, w, "\(p).attentions.0", groups: groups, eps: eps)
        h = L2D.resnet(h, w, "\(p).resnets.1", groups: groups, eps: eps)
        return h
    }

    /// z: [N,h,w,4] → image [N,H,W,3].
    public func decode(_ z: MLXArray) -> MLXArray {
        var x = L2D.conv(z, w, "post_quant_conv", padding: 0)   // 1x1
        x = L2D.conv(x, w, "decoder.conv_in")
        x = midBlock(x, "decoder.mid_block")
        let rev = Array(blockOut.reversed())                    // [512,512,256,128]
        for i in 0 ..< rev.count {
            let bp = "decoder.up_blocks.\(i)"
            for r in 0 ..< (layersPerBlock + 1) {
                x = L2D.resnet(x, w, "\(bp).resnets.\(r)", groups: groups, eps: eps)
            }
            if i != rev.count - 1 { x = L2D.upsample(x, w, "\(bp).upsamplers.0") }
        }
        x = L2D.silu(L2D.groupNorm(x, w, "decoder.conv_norm_out", groups: groups, eps: eps))
        return L2D.conv(x, w, "decoder.conv_out")
    }

    /// ximg: [N,H,W,3] in [-1,1] → mean latent [N,h,w,4].
    public func encodeMean(_ ximg: MLXArray) -> MLXArray {
        var x = L2D.conv(ximg, w, "encoder.conv_in")
        for i in 0 ..< blockOut.count {
            let bp = "encoder.down_blocks.\(i)"
            for r in 0 ..< layersPerBlock {
                x = L2D.resnet(x, w, "\(bp).resnets.\(r)", groups: groups, eps: eps)
            }
            if i != blockOut.count - 1 {
                x = L2D.downsample(x, w, "\(bp).downsamplers.0", padding: 0)
            }
        }
        x = midBlock(x, "encoder.mid_block")
        x = L2D.silu(L2D.groupNorm(x, w, "encoder.conv_norm_out", groups: groups, eps: eps))
        x = L2D.conv(x, w, "encoder.conv_out")                  // → 2*latent
        x = L2D.conv(x, w, "quant_conv", padding: 0)            // 1x1 moments
        return split(x, parts: 2, axis: -1)[0]                  // mean half
    }
}
