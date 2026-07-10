import Foundation
import MLX

/// RealESRGAN x4 (RRDBNet) in MLX Swift — functional port of hy3dpaint_mlx/realesrgan.py.
/// 23 RRDB blocks, dense connections, 0.2 residual scaling, x4 via two nearest-upsample+conv.
public struct RealESRGAN {
    let w: W
    let numBlock = 23
    public init(_ weights: W) { self.w = weights }

    private func lrelu(_ x: MLXArray) -> MLXArray { maximum(x, 0.2 * x) }

    private func up2x(_ x: MLXArray) -> MLXArray {            // NHWC nearest x2
        let N = x.dim(0), H = x.dim(1), Wd = x.dim(2), C = x.dim(3)
        return broadcast(x.reshaped([N, H, 1, Wd, 1, C]), to: [N, H, 2, Wd, 2, C])
            .reshaped([N, 2 * H, 2 * Wd, C])
    }

    private func rdb(_ x: MLXArray, _ p: String) -> MLXArray {
        let x1 = lrelu(L2D.conv(x, w, "\(p).conv1"))
        let x2 = lrelu(L2D.conv(concatenated([x, x1], axis: -1), w, "\(p).conv2"))
        let x3 = lrelu(L2D.conv(concatenated([x, x1, x2], axis: -1), w, "\(p).conv3"))
        let x4 = lrelu(L2D.conv(concatenated([x, x1, x2, x3], axis: -1), w, "\(p).conv4"))
        let x5 = L2D.conv(concatenated([x, x1, x2, x3, x4], axis: -1), w, "\(p).conv5")
        return x + x5 * 0.2
    }

    private func rrdb(_ x: MLXArray, _ p: String) -> MLXArray {
        x + 0.2 * rdb(rdb(rdb(x, "\(p).rdb1"), "\(p).rdb2"), "\(p).rdb3")
    }

    /// x: [N,H,W,3] in [0,1] → [N,4H,4W,3].
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let feat0 = L2D.conv(x, w, "conv_first")
        var b = feat0
        for i in 0 ..< numBlock { b = rrdb(b, "body.\(i)") }
        var feat = feat0 + L2D.conv(b, w, "conv_body")
        feat = lrelu(L2D.conv(up2x(feat), w, "conv_up1"))
        feat = lrelu(L2D.conv(up2x(feat), w, "conv_up2"))
        return L2D.conv(lrelu(L2D.conv(feat, w, "conv_hr")), w, "conv_last")
    }
}
