import Foundation
import MLX

/// 2.0 small (RGB) wrapper — functional port of unet2p5d.py UNet2p5DConditionModel.
/// Main UNet: 12-ch, multiview (MA) + reference (RA) attention, camera class-embedding,
/// learned_text_clip_gen. Dual UNet: 4-ch base, mode 'w' → reference features. No DINO/material/RoPE.
public struct Paint20Wrapper {
    let w: W            // main unet (+ learned_text_clip_gen/ref, class_embedding)
    let wd: W           // dual unet
    let maxNumRef = 5

    public init(main: W, dual: W) { self.w = main; self.wd = dual }

    /// Dual-stream reference pass (mode 'w') → condition_embed_dict. Constant across diffusion steps.
    public func prepare(refLat: MLXArray) -> [String: MLXArray] {
        let B = refLat.dim(0), nRef = refLat.dim(1), H = refLat.dim(2), Wd = refLat.dim(3)
        let rl = refLat.reshaped([B * nRef, H, Wd, 4])
        let ehsRef = broadcast(w.a("learned_text_clip_ref"), to: [B * nRef, 77, 1024])
        let xa = XAttn(mode: "w", numInBatch: nRef)
        let dual = PaintUNet(wd, pbr: false)
        _ = dual(rl, zeros([B * nRef]), ehsRef, xattn: xa)
        return xa.conditionEmbed
    }

    /// One velocity prediction. sample [B,N,h,w,4]; text [1,77,1024]; camGen = per-view camera indices.
    public func predict(_ sample: MLXArray, _ t: MLXArray, text: MLXArray, normalLat: MLXArray, positionLat: MLXArray,
                        camGen: [Int32], ced: [String: MLXArray]?, mvaScale: Float, refScale: Float) -> MLXArray {
        let B = sample.dim(0), N = sample.dim(1), H = sample.dim(2), Wd = sample.dim(3)
        let s = concatenated([sample, normalLat, positionLat], axis: -1).reshaped([B * N, H, Wd, 12])
        let ehs = broadcast(text.expandedDimensions(axis: 1), to: [B, N, 77, 1024]).reshaped([B * N, 77, 1024])
        let cam = MLXArray(camGen.map { $0 + Int32(maxNumRef) })                  // class labels [B*N]
        let unet = PaintUNet(w, useMa: true, useRa: true, hasClassEmbed: true, pbr: false)
        let xa = XAttn(mode: "r", numInBatch: N, conditionEmbed: ced ?? [:], refScale: refScale, mvaScale: mvaScale)
        return unet(s, t, ehs, classLabels: cam, xattn: xa).reshaped([B, N, H, Wd, 4])
    }
}
