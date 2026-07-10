import Foundation
import MLX
import MLXRandom

/// On-device sampler schedule + noise, so no Python-dumped fixtures are needed.
public enum Sampler {
    /// FlowMatchEulerDiscrete sigmas: linspace(0,1,N) with a rational shift warp (shift=1 -> identity).
    /// Matches hy3dmlx.sampler.flow_match_sigmas exactly.
    public static func flowMatchSigmas(_ steps: Int, shift: Float = 1.0) -> MLXArray {
        let s = (0 ..< steps).map { i -> Float in
            let x = steps == 1 ? 0 : Float(i) / Float(steps - 1)
            return shift * x / (1 + (shift - 1) * x)
        }
        return MLXArray(s)
    }

    /// ConsistencyFlowMatchEulerDiscrete grid for turbo / PCM-distilled models (guidance_embed).
    /// Ports hy3dmlx.sampler.consistency_sigmas exactly.
    public static func consistencySigmas(_ steps: Int, pcmTimesteps: Int = 100,
                                         numTrainTimesteps: Int = 1000) -> MLXArray {
        func base(_ i: Int) -> Float { Float(i) / Float(numTrainTimesteps - 1) }   // linspace(0,1,numTrain)
        let stepRatio = numTrainTimesteps / pcmTimesteps
        var eulerIdx = [0]
        for k in 1 ..< pcmTimesteps { eulerIdx.append(k * stepRatio - 1) }          // round(k*ratio)-1, prepend 0
        let pcm = eulerIdx.map(base)                                                // [pcmTimesteps]
        var sigmas = [Float]()
        for i in 0 ..< steps {                                                      // floor(linspace(0,pcm,steps,endpoint=False))
            let idx = Int((Float(pcmTimesteps) * Float(i) / Float(steps)).rounded(.down))
            sigmas.append(pcm[min(idx, pcm.count - 1)])
        }
        return MLXArray(sigmas)
    }

    /// Standard-normal latent noise [1, numLatents, 64] for a given seed.
    public static func noise(numLatents: Int, channels: Int = 64, seed: UInt64) -> MLXArray {
        MLXRandom.normal([1, numLatents, channels], key: MLXRandom.key(seed)).asType(.float32)
    }
}
