import Foundation
import MLX

/// DDIM (v-prediction, zero-terminal-SNR, trailing). acp recomputed in Double to match Python float64.
public final class DDIMScheduler {
    let acp: [Double]
    public let timesteps: [Int]
    let ratio: Int
    let finalAlpha: Double = 1.0

    public init(timesteps: [Int], ratio: Int,
                numTrain: Int = 1000, betaStart: Double = 0.00085, betaEnd: Double = 0.012) {
        // betas = linspace(sqrt(bs), sqrt(be), N)^2 ; acp = cumprod(1-betas) ; zero-terminal-SNR rescale.
        let a = betaStart.squareRoot(), b = betaEnd.squareRoot()
        var acpRaw = [Double](repeating: 0, count: numTrain)
        var prod = 1.0
        for i in 0 ..< numTrain {
            let bs = a + (b - a) * Double(i) / Double(numTrain - 1)
            prod *= (1.0 - bs * bs)
            acpRaw[i] = prod
        }
        // _rescale_zero_terminal_snr on alphas_cumprod
        var absSqrt = acpRaw.map { $0.squareRoot() }
        let a0 = absSqrt[0], aT = absSqrt[numTrain - 1]
        for i in 0 ..< numTrain { absSqrt[i] = (absSqrt[i] - aT) * (a0 / (a0 - aT)) }
        var out = absSqrt.map { $0 * $0 }
        out[numTrain - 1] = max(out[numTrain - 1], 1e-8)
        self.acp = out
        self.timesteps = timesteps
        self.ratio = ratio
    }

    public func step(_ v: MLXArray, _ t: Int, _ sample: MLXArray) -> MLXArray {
        let prevT = t - ratio
        let aT = acp[t]
        let aPrev = prevT >= 0 ? acp[prevT] : finalAlpha
        let saT = Float(aT.squareRoot()), somaT = Float((1 - aT).squareRoot())
        let x0 = saT * sample - somaT * v
        let eps = saT * v + somaT * sample
        return Float(aPrev.squareRoot()) * x0 + Float((1 - aPrev).squareRoot()) * eps
    }
}

/// Compute UniPC (sigmas, timesteps) for n steps — float32 path matching Python set_timesteps.
public func uniPCSchedule(_ n: Int, numTrain: Int = 1000,
                          betaStart: Double = 0.00085, betaEnd: Double = 0.012) -> (sigmas: [Float], timesteps: [Int]) {
    let a = betaStart.squareRoot(), b = betaEnd.squareRoot()
    var betas = (0..<numTrain).map { i -> Float in let bs = Float(a + (b - a) * Double(i) / Double(numTrain - 1)); return bs * bs }
    // _rescale_betas_zero_snr (float32)
    var acp = [Float](repeating: 0, count: numTrain); var p: Float = 1
    for i in 0..<numTrain { p *= (1 - betas[i]); acp[i] = p }
    var bar = acp.map { $0.squareRoot() }
    let a0 = bar[0], aT = bar[numTrain - 1]
    for i in 0..<numTrain { bar[i] = (bar[i] - aT) * (a0 / (a0 - aT)) }
    let abar = bar.map { $0 * $0 }
    var alphas = [Float](repeating: 0, count: numTrain); alphas[0] = abar[0]
    for i in 1..<numTrain { alphas[i] = abar[i] / abar[i - 1] }
    betas = alphas.map { 1 - $0 }
    var acp2 = [Float](repeating: 0, count: numTrain); p = 1
    for i in 0..<numTrain { p *= (1 - betas[i]); acp2[i] = p }
    acp2[numTrain - 1] = Float(pow(2.0, -24))
    let trainSig = (0..<numTrain).map { (((1 - acp2[$0]) / acp2[$0]).squareRoot()) }
    // trailing timesteps: arange(numTrain, 0, -ratio).round() - 1
    let ratio = Double(numTrain) / Double(n)
    var ts = [Int](); var k = 0
    while true { let val = Double(numTrain) - ratio * Double(k); if val <= 0 { break }; ts.append(Int(val.rounded()) - 1); k += 1 }
    let clamped = ts.map { Swift.max(0, Swift.min(numTrain - 1, $0)) }
    var sigmas = clamped.map { trainSig[$0] }; sigmas.append(0)
    return (sigmas, clamped)
}

/// UniPC multistep (v_prediction, predict_x0, bh2, order-2, trailing, lower_order_final).
/// sigmas + timesteps come from the Python tables (float32); the scalar math is Double (matches Python).
public final class UniPCScheduler {
    let sigmas: [Double]
    public let timesteps: [Int]
    let solverOrder = 2
    let predictX0 = true
    let solverType = "bh2"
    let lowerOrderFinal = true

    var modelOutputs: [MLXArray?]
    var lowerOrderNums = 0
    var lastSample: MLXArray?
    var stepIndex: Int?
    var thisOrder = 0

    public init(sigmas: [Float], timesteps: [Int]) {
        self.sigmas = sigmas.map(Double.init)
        self.timesteps = timesteps
        self.modelOutputs = Array(repeating: nil, count: solverOrder)
    }

    private func alphaSigma(_ sigma: Double) -> (Double, Double) {
        let alpha = 1.0 / (sigma * sigma + 1.0).squareRoot()
        return (alpha, sigma * alpha)
    }
    private func logD(_ v: Double) -> Double { v <= 0 ? -Double.infinity : Foundation.log(v) }

    private func initStepIndex(_ t: Int) {
        let cand = timesteps.indices.filter { timesteps[$0] == t }
        if cand.isEmpty { stepIndex = timesteps.count - 1 }
        else if cand.count > 1 { stepIndex = cand[1] }
        else { stepIndex = cand[0] }
    }

    /// solve A x = b (general, Gaussian elimination with partial pivoting).
    private func solve(_ A0: [[Double]], _ b0: [Double]) -> [Double] {
        var A = A0, b = b0, n = b0.count
        for col in 0 ..< n {
            var piv = col
            for r in (col + 1) ..< n where abs(A[r][col]) > abs(A[piv][col]) { piv = r }
            A.swapAt(col, piv); b.swapAt(col, piv)
            let d = A[col][col]
            for r in 0 ..< n where r != col {
                let f = A[r][col] / d
                for c in col ..< n { A[r][c] -= f * A[col][c] }
                b[r] -= f * b[col]
            }
        }
        return (0 ..< n).map { b[$0] / A[$0][$0] }
    }

    private func coeff(_ order: Int, _ rks: [Double], _ hh: Double) -> ([[Double]], [Double], Double, Double) {
        let hPhi1 = Foundation.expm1(hh)
        var hPhiK = hPhi1 / hh - 1
        var fact = 1.0
        let Bh = solverType == "bh1" ? hh : Foundation.expm1(hh)
        var R = [[Double]](), b = [Double]()
        for i in 1 ... order {
            R.append(rks.map { pow($0, Double(i - 1)) })
            b.append(hPhiK * fact / Bh)
            fact *= Double(i + 1)
            hPhiK = hPhiK / hh - 1.0 / fact
        }
        return (R, b, Bh, hPhi1)
    }

    private func convert(_ modelOutput: MLXArray, _ sample: MLXArray) -> MLXArray {
        let (alphaT, sigmaT) = alphaSigma(sigmas[stepIndex!])
        return Float(alphaT) * sample - Float(sigmaT) * modelOutput   // predict_x0 + v_pred
    }

    private func predictor(_ sample: MLXArray, _ order: Int) -> MLXArray {
        let si = stepIndex!, m0 = modelOutputs.last!!
        let (alphaT, sigmaT) = alphaSigma(sigmas[si + 1])
        let (alphaS0, sigmaS0) = alphaSigma(sigmas[si])
        let h = (logD(alphaT) - logD(sigmaT)) - (logD(alphaS0) - logD(sigmaS0))
        let lambdaS0 = logD(alphaS0) - logD(sigmaS0)
        var rks = [Double](), D1s = [MLXArray]()
        for i in 1 ..< order {
            let mi = modelOutputs[modelOutputs.count - 1 - i]!
            let (aSi, sSi) = alphaSigma(sigmas[si - i])
            let rk = ((logD(aSi) - logD(sSi)) - lambdaS0) / h
            rks.append(rk); D1s.append((mi - m0) * Float(1.0 / rk))
        }
        rks.append(1.0)
        let hh = predictX0 ? -h : h
        let (R, b, Bh, hPhi1) = coeff(order, rks, hh)
        var rhos = [Double]()
        if !D1s.isEmpty { rhos = order == 2 ? [0.5] : { var rr = R; rr.removeLast(); for i in rr.indices { rr[i].removeLast() }; return solve(rr, Array(b.dropLast())) }() }
        let xt_ = predictX0 ? (Float(sigmaT / sigmaS0) * sample - Float(alphaT * hPhi1) * m0)
                            : (Float(alphaT / alphaS0) * sample - Float(sigmaT * hPhi1) * m0)
        if D1s.isEmpty { return xt_ }
        var pred = Float(rhos[0]) * D1s[0]
        for k in 1 ..< D1s.count { pred = pred + Float(rhos[k]) * D1s[k] }
        return predictX0 ? (xt_ - Float(alphaT * Bh) * pred) : (xt_ - Float(sigmaT * Bh) * pred)
    }

    private func corrector(_ thisModelOutput: MLXArray, _ lastSample: MLXArray, _ order: Int) -> MLXArray {
        let si = stepIndex!, m0 = modelOutputs.last!!
        let (alphaT, sigmaT) = alphaSigma(sigmas[si])
        let (alphaS0, sigmaS0) = alphaSigma(sigmas[si - 1])
        let lambdaS0 = logD(alphaS0) - logD(sigmaS0)
        let h = (logD(alphaT) - logD(sigmaT)) - lambdaS0
        var rks = [Double](), D1s = [MLXArray]()
        for i in 1 ..< order {
            let mi = modelOutputs[modelOutputs.count - 1 - i]!
            let (aSi, sSi) = alphaSigma(sigmas[si - (i + 1)])
            let rk = ((logD(aSi) - logD(sSi)) - lambdaS0) / h
            rks.append(rk); D1s.append((mi - m0) * Float(1.0 / rk))
        }
        rks.append(1.0)
        let hh = predictX0 ? -h : h
        let (R, b, Bh, hPhi1) = coeff(order, rks, hh)
        let rhos = order == 1 ? [0.5] : solve(R, b)
        let xt_ = predictX0 ? (Float(sigmaT / sigmaS0) * lastSample - Float(alphaT * hPhi1) * m0)
                            : (Float(alphaT / alphaS0) * lastSample - Float(sigmaT * hPhi1) * m0)
        let D1t = thisModelOutput - m0
        var inner = Float(rhos[rhos.count - 1]) * D1t
        for k in 0 ..< D1s.count { inner = inner + Float(rhos[k]) * D1s[k] }
        return predictX0 ? (xt_ - Float(alphaT * Bh) * inner) : (xt_ - Float(sigmaT * Bh) * inner)
    }

    public func step(_ modelOutput: MLXArray, _ t: Int, _ sample: MLXArray) -> MLXArray {
        if stepIndex == nil { initStepIndex(t) }
        let useCorrector = stepIndex! > 0 && lastSample != nil
        let moConvert = convert(modelOutput, sample)
        var samp = sample
        if useCorrector { samp = corrector(moConvert, lastSample!, thisOrder) }
        for i in 0 ..< (solverOrder - 1) { modelOutputs[i] = modelOutputs[i + 1] }
        modelOutputs[solverOrder - 1] = moConvert
        let calc = lowerOrderFinal ? min(solverOrder, timesteps.count - stepIndex!) : solverOrder
        thisOrder = min(calc, lowerOrderNums + 1)
        lastSample = samp
        let prev = predictor(samp, thisOrder)
        if lowerOrderNums < solverOrder { lowerOrderNums += 1 }
        stepIndex! += 1
        return prev
    }
}
