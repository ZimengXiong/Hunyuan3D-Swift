import XCTest
import MLX
import HunyuanPaintMLX

/// Threshold-gated paint parity vs the known-good Python MLX port. Thresholds are DESIGN.md §7.
/// Every test XCTSkips when its fixture is absent, so the suite is green with zero fixtures present.
/// Fixture names + tensor keys are documented in parity/README.md.
final class PaintParityTests: XCTestCase {
    let fx = FixtureStore()

    // §7: Paint VAE enc/dec — maxabs ≤ 1e-6 (bit-exact)
    func testPaintVAE() throws {
        let vae = PaintVAE(try fx.requireW("vae_weights.safetensors"))
        let f = try fx.require("vae_fixture.safetensors")
        XCTAssertLessThanOrEqual(Metric.maxabs(vae.decode(f["z"]!), f["img"]!), 1e-6, "VAE decode")
        XCTAssertLessThanOrEqual(Metric.maxabs(vae.encodeMean(f["ximg"]!), f["mean"]!), 1e-6, "VAE encode_mean")
    }

    // §7: DDIM trajectory — maxabs ≤ 1e-6 (bit-exact)
    func testDDIMTrajectory() throws {
        let f = try fx.require("sched_fixture.safetensors")
        let vs = f["vs"]!
        let dts = f["ddim_timesteps"]!.asArray(Int32.self).map(Int.init)
        let ratio = Int(f["ddim_ratio"]!.asArray(Int32.self)[0])
        let ddim = DDIMScheduler(timesteps: dts, ratio: ratio)
        var x = f["x0"]!
        for i in 0 ..< dts.count { x = ddim.step(vs[i], dts[i], x) }
        eval(x)
        XCTAssertLessThanOrEqual(Metric.maxabs(x, f["ddim_traj"]!), 1e-6, "DDIM trajectory")
    }

    // §7: UniPC trajectory — maxabs ≤ 1e-5 (measured 3.6e-7)
    func testUniPCTrajectory() throws {
        let f = try fx.require("sched_fixture.safetensors")
        let vs = f["vs"]!
        let usig = f["unipc_sigmas"]!.asArray(Float.self)
        let uts = f["unipc_timesteps"]!.asArray(Int32.self).map(Int.init)
        let unipc = UniPCScheduler(sigmas: usig, timesteps: uts)
        var x = f["x0"]!
        for i in 0 ..< uts.count { x = unipc.step(vs[i], uts[i], x) }
        eval(x)
        XCTAssertLessThanOrEqual(Metric.maxabs(x, f["unipc_traj"]!), 1e-5, "UniPC trajectory")
    }

    // §7: SD2.1 UNet fwd — maxabs ≤ 1e-4 (measured 3.2e-6)
    func testSD21UNet() throws {
        let unet = PaintUNet(try fx.requireW("unet_base_weights.safetensors"))
        let f = try fx.require("unet_base_fixture.safetensors")
        let out = unet(f["sample"]!, f["ts"]!, f["ctx"]!); eval(out)
        XCTAssertLessThanOrEqual(Metric.maxabs(out, f["out"]!), 1e-4, "SD2.1 UNet forward")
    }

    // §7: PBR UNet fwd (MDA+RA+MA+DINO+RoPE) — cos ≥ 0.9999 (measured 8.2e-5 maxabs)
    func testPBRUNet() throws {
        let all = (try fx.require("pbr_unet_fixture.safetensors")).mapValues { $0.asType(.float32) }
        var wd = [String: MLXArray](), ced = [String: MLXArray]()
        var rcos = [Int: MLXArray](), rsin = [Int: MLXArray]()
        for (k, v) in all {
            if k.hasPrefix("ced::") { ced[String(k.dropFirst(5))] = v }
            else if k.hasPrefix("rope::") {
                let c = k.components(separatedBy: "::"); let tok = Int(c[1])!
                if c[2] == "cos" { rcos[tok] = v } else { rsin[tok] = v }
            } else if !k.hasPrefix("_"), k != "dino" { wd[k] = v }
        }
        var rope = [Int: (MLXArray, MLXArray)]()
        for (tok, c) in rcos { rope[tok] = (c, rsin[tok]!) }
        let xa = XAttn(mode: "r", numInBatch: 2, conditionEmbed: ced, refScale: 1, mvaScale: 1,
                       nPbr: 2, dino: all["dino"]!, ropeByTokens: rope)
        let unet = PaintUNet(W(wd), useMa: true, useRa: true, useMda: true, useDino: true, pbr: true)
        let out = unet(all["_in.s"]!, all["_in.t"]!, all["_in.ehs"]!, xattn: xa); eval(out)
        XCTAssertGreaterThanOrEqual(Metric.cosine(out, all["_out"]!), 0.9999, "PBR UNet forward")
    }

    // §7: RealESRGAN — maxabs ≤ 1e-6 (bit-exact)
    func testRealESRGAN() throws {
        let sr = RealESRGAN(try fx.requireW("resrgan_weights.safetensors"))
        let f = try fx.require("resrgan_fixture.safetensors")
        let y = sr(f["x"]!); eval(y)
        XCTAssertLessThanOrEqual(Metric.maxabs(y, f["y"]!), 1e-6, "RealESRGAN x4")
    }

    // §7: Rasterizer face-id / bary — 100% / ≤ 2e-4 (bit-exact / 1.8e-7)
    func testRasterizer() throws {
        let f = try fx.require("raster_fixture.safetensors")
        let (fi, ba) = SwiftRaster.rasterize(f["V"]!, f["F"]!, 128); eval(fi, ba)
        XCTAssertEqual(Metric.matchFraction(fi, f["findices"]!), 1.0, "rasterizer face-id match")
        let bmax = abs(ba - f["bary"]!).max(); eval(bmax)
        XCTAssertLessThanOrEqual(bmax.item(Float.self), 2e-4, "rasterizer bary maxabs")
        let interp = SwiftRaster.interpolate(f["col"]!, fi, ba, f["F"]!); eval(interp)
        XCTAssertLessThanOrEqual(Metric.maxabs(interp, f["interp"]!), 2e-4, "rasterizer interpolate")
    }

    // §7: Control maps (normal/position) — PSNR ≥ 80 dB (measured 88–167 dB)
    func testControlMaps() throws {
        let f = try fx.require("render_fixture.safetensors")
        let R = MeshRender()
        R.loadMesh(f["V"]!.asArray(Float.self), f["F"]!.asArray(Int32.self).map { UInt32($0) })
        let (normal, position) = R.renderControl(0, 30, 256); eval(normal, position)
        XCTAssertGreaterThanOrEqual(Metric.psnr(normal, f["normal"]!), 80, "control normal PSNR")
        XCTAssertGreaterThanOrEqual(Metric.psnr(position, f["position"]!), 80, "control position PSNR")
    }

    // §7: Bake — PSNR ≥ 100 dB (measured 151 dB)
    func testBake() throws {
        let f = try fx.require("bake_fixture.safetensors")
        let R = MeshRender()
        R.loadMesh(f["V"]!.asArray(Float.self), f["F"]!.asArray(Int32.self).map { UInt32($0) })
        R.setUV(f["uv"]!.asArray(Float.self), flipV: true)
        let viewsArr = f["views"]!
        let views = (0..<6).map { viewsArr[$0] }
        let elevs: [Float] = [0, 0, 0, 0, 90, -90], azims: [Float] = [0, 90, 180, 270, 0, 180]
        let vw: [Float] = [1, 0.1, 0.5, 0.1, 0.05, 0.05]
        let (texs, covered) = R.bakeMulti([views], elevs, azims, textureSize: 256, weights: vw); eval(texs[0], covered)
        XCTAssertEqual(Metric.matchFraction(covered, f["covered"]!), 1.0, "bake coverage match")
        XCTAssertGreaterThanOrEqual(Metric.psnr(texs[0], f["tex"]!), 100, "bake texture PSNR")
    }

    // §7: Paint RGB e2e (3-step, fixed UVs) — cos ≥ 0.999. Guidance is read from the fixture
    // (dumped at 2.0, the RGB pipeline's shipping value; legacy fixtures baked 3.0 and lack the key).
    func testPaintRGBe2e() throws {
        let all = (try fx.require("p20_e2e_fixture.safetensors")).mapValues { $0.asType(.float32) }
        var mainW = [String: MLXArray](), dualW = [String: MLXArray]()
        for (k, v) in all {
            if k.hasPrefix("main::") { mainW[String(k.dropFirst(6))] = v }
            else if k.hasPrefix("dual::") { dualW[String(k.dropFirst(6))] = v }
        }
        let wrap = Paint20Wrapper(main: W(mainW), dual: W(dualW))
        let GUID: Float = all["guidance"]?.item(Float.self) ?? 3.0
        let ced = wrap.prepare(refLat: all["ref_lat"]!)
        let (sig, ts) = uniPCSchedule(3)
        let sched = UniPCScheduler(sigmas: sig, timesteps: ts)
        var latents = all["latents0"]!
        let camGen: [Int32] = [0, 1]
        let gen = mainW["learned_text_clip_gen"]!, neg = zeros(gen.shape)
        let nl = all["normal_lat"]!, pl = all["position_lat"]!
        for t in ts {
            let tArr = MLXArray(Array(repeating: Float(t), count: 2))
            let vc = wrap.predict(latents, tArr, text: gen, normalLat: nl, positionLat: pl, camGen: camGen, ced: ced, mvaScale: 1, refScale: 1)
            let vu = wrap.predict(latents, tArr, text: neg, normalLat: nl, positionLat: pl, camGen: camGen, ced: nil, mvaScale: 1, refScale: 0)
            latents = sched.step(vu + GUID * (vc - vu), t, latents); eval(latents)
        }
        XCTAssertGreaterThanOrEqual(Metric.cosine(latents, all["final"]!), 0.999, "Paint RGB e2e")
    }

    // §7: Paint PBR e2e (3-step, fixed UVs) — cos ≥ 0.999 (measured 0.9999994). GUID matches fixture.
    func testPaintPBRe2e() throws {
        let all = (try fx.require("pbr_e2e_fixture.safetensors")).mapValues { $0.asType(.float32) }
        var mainW = [String: MLXArray](), dualW = [String: MLXArray]()
        for (k, v) in all {
            if k.hasPrefix("main::") { mainW[String(k.dropFirst(6))] = v }
            else if k.hasPrefix("dual::") { dualW[String(k.dropFirst(6))] = v }
        }
        let wrap = PBRWrapper(main: W(mainW), dual: W(dualW), nPbr: 2)
        let GUID: Float = all["guidance"]?.item(Float.self) ?? 3.0
        var pcos = [Int: MLXArray](), psin = [Int: MLXArray]()
        for (k, v) in all where k.hasPrefix("prope::") {
            let c = k.components(separatedBy: "::"); let tok = Int(c[1])!
            if c[2] == "cos" { pcos[tok] = v } else { psin[tok] = v }
        }
        var pyRope = [Int: (MLXArray, MLXArray)]()
        for (tok, c) in pcos { pyRope[tok] = (c, psin[tok]!) }
        // Self-computed PoseRoPE: the fp16 voxel indices are ported exactly, so the tables no
        // longer need to be injected from Python — assert they match, then run the loop on them.
        if let pvox = all["pvox8"] {
            let vox = PBRWrapper.voxelIndices(all["posmap"]!, gridRes: 8, voxelRes: 64); eval(vox)
            XCTAssertEqual(Metric.matchFraction(vox, pvox), 1.0, "fp16 voxel indices exact (grid 8)")
        }
        let rope = wrap.ropeByTokens(all["posmap"]!, hLat: 8, nGen: 2)
        for (tok, py) in pyRope {
            guard let mine = rope[tok] else { XCTFail("missing self-computed RoPE for \(tok) tokens"); continue }
            XCTAssertLessThanOrEqual(Metric.maxabs(mine.0, py.0), 1e-6, "RoPE cos (\(tok) tokens)")
            XCTAssertLessThanOrEqual(Metric.maxabs(mine.1, py.1), 1e-6, "RoPE sin (\(tok) tokens)")
        }
        let (ced, dino, _) = wrap.prepare(refLat: all["ref_lat"]!, dinoHidden: all["dino_hs"]!,
                                          posmap: all["posmap"]!, H: 8, nGen: 2)
        let dinoZero = zeros(dino.shape)
        let uts = all["unipc_timesteps"]!.asArray(Int32.self).map(Int.init)
        let sched = UniPCScheduler(sigmas: all["unipc_sigmas"]!.asArray(Float.self), timesteps: uts)
        var latents = all["latents0"]!
        let nb = latents.dim(0) * latents.dim(1) * latents.dim(2)
        let nl = all["normal_lat"]!, pl = all["position_lat"]!
        for i in 0 ..< uts.count {
            let t = MLXArray(Array(repeating: Float(uts[i]), count: nb))
            let vc = wrap.predict(latents, t, normalLat: nl, positionLat: pl, ced: ced, dino: dino, rope: rope, mvaScale: 1, refScale: 1)
            let vu = wrap.predict(latents, t, normalLat: nl, positionLat: pl, ced: nil, dino: dinoZero, rope: rope, mvaScale: 1, refScale: 0)
            latents = sched.step(vu + GUID * (vc - vu), uts[i], latents); eval(latents)
        }
        XCTAssertGreaterThanOrEqual(Metric.cosine(latents, all["final"]!), 0.999, "Paint PBR e2e")
    }

    // PoseRoPE fp16 voxel indices — exact vs numpy fp16 reference at the real pipeline levels
    // (posmap 512x512; grid_res 64/32/16/8, voxel_res 512/256/128/64).
    // voxel_fixture.safetensors: pos + vox{64,32,16,8} from compute_voxel_indices.
    func testVoxelIndices() throws {
        let f = try fx.require("voxel_fixture.safetensors")
        for (g, vr) in [(64, 512), (32, 256), (16, 128), (8, 64)] {
            let vox = PBRWrapper.voxelIndices(f["pos"]!, gridRes: g, voxelRes: vr); eval(vox)
            XCTAssertEqual(Metric.matchFraction(vox, f["vox\(g)"]!), 1.0,
                           "voxel indices exact (grid \(g), voxel \(vr))")
        }
    }

    // §7: DINOv2-giant (feeds the PBR path) — cos ≥ 0.9999
    func testDINOv2Giant() throws {
        let dino = Dinov2(try fx.requireW("dino_weights.safetensors"))
        let f = try fx.require("dino_fixture.safetensors")
        let out = dino(f["px"]!); eval(out)
        XCTAssertGreaterThanOrEqual(Metric.cosine(out, f["out"]!), 0.9999, "DINOv2-giant")
    }

    // §7: Inpaint (new impl) — exact vs Python. The Swift implementation ports scipy's
    // Euclidean feature transform (incl. tie-breaking) and OpenCV's Navier-Stokes inpaint,
    // so all three stages gate at bit-exact: EDT indices, EDT fill, and the final result.
    // inpaint_fixture.safetensors: texture, covered, filled (full Python reference),
    // filled_edt (EDT-only intermediate), edt_rows/edt_cols (scipy feature indices).
    func testInpaint() throws {
        let f = try fx.require("inpaint_fixture.safetensors")
        // EDT nearest-fill: index-exact (ties included) and value-exact
        let (er, ec) = MeshRender.edtIndices(f["covered"]!); eval(er, ec)
        XCTAssertEqual(Metric.matchFraction(er, f["edt_rows"]!), 1.0, "EDT row indices exact")
        XCTAssertEqual(Metric.matchFraction(ec, f["edt_cols"]!), 1.0, "EDT col indices exact")
        if let fe = f["filled_edt"] {
            let clipped = clip(f["texture"]!.asType(.float32), min: 0, max: 1)
            let flat = clipped.reshaped([-1, 3])
            let gather = take(flat, (er.asType(.int32) * f["covered"]!.dim(1) + ec.asType(.int32)).reshaped([-1]), axis: 0)
            XCTAssertEqual(Metric.maxabs(gather.reshaped(fe.shape), fe), 0, "EDT fill bit-exact")
        }
        // Full pipeline (EDT + uint8 round-trip + Navier-Stokes): bit-exact vs Python
        let out = MeshRender.inpaint(f["texture"]!, f["covered"]!); eval(out)
        XCTAssertEqual(Metric.maxabs(out, f["filled"]!), 0, "inpaint bit-exact vs Python")
    }
}
