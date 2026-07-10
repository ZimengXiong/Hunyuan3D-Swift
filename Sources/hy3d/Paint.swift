import Foundation
import MLX
import MLXRandom
import HunyuanPaintMLX

// MARK: - hy3d paint

func cmdPaint(_ args: Args) throws {
    guard args.positional.count >= 2 else {
        throw CLIError("paint: usage: hy3d paint <mesh.glb|obj> <image.png> -o <out.glb> --weights <dir>")
    }
    let meshPath = args.positional[0], imagePath = args.positional[1]
    guard let out = args.str("o", "output") else { throw CLIError("paint: missing -o <out.glb>") }
    guard let weights = args.str("weights") else { throw CLIError("paint: missing --weights <dir>") }
    let model = (args.str("model") ?? "rgb").lowercased()
    let res = args.int("res") ?? 512
    let steps = args.int("steps") ?? 15
    let tex = args.int("tex") ?? (model == "pbr" ? 4096 : 2048)
    let superRes = !args.flag("no-superres")
    if let seed = args.int("seed") { MLXRandom.seed(UInt64(seed)) }   // see note in `hy3d help`

    let pipe = PaintPipeline(weightsRoot: weights, res: res, steps: steps, tex: tex, superRes: superRes)
    print("paint (\(model)): mesh=\(meshPath) image=\(imagePath) res=\(res) steps=\(steps) tex=\(tex) super-res=\(superRes)")
    switch model {
    case "pbr":
        try pipe.run(meshPath: meshPath, imagePath: imagePath, outGLB: out)
        print("paint: wrote \(out)")
    case "rgb":
        let lm = loadMesh(meshPath)
        guard lm.vertexCount > 0 else { throw CLIError("paint: failed to load mesh \(meshPath)") }
        guard let r = try pipe.paintRGB(mesh: lm, imagePath: imagePath, onProgress: { s, f in
            print(String(format: "  [%3.0f%%] %@", f * 100, s))
        }) else { throw CLIError("paint: RGB pipeline returned no result (UV unwrap failed?)") }
        try writeGLB(path: out, vertices: r.vertices, faces: r.faces, uvs: r.uvs,
                     baseColorPNG: r.albedoPNG, metallicRoughnessPNG: nil)
        print("paint: wrote \(out)")
    default:
        throw CLIError("paint: --model must be rgb or pbr")
    }
}

// MARK: - hy3d parity-paint (print-panel; ported from the v1 paint-cli parity harness)

func cmdParityPaint(_ args: Args) throws {
    let fx = FixtureStore(args)
    let allFixtures = [
        "vae_fixture.safetensors", "vae_weights.safetensors", "sched_fixture.safetensors",
        "unet_base_fixture.safetensors", "unet_base_weights.safetensors", "pbr_unet_fixture.safetensors",
        "resrgan_fixture.safetensors", "resrgan_weights.safetensors", "raster_fixture.safetensors",
        "render_fixture.safetensors", "bake_fixture.safetensors", "p20_e2e_fixture.safetensors",
        "pbr_e2e_fixture.safetensors", "dino_fixture.safetensors", "dino_weights.safetensors",
        "image_fixture.safetensors", "inpaint_fixture.safetensors", "voxel_fixture.safetensors",
    ]
    guard fx.anyExists(allFixtures) else {
        print("no fixtures found at \(fx.dir)")
        return
    }
    print("paint parity — fixtures: \(fx.dir)")

    func report(_ name: String, _ got: MLXArray, _ exp: MLXArray) {
        print(String(format: "  %@: cosine %.7f  maxabs %.3e  psnr %.1f dB",
                     name, Metric.cosine(got, exp), Metric.maxabs(got, exp), Metric.psnr(got, exp)))
    }
    func loadW(_ name: String) throws -> W { W(try fx.load(name).mapValues { $0.asType(.float32) }) }

    // ---- Image preprocessing (prepRGB vs PIL); reference image = <fixtures>/input.png ----
    if fx.exists("image_fixture.safetensors") {
        let f = try fx.load("image_fixture.safetensors")
        if fx.exists("input.png") {
            let got = prepRGB(fx.path("input.png"), 512); eval(got)
            report("Image prepRGB  ", got, f["ref512"]!)
        } else {
            print("  Image prepRGB    [skipped: put the reference image at \(fx.path("input.png"))]")
        }
    }

    // ---- MeshRender control maps + uv_rasterize ----
    if fx.exists("render_fixture.safetensors") {
        let f = try fx.load("render_fixture.safetensors")
        let R = MeshRender()
        R.loadMesh(f["V"]!.asArray(Float.self), f["F"]!.asArray(Int32.self).map { UInt32($0) })
        let (normal, position) = R.renderControl(0, 30, 256); eval(normal, position)
        report("Render normal  ", normal, f["normal"]!)
        report("Render position", position, f["position"]!)
        R.setUV(f["uv"]!.asArray(Float.self), flipV: true)
        let (texPos, _, m) = R.uvRasterize(512); eval(texPos, m)
        let mMatch = (m.asType(.int32) .== f["mask"]!.asType(.int32)).asType(.float32).mean(); eval(mMatch)
        print(String(format: "  Render uv mask match %.6f", mMatch.item(Float.self)))
        report("Render uv_pos  ", texPos, f["tex_pos"]!)
    }

    // ---- inpaint (EDT nearest-fill + Navier-Stokes; exact port, expect maxabs 0) ----
    if fx.exists("inpaint_fixture.safetensors") {
        let f = try fx.load("inpaint_fixture.safetensors")
        let (er, ec) = MeshRender.edtIndices(f["covered"]!); eval(er, ec)
        let idxMatch = ((er.asType(.int32) .== f["edt_rows"]!.asType(.int32))
            .&& (ec.asType(.int32) .== f["edt_cols"]!.asType(.int32))).asType(.float32).mean()
        eval(idxMatch)
        print(String(format: "  Inpaint EDT index match %.6f", idxMatch.item(Float.self)))
        let out = MeshRender.inpaint(f["texture"]!, f["covered"]!); eval(out)
        report("Inpaint (EDT+NS)", out, f["filled"]!)
    }

    // ---- PoseRoPE fp16 voxel indices (512x512 posmap, all four pipeline levels) ----
    if fx.exists("voxel_fixture.safetensors") {
        let f = try fx.load("voxel_fixture.safetensors")
        for (g, vr) in [(64, 512), (32, 256), (16, 128), (8, 64)] {
            let vox = PBRWrapper.voxelIndices(f["pos"]!, gridRes: g, voxelRes: vr)
            let diff = abs(vox.asType(.int32) - f["vox\(g)"]!.asType(.int32)).max(); eval(diff)
            print("  Voxel indices (grid \(g), voxel \(vr)): maxdiff \(diff.item(Int32.self))")
        }
    }

    // ---- bake_multi (back-sample gather) ----
    if fx.exists("bake_fixture.safetensors") {
        let f = try fx.load("bake_fixture.safetensors")
        let R = MeshRender()
        R.loadMesh(f["V"]!.asArray(Float.self), f["F"]!.asArray(Int32.self).map { UInt32($0) })
        R.setUV(f["uv"]!.asArray(Float.self), flipV: true)
        let viewsArr = f["views"]!                                     // [6,256,256,3]
        let views = (0..<6).map { viewsArr[$0] }
        let elevs: [Float] = [0, 0, 0, 0, 90, -90], azims: [Float] = [0, 90, 180, 270, 0, 180]
        let vw: [Float] = [1, 0.1, 0.5, 0.1, 0.05, 0.05]
        let (texs, covered) = R.bakeMulti([views], elevs, azims, textureSize: 256, weights: vw); eval(texs[0], covered)
        let cMatch = (covered.asType(.int32) .== f["covered"]!.asType(.int32)).asType(.float32).mean(); eval(cMatch)
        print(String(format: "  Bake covered match %.6f", cMatch.item(Float.self)))
        report("Bake texture   ", texs[0], f["tex"]!)
    }

    // ---- GPU rasterizer (Metal kernel) ----
    if fx.exists("raster_fixture.safetensors") {
        let f = try fx.load("raster_fixture.safetensors")
        let (fi, ba) = SwiftRaster.rasterize(f["V"]!, f["F"]!, 128); eval(fi, ba)
        let match = (fi.asType(.int32) .== f["findices"]!.asType(.int32)).asType(.float32).mean()
        let bmax = abs(ba - f["bary"]!).max(); eval(match, bmax)
        print(String(format: "  Raster (Metal): face-id match %.6f  bary maxabs %.3e",
                     match.item(Float.self), bmax.item(Float.self)))
        let interp = SwiftRaster.interpolate(f["col"]!, fi, ba, f["F"]!); eval(interp)
        report("Raster interp  ", interp, f["interp"]!)
    }

    // ---- VAE ----
    if fx.exists("vae_weights.safetensors"), fx.exists("vae_fixture.safetensors") {
        let vae = PaintVAE(try loadW("vae_weights.safetensors"))
        let f = try fx.load("vae_fixture.safetensors")
        let dec = vae.decode(f["z"]!); eval(dec)
        report("VAE decode     ", dec, f["img"]!)
        let mean = vae.encodeMean(f["ximg"]!); eval(mean)
        report("VAE encode_mean", mean, f["mean"]!)
    }

    // ---- RealESRGAN x4 ----
    if fx.exists("resrgan_weights.safetensors"), fx.exists("resrgan_fixture.safetensors") {
        let sr = RealESRGAN(try loadW("resrgan_weights.safetensors"))
        let f = try fx.load("resrgan_fixture.safetensors")
        let y = sr(f["x"]!); eval(y)
        report("RealESRGAN x4  ", y, f["y"]!)
    }

    // ---- SD2.1 UNet backbone (base, random weights) ----
    if fx.exists("unet_base_weights.safetensors"), fx.exists("unet_base_fixture.safetensors") {
        let unet = PaintUNet(try loadW("unet_base_weights.safetensors"))
        let f = try fx.load("unet_base_fixture.safetensors")
        let out = unet(f["sample"]!, f["ts"]!, f["ctx"]!); eval(out)
        report("UNet backbone  ", out, f["out"]!)
    }

    // ---- Full PBR diffusion core end-to-end (prepare + loop + CFG); GUID matches the fixture ----
    if fx.exists("pbr_e2e_fixture.safetensors") {
        let all = try fx.load("pbr_e2e_fixture.safetensors").mapValues { $0.asType(.float32) }
        var mainW = [String: MLXArray](), dualW = [String: MLXArray]()
        for (k, v) in all {
            if k.hasPrefix("main::") { mainW[String(k.dropFirst(6))] = v }
            else if k.hasPrefix("dual::") { dualW[String(k.dropFirst(6))] = v }
        }
        let wrap = PBRWrapper(main: W(mainW), dual: W(dualW), nPbr: 2)
        let GUID: Float = all["guidance"]?.item(Float.self) ?? 3.0     // recorded by the dumper
        let vtest = PBRWrapper.voxelIndices(all["posmap"]!, gridRes: 8, voxelRes: 64)
        let voxDiff = abs(vtest.asType(.int32) - all["pvox8"]!.asType(.int32)).max()
        eval(voxDiff); print("  e2e: voxel maxdiff vs Python = \(voxDiff.item(Int32.self))")
        var pcos = [Int: MLXArray](), psin = [Int: MLXArray]()
        for (k, v) in all where k.hasPrefix("prope::") {
            let c = k.components(separatedBy: "::"); let tok = Int(c[1])!
            if c[2] == "cos" { pcos[tok] = v } else { psin[tok] = v }
        }
        // RoPE is self-computed (fp16 voxel port is exact); print the drift vs the Python tables.
        let (ced, dino, rope) = wrap.prepare(refLat: all["ref_lat"]!, dinoHidden: all["dino_hs"]!,
                                             posmap: all["posmap"]!, H: 8, nGen: 2)
        var ropeErr: Float = 0
        for (tok, c) in pcos {
            guard let mine = rope[tok] else { ropeErr = .infinity; continue }
            ropeErr = max(ropeErr, Metric.maxabs(mine.0, c), Metric.maxabs(mine.1, psin[tok]!))
        }
        print(String(format: "  e2e: self-computed RoPE tables maxabs vs Python = %.3e", ropeErr))
        eval(dino)
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
        report("PBR diffusion e2e", latents, all["final"]!)
    }

    // ---- Full 2.0 small (RGB) diffusion end-to-end; GUID matches the fixture ----
    if fx.exists("p20_e2e_fixture.safetensors") {
        let all = try fx.load("p20_e2e_fixture.safetensors").mapValues { $0.asType(.float32) }
        var mainW = [String: MLXArray](), dualW = [String: MLXArray]()
        for (k, v) in all {
            if k.hasPrefix("main::") { mainW[String(k.dropFirst(6))] = v }
            else if k.hasPrefix("dual::") { dualW[String(k.dropFirst(6))] = v }
        }
        let wrap = Paint20Wrapper(main: W(mainW), dual: W(dualW))
        let GUID: Float = all["guidance"]?.item(Float.self) ?? 3.0     // recorded by the dumper (2.0)
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
        report("2.0 RGB e2e    ", latents, all["final"]!)
    }

    // ---- DINOv2-giant ----
    if fx.exists("dino_weights.safetensors"), fx.exists("dino_fixture.safetensors") {
        let dino = Dinov2(try loadW("dino_weights.safetensors"))
        let f = try fx.load("dino_fixture.safetensors")
        let out = dino(f["px"]!); eval(out)
        report("DINOv2-giant   ", out, f["out"]!)
    }

    // ---- 2.1 PBR UNet forward (real weights, dual-pass cond + DINO + RoPE) ----
    if fx.exists("pbr_unet_fixture.safetensors") {
        let all = try fx.load("pbr_unet_fixture.safetensors").mapValues { $0.asType(.float32) }
        var wd = [String: MLXArray](), ced = [String: MLXArray]()
        var rcos = [Int: MLXArray](), rsin = [Int: MLXArray]()
        for (k, v) in all {
            if k.hasPrefix("ced::") { ced[String(k.dropFirst(5))] = v }
            else if k.hasPrefix("rope::") {
                let c = k.components(separatedBy: "::")
                let tok = Int(c[1])!
                if c[2] == "cos" { rcos[tok] = v } else { rsin[tok] = v }
            } else if !k.hasPrefix("_"), k != "dino" { wd[k] = v }
        }
        var rope = [Int: (MLXArray, MLXArray)]()
        for (tok, c) in rcos { rope[tok] = (c, rsin[tok]!) }
        let xa = XAttn(mode: "r", numInBatch: 2, conditionEmbed: ced, refScale: 1, mvaScale: 1,
                       nPbr: 2, dino: all["dino"]!, ropeByTokens: rope)
        let unet = PaintUNet(W(wd), useMa: true, useRa: true, useMda: true, useDino: true, pbr: true)
        let out = unet(all["_in.s"]!, all["_in.t"]!, all["_in.ehs"]!, xattn: xa); eval(out)
        report("PBR UNet forward", out, all["_out"]!)
    }

    // ---- Schedulers (DDIM + UniPC trajectory) ----
    if fx.exists("sched_fixture.safetensors") {
        let f = try fx.load("sched_fixture.safetensors")
        let vs = f["vs"]!
        let dts = f["ddim_timesteps"]!.asArray(Int32.self).map(Int.init)
        let ratio = Int(f["ddim_ratio"]!.asArray(Int32.self)[0])
        let ddim = DDIMScheduler(timesteps: dts, ratio: ratio)
        var xd = f["x0"]!
        for i in 0 ..< dts.count { xd = ddim.step(vs[i], dts[i], xd) }
        eval(xd); report("DDIM trajectory ", xd, f["ddim_traj"]!)

        let usig = f["unipc_sigmas"]!.asArray(Float.self)
        let uts = f["unipc_timesteps"]!.asArray(Int32.self).map(Int.init)
        let (csig, cts) = uniPCSchedule(15)
        let sigErr = zip(csig, usig).map { abs($0 - $1) }.max() ?? 0
        let tsErr = zip(cts, uts).map { abs($0 - $1) }.max() ?? 0
        print("  uniPCSchedule(15): sigma maxdiff \(sigErr)  ts maxdiff \(tsErr)")
        let unipc = UniPCScheduler(sigmas: usig, timesteps: uts)
        var xu = f["x0"]!
        for i in 0 ..< uts.count { xu = unipc.step(vs[i], uts[i], xu) }
        eval(xu); report("UniPC trajectory", xu, f["unipc_traj"]!)
    }
}
