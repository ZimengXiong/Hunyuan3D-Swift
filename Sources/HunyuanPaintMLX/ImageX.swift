import CoreGraphics
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers

/// Image preprocessing for paint conditioning — Swift port of the Python `prep_rgb`.
/// Reads STRAIGHT (non-premultiplied) RGBA, resizes all channels with a PIL-faithful
/// bicubic separable resample (a = -0.5, support 2.0), then composites over white.
/// Designed to match PIL's `Image.resize` antialiasing closely for CUDA/MLX parity.
public enum ImageX {

    // MARK: - ImageNet constants

    static let mean: [Float] = [0.485, 0.456, 0.406]
    static let std: [Float] = [0.229, 0.224, 0.225]

    // MARK: - Bicubic (PIL/Pillow `ImagingResample`) kernel

    /// Pillow cubic filter with a = -0.5, support = 2.0.
    /// f(x): x=|x|; if x<1: ((a+2)x - (a+3))x*x + 1 ; elif x<2: (((x-5)x+8)x-4)*a ; else 0
    @inline(__always)
    private static func cubic(_ xIn: Double) -> Double {
        let a = -0.5
        let x = abs(xIn)
        if x < 1.0 {
            return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
            return (((x - 5.0) * x + 8.0) * x - 4.0) * a
        }
        return 0.0
    }

    /// Resample a flat `[Float]` image (row-major, `srcH * srcW * channels`) along ONE axis.
    /// When `horizontal` is true, resamples columns (W: src→dst); otherwise rows (H: src→dst).
    /// Returns a new flat buffer with the resampled axis at `dstSize`.
    private static func resampleAxis(
        _ src: [Float], srcH: Int, srcW: Int, channels c: Int,
        dstSize: Int, horizontal: Bool
    ) -> ([Float], Int, Int) {
        let srcSize = horizontal ? srcW : srcH
        if dstSize == srcSize {
            return (src, srcH, srcW)
        }
        let scale = Double(srcSize) / Double(dstSize)
        let filterscale = max(scale, 1.0)
        let supportS = 2.0 * filterscale          // bicubic support * filterscale
        let ss = 1.0 / filterscale

        // Precompute, per output index: [xmin, count) range and normalized weights.
        var bounds = [(min: Int, count: Int)](repeating: (0, 0), count: dstSize)
        var weightRows = [[Double]](repeating: [], count: dstSize)
        for xx in 0 ..< dstSize {
            let center = (Double(xx) + 0.5) * scale
            var xmin = Int((center - supportS + 0.5).rounded(.down))
            if xmin < 0 { xmin = 0 }
            var xmax = Int((center + supportS + 0.5).rounded(.down))
            if xmax > srcSize { xmax = srcSize }
            let count = xmax - xmin
            var w = [Double](repeating: 0.0, count: max(count, 0))
            var total = 0.0
            for k in 0 ..< count {
                let x = xmin + k
                let wv = cubic((Double(x) + 0.5 - center) * ss)
                w[k] = wv
                total += wv
            }
            if total != 0.0 {
                for k in 0 ..< count { w[k] /= total }
            }
            bounds[xx] = (xmin, count)
            weightRows[xx] = w
        }

        let dstH = horizontal ? srcH : dstSize
        let dstW = horizontal ? dstSize : srcW
        var out = [Float](repeating: 0.0, count: dstH * dstW * c)

        if horizontal {
            // For each row y and each output column xx, blend source columns.
            for y in 0 ..< srcH {
                let rowBase = y * srcW * c
                for xx in 0 ..< dstW {
                    let (xmin, count) = bounds[xx]
                    let w = weightRows[xx]
                    let dstBase = (y * dstW + xx) * c
                    for ch in 0 ..< c {
                        var acc = 0.0
                        for k in 0 ..< count {
                            acc += w[k] * Double(src[rowBase + (xmin + k) * c + ch])
                        }
                        out[dstBase + ch] = Float(acc)
                    }
                }
            }
        } else {
            // For each output row yy and each column x, blend source rows.
            for yy in 0 ..< dstH {
                let (ymin, count) = bounds[yy]
                let w = weightRows[yy]
                for x in 0 ..< srcW {
                    let dstBase = (yy * srcW + x) * c
                    for ch in 0 ..< c {
                        var acc = 0.0
                        for k in 0 ..< count {
                            acc += w[k] * Double(src[((ymin + k) * srcW + x) * c + ch])
                        }
                        out[dstBase + ch] = Float(acc)
                    }
                }
            }
        }
        return (out, dstH, dstW)
    }

    /// Two-pass separable resize (horizontal then vertical) to `size x size`.
    /// `src` is row-major `srcH*srcW*c`. Output is `size*size*c`, values left unclamped here.
    private static func resize(
        _ src: [Float], srcH: Int, srcW: Int, channels c: Int, size: Int
    ) -> [Float] {
        let (h1, h1H, h1W) = resampleAxis(
            src, srcH: srcH, srcW: srcW, channels: c, dstSize: size, horizontal: true)
        let (v1, _, _) = resampleAxis(
            h1, srcH: h1H, srcW: h1W, channels: c, dstSize: size, horizontal: false)
        return v1
    }

    // MARK: - Straight-alpha PNG decode

    /// Decoded image: flat RGBA `[Float]` in [0,1], row-major `H*W*4`, plus whether alpha existed.
    private struct Decoded {
        var rgba: [Float]
        var width: Int
        var height: Int
        var hasAlpha: Bool
    }

    /// Load a PNG/CGImage and return STRAIGHT-alpha RGBA float in [0,1].
    /// Reads raw provider bytes (no CGContext draw, which would premultiply); honors
    /// `bytesPerRow`, `bitsPerPixel`, byte order, and `CGImageAlphaInfo`. If the source is
    /// premultiplied, un-premultiplies by dividing rgb by alpha where alpha > 0.
    private static func decodeStraightRGBA(_ path: String) -> Decoded? {
        let url = URL(fileURLWithPath: path)
        guard let srcRef = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(srcRef, 0, nil)
        else { return nil }

        let w = cg.width
        let h = cg.height
        let bpc = cg.bitsPerComponent
        let bpp = cg.bitsPerPixel
        let bytesPerRow = cg.bytesPerRow
        let alphaInfo = cg.alphaInfo
        let bitmapInfo = cg.bitmapInfo

        guard let provider = cg.dataProvider,
              let data = provider.data,
              let base = CFDataGetBytePtr(data)
        else { return nil }
        let dataLen = CFDataGetLength(data)

        // We support 8- and 16-bit-per-component integer formats. Floats are uncommon for PNG.
        let isFloat = bitmapInfo.contains(.floatComponents)
        let bytesPerComponent = bpc / 8
        guard !isFloat, bytesPerComponent == 1 || bytesPerComponent == 2 else {
            return nil
        }
        let componentsPerPixel = bpp / bpc
        guard componentsPerPixel >= 1, componentsPerPixel <= 4 else { return nil }

        let maxVal = Float((1 << bpc) - 1)

        // Byte order for 16-bit reads.
        let orderRaw = bitmapInfo.rawValue & CGBitmapInfo.byteOrderMask.rawValue
        let little16 = orderRaw == CGBitmapInfo.byteOrder16Little.rawValue
        let little32 = orderRaw == CGBitmapInfo.byteOrder32Little.rawValue
        let little = little16 || little32

        @inline(__always)
        func readComp(_ rowPtr: UnsafePointer<UInt8>, _ idx: Int) -> Float {
            if bytesPerComponent == 1 {
                return Float(rowPtr[idx]) / maxVal
            } else {
                let off = idx * 2
                let b0 = UInt16(rowPtr[off])
                let b1 = UInt16(rowPtr[off + 1])
                let v = little ? (b0 | (b1 << 8)) : ((b0 << 8) | b1)
                return Float(v) / maxVal
            }
        }

        // Determine channel layout.
        // Alpha-first (ARGB) vs alpha-last (RGBA); presence of alpha; gray vs RGB.
        let alphaFirst = alphaInfo == .premultipliedFirst
            || alphaInfo == .first
            || alphaInfo == .noneSkipFirst
        let hasAlpha = alphaInfo == .premultipliedFirst
            || alphaInfo == .premultipliedLast
            || alphaInfo == .first
            || alphaInfo == .last
        let premultiplied = alphaInfo == .premultipliedFirst
            || alphaInfo == .premultipliedLast
        let isGray = componentsPerPixel <= 2

        var out = [Float](repeating: 0.0, count: w * h * 4)

        // Bytes actually needed for the last pixel of a row (may be < bytesPerRow if padded).
        let rowBytesNeeded = w * componentsPerPixel * bytesPerComponent
        for y in 0 ..< h {
            let rowOffset = y * bytesPerRow
            // Defensive: bail rather than read out of bounds on a truncated buffer.
            if rowOffset + rowBytesNeeded > dataLen { break }
            let rowPtr = base.advanced(by: rowOffset)
            for x in 0 ..< w {
                let pix = x * componentsPerPixel
                var r: Float = 0, g: Float = 0, b: Float = 0, a: Float = 1.0

                if isGray {
                    // Gray or Gray+Alpha.
                    if alphaFirst && hasAlpha {
                        a = readComp(rowPtr, pix + 0)
                        let lum = readComp(rowPtr, pix + 1)
                        r = lum; g = lum; b = lum
                    } else {
                        let lum = readComp(rowPtr, pix + 0)
                        r = lum; g = lum; b = lum
                        if hasAlpha { a = readComp(rowPtr, pix + 1) }
                    }
                } else {
                    // RGB / RGBA / ARGB / RGBX / XRGB.
                    if alphaFirst {
                        // [A/X, R, G, B]
                        if hasAlpha { a = readComp(rowPtr, pix + 0) }
                        r = readComp(rowPtr, pix + 1)
                        g = readComp(rowPtr, pix + 2)
                        b = readComp(rowPtr, pix + 3)
                    } else {
                        // [R, G, B, A/X]
                        r = readComp(rowPtr, pix + 0)
                        g = readComp(rowPtr, pix + 1)
                        b = readComp(rowPtr, pix + 2)
                        if componentsPerPixel >= 4 && hasAlpha {
                            a = readComp(rowPtr, pix + 3)
                        }
                    }
                }

                // Un-premultiply to STRAIGHT alpha if needed.
                if premultiplied && hasAlpha {
                    if a > 0.0 {
                        let inv = 1.0 / a
                        r = min(r * inv, 1.0)
                        g = min(g * inv, 1.0)
                        b = min(b * inv, 1.0)
                    } else {
                        r = 0; g = 0; b = 0
                    }
                }

                let o = (y * w + x) * 4
                out[o + 0] = r
                out[o + 1] = g
                out[o + 2] = b
                out[o + 3] = a
            }
        }

        return Decoded(rgba: out, width: w, height: h, hasAlpha: hasAlpha)
    }

    // MARK: - Public-facing helpers used by the free functions below

    /// Core of `prepRGB`: returns flat `size*size*3` Float in [0,1].
    static func prepRGBFlat(_ path: String, _ size: Int) -> [Float] {
        guard let dec = decodeStraightRGBA(path) else {
            return [Float](repeating: 0.0, count: size * size * 3)
        }
        // Resize ALL 4 channels (incl. alpha) with PIL-faithful bicubic.
        let resized = resize(
            dec.rgba, srcH: dec.height, srcW: dec.width, channels: 4, size: size)

        var rgb = [Float](repeating: 0.0, count: size * size * 3)
        let n = size * size
        for i in 0 ..< n {
            let s = i * 4
            var r = resized[s + 0]
            var g = resized[s + 1]
            var b = resized[s + 2]
            if dec.hasAlpha {
                // Composite over white: rgb' = rgb*alpha + 1*(1-alpha).
                let a = min(max(resized[s + 3], 0.0), 1.0)
                r = r * a + (1.0 - a)
                g = g * a + (1.0 - a)
                b = b * a + (1.0 - a)
            }
            let d = i * 3
            rgb[d + 0] = min(max(r, 0.0), 1.0)
            rgb[d + 1] = min(max(g, 0.0), 1.0)
            rgb[d + 2] = min(max(b, 0.0), 1.0)
        }
        return rgb
    }
}

// MARK: - Public API

/// Load `path` as RGBA, resize to `size x size` (PIL bicubic), composite over white if it had
/// alpha, and return `[size, size, 3]` float32 in [0,1].
public func prepRGB(_ path: String, _ size: Int) -> MLXArray {
    let flat = ImageX.prepRGBFlat(path, size)
    return MLXArray(flat, [size, size, 3])
}

/// ImageNet normalization broadcast over the last (channel) dimension: `(img - mean) / std`.
public func imagenetNorm(_ img: MLXArray) -> MLXArray {
    let mean = MLXArray(ImageX.mean, [1, 1, 3])
    let std = MLXArray(ImageX.std, [1, 1, 3])
    return (img - mean) / std
}

/// Save `[H, W, 3]` float in [0,1] as an 8-bit RGB PNG at `path`.
/// Build a CGImage from an `[H,W,3]` float array in [0,1].
public func cgImageRGB(_ img: MLXArray) -> CGImage? {
    let shp = img.shape
    precondition(shp.count == 3 && shp[2] == 3, "cgImageRGB expects [H,W,3], got \(shp)")
    let h = shp[0], w = shp[1]
    let flat = img.asType(.float32).asArray(Float.self)   // row-major H*W*3
    var bytes = [UInt8](repeating: 255, count: h * w * 4)
    for i in 0 ..< (h * w) {
        let s = i * 3, d = i * 4
        for ch in 0 ..< 3 {
            let v = (flat[s + ch] * 255.0).rounded()
            bytes[d + ch] = UInt8(min(max(v, 0.0), 255.0))
        }
        bytes[d + 3] = 255
    }
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
    guard let provider = CGDataProvider(data: Data(bytes) as CFData) else { return nil }
    return CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                   space: colorSpace, bitmapInfo: bitmapInfo, provider: provider,
                   decode: nil, shouldInterpolate: false, intent: .defaultIntent)
}

public func saveRGB(_ img: MLXArray, _ path: String) {
    guard let cg = cgImageRGB(img) else { return }
    let url = URL(fileURLWithPath: path) as CFURL
    guard let dest = CGImageDestinationCreateWithURL(url, UTType.png.identifier as CFString, 1, nil)
    else { return }
    CGImageDestinationAddImage(dest, cg, nil)
    CGImageDestinationFinalize(dest)
}

/// Encode an `[H,W,3]` float image to PNG bytes (for returning textures without a temp file).
public func pngData(_ img: MLXArray) -> Data? {
    guard let cg = cgImageRGB(img) else { return nil }
    let data = NSMutableData()
    guard let dest = CGImageDestinationCreateWithData(data as CFMutableData,
                                                      UTType.png.identifier as CFString, 1, nil)
    else { return nil }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { return nil }
    return data as Data
}
