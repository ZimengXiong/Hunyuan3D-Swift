import Foundation
import CoreGraphics
import ImageIO
import MLX

/// Image preprocessing for the DINOv2 encoder. The PNG decode is the only host step (no MLX image
/// codec exists — the Python reference uses PIL/cv2 here too); everything after it — bilinear
/// resize, alpha-recenter, white composite, ImageNet normalize — runs as MLX tensor ops.
/// Mirrors hy3dmlx/preprocess.py: alpha-recenter with border -> composite on white -> normalize.
public enum Preprocess {
    /// decode a file -> straight-alpha RGBA floats [0,1], top-left row-major (host: ImageIO)
    static func decode(_ path: String) -> (rgba: [Float], w: Int, h: Int)? {
        guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
        return decode(img)
    }

    /// decode a CGImage (e.g. from the photo picker) -> straight-alpha RGBA floats [0,1]
    static func decode(_ img: CGImage) -> (rgba: [Float], w: Int, h: Int)? {
        let w = img.width, h = img.height
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = CGContext(data: &bytes, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))   // bitmap rows are top-left
        var out = [Float](repeating: 0, count: w * h * 4)
        for i in 0 ..< w * h {
            let a = Float(bytes[i * 4 + 3]) / 255
            let inv: Float = a > 0 ? 1 / a : 0                        // un-premultiply -> straight
            out[i * 4] = Float(bytes[i * 4]) / 255 * inv
            out[i * 4 + 1] = Float(bytes[i * 4 + 1]) / 255 * inv
            out[i * 4 + 2] = Float(bytes[i * 4 + 2]) / 255 * inv
            out[i * 4 + 3] = a
        }
        return (out, w, h)
    }

    /// image file -> ImageNet-normalized NHWC pixels [1, size, size, 3] for DINOv2.
    public static func dinoPixels(_ path: String, size: Int = 518, border: Float = 0.15) -> MLXArray? {
        guard let (rgba, w, h) = decode(path) else { return nil }
        return pixels(rgba: rgba, w: w, h: h, size: size, border: border)
    }

    /// CGImage (e.g. photo picker) -> ImageNet-normalized NHWC pixels [1, size, size, 3].
    public static func dinoPixels(cgImage img: CGImage, size: Int = 518, border: Float = 0.15) -> MLXArray? {
        guard let (rgba, w, h) = decode(img) else { return nil }
        return pixels(rgba: rgba, w: w, h: h, size: size, border: border)
    }

    static func pixels(rgba: [Float], w: Int, h: Int, size: Int, border: Float) -> MLXArray {
        // alpha bounding box (4 scalars off the decoded image; the heavy work below is all MLX)
        var xmin = w, xmax = -1, ymin = h, ymax = -1
        for y in 0 ..< h {
            for x in 0 ..< w where rgba[(y * w + x) * 4 + 3] > 0.5 {
                xmin = min(xmin, x); xmax = max(xmax, x); ymin = min(ymin, y); ymax = max(ymax, y)
            }
        }
        if xmax < 0 { xmin = 0; xmax = w - 1; ymin = 0; ymax = h - 1 }
        let bw = xmax - xmin + 1, bh = ymax - ymin + 1
        let scale = Float(size) * (1 - border) / Float(max(bw, bh))
        let ow = Int(Float(bw) * scale), oh = Int(Float(bh) * scale)
        let ox = (size - ow) / 2, oy = (size - oh) / 2

        // 1D per-axis sample coords + window validity (size-length; the only non-MLX setup)
        var fxH = [Float](repeating: 0, count: size), fyH = [Float](repeating: 0, count: size)
        var vxH = [Float](repeating: 0, count: size), vyH = [Float](repeating: 0, count: size)
        for p in 0 ..< size {
            fxH[p] = Float(xmin) + (Float(p - ox) + 0.5) / scale - 0.5
            fyH[p] = Float(ymin) + (Float(p - oy) + 0.5) / scale - 0.5
            vxH[p] = (p >= ox && p < ox + ow) ? 1 : 0
            vyH[p] = (p >= oy && p < oy + oh) ? 1 : 0
        }

        // ---- everything below is MLX ----
        let img = MLXArray(rgba, [h, w, 4]).asType(.float32)         // one host->device upload
        let fx = MLXArray(fxH), fy = MLXArray(fyH)
        let x0 = clip(floor(fx), min: 0, max: Float(w - 1)).asType(.int32)
        let x1 = clip(floor(fx) + 1, min: 0, max: Float(w - 1)).asType(.int32)
        let y0 = clip(floor(fy), min: 0, max: Float(h - 1)).asType(.int32)
        let y1 = clip(floor(fy) + 1, min: 0, max: Float(h - 1)).asType(.int32)
        let tx = (fx - floor(fx)).reshaped([1, size, 1])             // per output column
        let ty = (fy - floor(fy)).reshaped([size, 1, 1])             // per output row

        let r0 = take(img, y0, axis: 0), r1 = take(img, y1, axis: 0) // [size, w, 4]
        let p00 = take(r0, x0, axis: 1), p01 = take(r0, x1, axis: 1) // [size, size, 4]
        let p10 = take(r1, x0, axis: 1), p11 = take(r1, x1, axis: 1)
        let top = p00 * (1 - tx) + p01 * tx
        let bot = p10 * (1 - tx) + p11 * tx
        let samp = top * (1 - ty) + bot * ty                         // bilinear [size, size, 4]

        let parts = split(samp, parts: 4, axis: -1)                  // r,g,b,a
        let rgbS = concatenated([parts[0], parts[1], parts[2]], axis: -1)
        let a = parts[3]
        let comp = rgbS * a + (1 - a)                                // composite on white
        let valid = MLXArray(vyH).reshaped([size, 1, 1]) * MLXArray(vxH).reshaped([1, size, 1])
        let windowed = comp * valid + (1 - valid)                    // white outside the object window
        let mean = MLXArray([Float(0.485), 0.456, 0.406])
        let std = MLXArray([Float(0.229), 0.224, 0.225])
        return ((windowed - mean) / std).reshaped([1, size, size, 3])
    }
}
