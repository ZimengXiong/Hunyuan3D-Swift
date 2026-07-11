"""RealESRGAN x4 (RRDBNet) in MLX — view super-resolution before texture baking.

Standard ESRGAN RRDBNet (num_feat=64, num_block=23, num_grow_ch=32, scale=4). NHWC; conv weights
NCHW->NHWC at load. Module names mirror the RealESRGAN_x4plus.pth checkpoint (conv_first, body.N.rdbM.convK,
conv_body, conv_up1/2, conv_hr, conv_last) so it loads 1:1. Input/output are [0,1] RGB.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def _lrelu(x):
    return mx.maximum(x, 0.2 * x)            # LeakyReLU(0.2)


def _up2x(x):                                # nearest-neighbor x2 (NHWC), matches F.interpolate('nearest')
    N, H, W, C = x.shape
    return mx.broadcast_to(x[:, :, None, :, None, :], (N, H, 2, W, 2, C)).reshape(N, H * 2, W * 2, C)


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)

    def __call__(self, x):
        x1 = _lrelu(self.conv1(x))
        x2 = _lrelu(self.conv2(mx.concatenate([x, x1], -1)))
        x3 = _lrelu(self.conv3(mx.concatenate([x, x1, x2], -1)))
        x4 = _lrelu(self.conv4(mx.concatenate([x, x1, x2, x3], -1)))
        x5 = self.conv5(mx.concatenate([x, x1, x2, x3, x4], -1))
        return x + x5 * 0.2


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def __call__(self, x):
        return x + 0.2 * self.rdb3(self.rdb2(self.rdb1(x)))


class RRDBNet(nn.Module):
    def __init__(self, num_block=23, nf=64, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        self.body = [RRDB(nf, gc) for _ in range(num_block)]
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)

    def __call__(self, x):                   # x: [N,H,W,3] in [0,1] -> [N,4H,4W,3]
        feat = self.conv_first(x)
        b = feat
        for blk in self.body:
            b = blk(b)
        feat = feat + self.conv_body(b)
        feat = _lrelu(self.conv_up1(_up2x(feat)))
        feat = _lrelu(self.conv_up2(_up2x(feat)))
        return self.conv_last(_lrelu(self.conv_hr(feat)))


def load_rrdbnet(npz_path="weights/realesrgan/rrdbnet.npz"):
    import numpy as np
    from mlx.utils import tree_unflatten
    m = RRDBNet()
    sd = dict(np.load(npz_path))
    flat = [(k, (mx.array(v).transpose(0, 2, 3, 1) if v.ndim == 4 else mx.array(v))) for k, v in sd.items()]
    m.update(tree_unflatten(flat)); mx.eval(m.parameters())
    return m


def upscale(model, img, tile=0):
    """img: [H,W,3] in [0,1] -> [4H,4W,3]. tile>0 splits into tiles to bound memory."""
    x = mx.array(img)[None]
    if tile <= 0:
        return mx.clip(model(x)[0], 0, 1)
    H, W = img.shape[:2]
    out = mx.zeros((H * 4, W * 4, 3))
    pad = 10
    for r0 in range(0, H, tile):
        for c0 in range(0, W, tile):
            r1, c1 = min(r0 + tile, H), min(c0 + tile, W)
            pr0, pc0 = max(r0 - pad, 0), max(c0 - pad, 0)
            pr1, pc1 = min(r1 + pad, H), min(c1 + pad, W)
            t = model(mx.array(img[pr0:pr1, pc0:pc1])[None])[0]
            out[r0 * 4:r1 * 4, c0 * 4:c1 * 4] = t[(r0 - pr0) * 4:(r0 - pr0) * 4 + (r1 - r0) * 4,
                                                  (c0 - pc0) * 4:(c0 - pc0) * 4 + (c1 - c0) * 4]
            mx.eval(out)
    return mx.clip(out, 0, 1)
