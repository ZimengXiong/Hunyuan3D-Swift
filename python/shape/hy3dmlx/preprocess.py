"""Image preprocessing — ImageProcessorV2 (recenter+resize) and the DINOv2 transform.

Mirrors reference preprocessors.py + the in-conditioner torchvision transform:
  ImageProcessorV2: RGBA -> recenter(border_ratio) -> resize(size) -> [-1,1] CHW tensor
  DINO transform:   [-1,1] -> [0,1] -> resize(518, bilinear) -> ImageNet-normalize -> NHWC
All on CPU/numpy (the irreducible host part); output feeds the MLX DINO.
"""
import cv2
import numpy as np
from PIL import Image

import mlx.core as mx

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def recenter(image: np.ndarray, border_ratio: float = 0.15):
    """image: HxWx[3|4] uint8. Returns recentered RGB uint8 [size,size,3] on white bg."""
    if image.shape[-1] == 4:
        mask = image[..., 3]
    else:
        mask = np.ones_like(image[..., 0]) * 255
        image = np.concatenate([image, mask[..., None]], axis=-1)
    H, W, C = image.shape
    size = max(H, W)
    result = np.zeros((size, size, C), dtype=np.uint8)
    coords = np.nonzero(mask)
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    h, w = x_max - x_min, y_max - y_min
    if h == 0 or w == 0:
        raise ValueError("input image is empty")
    desired = int(size * (1 - border_ratio))
    scale = desired / max(h, w)
    h2, w2 = int(h * scale), int(w * scale)
    x2 = (size - h2) // 2
    y2 = (size - w2) // 2
    result[x2:x2 + h2, y2:y2 + w2] = cv2.resize(image[x_min:x_max, y_min:y_max], (w2, h2),
                                                interpolation=cv2.INTER_AREA)
    bg = np.ones((size, size, 3), dtype=np.uint8) * 255
    m = result[..., 3:].astype(np.float32) / 255
    out = result[..., :3] * m + bg * (1 - m)
    return out.clip(0, 255).astype(np.uint8)


def load_image(path_or_pil, size: int = 512, border_ratio: float = 0.15) -> np.ndarray:
    """Returns [1, 3, size, size] float32 in [-1, 1] (RGB), matching array_to_tensor."""
    if isinstance(path_or_pil, str):
        img = Image.open(path_or_pil)
    else:
        img = path_or_pil
    img = img.convert("RGBA")
    arr = np.asarray(img)
    arr = recenter(arr, border_ratio)
    arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_CUBIC)
    t = arr.astype(np.float32) / 255 * 2 - 1  # [-1,1]
    t = np.transpose(t, (2, 0, 1))[None]  # [1,3,H,W]
    return t


def dino_transform(img_chw_m1p1: np.ndarray, image_size: int = 518) -> mx.array:
    """[3, H, W] in [-1,1] -> NHWC [1, 518, 518, 3] ImageNet-normalized (mx.array float32)."""
    img01 = (img_chw_m1p1 + 1.0) / 2.0          # [0,1]
    hwc = np.transpose(img01, (1, 2, 0))         # HWC
    resized = cv2.resize(hwc, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    norm = (resized - IMAGENET_MEAN) / IMAGENET_STD
    return mx.array(norm[None].astype(np.float32))  # NHWC
