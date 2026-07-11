"""DINOv2-giant conditioner (ViT + SwiGLU FFN), MLX port of HF Dinov2Model.

Config (2mini): hidden 1536, 24 heads, 40 layers, patch 14, img 518 -> 1370
tokens (1369 patches + CLS), use_swiglu_ffn=True, LN eps 1e-6.
Attribute names mirror HF so `conditioner.main_image_encoder.model.*` keys load 1:1.
Input is NHWC pixel values already resized/cropped/ImageNet-normalized.
"""
import mlx.core as mx
import mlx.nn as nn


def sdpa(q, k, v):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5)


def gelu_erf(x):
    return nn.gelu(x)


class LayerNorm(nn.Module):
    """fp32-internal LayerNorm."""

    def __init__(self, dim, eps=1e-6, affine=True):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.bias = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x):
        d = x.dtype
        xf = x.astype(mx.float32)
        mu = xf.mean(-1, keepdims=True)
        var = xf.var(-1, keepdims=True)
        return ((xf - mu) * mx.rsqrt(var + self.eps) * self.weight.astype(mx.float32)
                + self.bias.astype(mx.float32)).astype(d)


def _heads(x: mx.array, heads: int) -> mx.array:
    # [B, N, H*D] -> [B, H, N, D]
    B, N, HD = x.shape
    return x.reshape(B, N, heads, HD // heads).transpose(0, 2, 1, 3)


class Dinov2SelfAttention(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.num_heads = heads
        self.query = nn.Linear(hidden, hidden, bias=True)
        self.key = nn.Linear(hidden, hidden, bias=True)
        self.value = nn.Linear(hidden, hidden, bias=True)

    def __call__(self, x):
        B, N, _ = x.shape
        q = _heads(self.query(x), self.num_heads)
        k = _heads(self.key(x), self.num_heads)
        v = _heads(self.value(x), self.num_heads)
        out = sdpa(q, k, v).transpose(0, 2, 1, 3).reshape(B, N, -1)
        return out


class _SelfOutput(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.dense = nn.Linear(hidden, hidden, bias=True)

    def __call__(self, x):
        return self.dense(x)


class Dinov2Attention(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.attention = Dinov2SelfAttention(hidden, heads)
        self.output = _SelfOutput(hidden)

    def __call__(self, x):
        return self.output(self.attention(x))


class LayerScale(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.lambda1 = mx.ones((hidden,))

    def __call__(self, x):
        return x * self.lambda1


class Dinov2SwiGLUFFN(nn.Module):
    def __init__(self, hidden: int, hidden_features: int):
        super().__init__()
        self.weights_in = nn.Linear(hidden, 2 * hidden_features, bias=True)
        self.weights_out = nn.Linear(hidden_features, hidden, bias=True)

    def __call__(self, x):
        h = self.weights_in(x)
        x1, x2 = mx.split(h, 2, axis=-1)
        return self.weights_out(nn.silu(x1) * x2)


class Dinov2MLP(nn.Module):
    """Standard FFN for use_swiglu_ffn=False (DINOv2-large): fc1 -> gelu -> fc2."""

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, intermediate, bias=True)
        self.fc2 = nn.Linear(intermediate, hidden, bias=True)

    def __call__(self, x):
        return self.fc2(gelu_erf(self.fc1(x)))


class Dinov2Layer(nn.Module):
    def __init__(self, hidden: int, heads: int, ffn_dim: int, swiglu: bool = True, eps: float = 1e-6):
        super().__init__()
        self.norm1 = LayerNorm(hidden, eps=eps, affine=True)
        self.attention = Dinov2Attention(hidden, heads)
        self.layer_scale1 = LayerScale(hidden)
        self.norm2 = LayerNorm(hidden, eps=eps, affine=True)
        self.mlp = Dinov2SwiGLUFFN(hidden, ffn_dim) if swiglu else Dinov2MLP(hidden, ffn_dim)
        self.layer_scale2 = LayerScale(hidden)

    def __call__(self, x):
        x = x + self.layer_scale1(self.attention(self.norm1(x)))
        x = x + self.layer_scale2(self.mlp(self.norm2(x)))
        return x


class _Encoder(nn.Module):
    def __init__(self, hidden, heads, ffn_dim, layers, swiglu, eps):
        super().__init__()
        self.layer = [Dinov2Layer(hidden, heads, ffn_dim, swiglu, eps) for _ in range(layers)]


class Dinov2Embeddings(nn.Module):
    def __init__(self, hidden: int, patch: int = 14, num_tokens: int = 1370):
        super().__init__()
        self.patch_embeddings = _Patch(hidden, patch)
        self.cls_token = mx.zeros((1, 1, hidden))
        self.mask_token = mx.zeros((1, hidden))  # unused at inference, present in ckpt
        self.position_embeddings = mx.zeros((1, num_tokens, hidden))

    def __call__(self, pixel_nhwc: mx.array) -> mx.array:
        p = self.patch_embeddings(pixel_nhwc)         # [B, gh, gw, hidden]
        B = p.shape[0]
        p = p.reshape(B, -1, p.shape[-1])             # [B, 1369, hidden] (row-major h*W+w)
        cls = mx.broadcast_to(self.cls_token, (B, 1, p.shape[-1]))
        x = mx.concatenate([cls, p], axis=1)          # [B, 1370, hidden]
        return x + self.position_embeddings


class _Patch(nn.Module):
    def __init__(self, hidden: int, patch: int):
        super().__init__()
        self.projection = nn.Conv2d(3, hidden, kernel_size=patch, stride=patch)

    def __call__(self, pixel_nhwc):
        return self.projection(pixel_nhwc)


class Dinov2Model(nn.Module):
    def __init__(self, hidden=1536, heads=24, layers=40, ffn_dim=4096, swiglu=True,
                 patch=14, num_tokens=1370, eps=1e-6):
        super().__init__()
        self.embeddings = Dinov2Embeddings(hidden, patch, num_tokens)
        self.encoder = _Encoder(hidden, heads, ffn_dim, layers, swiglu, eps)
        self.layernorm = LayerNorm(hidden, eps=eps, affine=True)

    def __call__(self, pixel_nhwc: mx.array) -> mx.array:
        x = self.embeddings(pixel_nhwc)
        for layer in self.encoder.layer:
            x = layer(x)
        return self.layernorm(x)


class DinoImageEncoder(nn.Module):
    """Wraps Dinov2Model as `.model` to match the `...main_image_encoder.model.*` keys."""

    def __init__(self, hidden=1536, heads=24, layers=40, ffn_dim=4096, swiglu=True,
                 patch=14, num_tokens=1370, eps=1e-6):
        super().__init__()
        self.model = Dinov2Model(hidden, heads, layers, ffn_dim, swiglu, patch, num_tokens, eps)
        self.num_patches = num_tokens
        self.hidden = hidden

    def __call__(self, pixel_nhwc: mx.array) -> mx.array:
        return self.model(pixel_nhwc)  # last_hidden_state [B, 1370, hidden]

    def unconditional_embedding(self, batch_size: int, dtype=mx.float32) -> mx.array:
        return mx.zeros((batch_size, self.num_patches, self.hidden), dtype=dtype)
