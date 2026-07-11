import os

import pytest

_MODELS = {
    "2mini": "weights/Hunyuan3D-2mini/hunyuan3d-dit-v2-mini",
    "2.0": "weights/Hunyuan3D-2/hunyuan3d-dit-v2-0",
    "2.1": "weights/Hunyuan3D-2.1/hunyuan3d-dit-v2-1",
    "mini-turbo": "weights/Hunyuan3D-2mini/hunyuan3d-dit-v2-mini-turbo",
}


def available_models():
    return {k: v for k, v in _MODELS.items()
            if os.path.exists(os.path.join(v, "model.fp16.safetensors"))}


@pytest.fixture(scope="session")
def models():
    m = available_models()
    if not m:
        pytest.skip("no model weights present")
    return m
