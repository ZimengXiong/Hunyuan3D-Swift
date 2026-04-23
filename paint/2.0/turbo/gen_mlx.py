import sys
import argparse
import time
import trimesh
from pathlib import Path
from PIL import Image

# Add repo root to path
ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from hy3dgen.texgen import Hunyuan3DPaintPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="Path to input .obj or .glb")
    parser.add_argument("--image", default=str(ROOT / "images/penguin.png"), help="Path to reference image")
    parser.add_argument("--output", default="output_20_turbo_mlx.glb")
    parser.add_argument(
        "--mlx-weights",
        default=None,
        help="Optional local MLX weights directory (contains unet.npz). If omitted, auto-pulls from Hugging Face.",
    )
    args = parser.parse_args()

    print("Loading 2.0 TURBO with MLX backend...")
    painter = Hunyuan3DPaintPipeline.from_pretrained(
        "tencent/Hunyuan3D-2",
        subfolder="hunyuan3d-paint-v2-0-turbo",
        diffusion_backend="mlx",
        mlx_weights_path=args.mlx_weights,
    )

    mesh = trimesh.load(args.mesh, force="mesh")
    image = Image.open(args.image).convert("RGBA")

    print(f"Painting {args.mesh} using {args.image} (2.0 turbo MLX)...")
    t0 = time.time()
    textured = painter(mesh, image=image)
    textured.export(args.output)

    print(f"Done in {time.time()-t0:.1f}s. Saved to {args.output}")


if __name__ == "__main__":
    main()
