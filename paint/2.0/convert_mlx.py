import os
import sys
import argparse
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from hy3dgen.texgen.mlx.convert_weights import convert_and_save, PROFILE_PAINT_20

def main():
    parser = argparse.ArgumentParser(description="Convert 2.0 (non-turbo) paint weights to MLX")
    parser.add_argument("model_path", help="Path to the model subfolder (contains unet/ and vae/)")
    args = parser.parse_args()

    # Determine output directory
    output_dir = str(ROOT / "converted/Hunyuan3D-2.0-Paint-MLX/2.0")

    print("Converting 2.0 (non-turbo) paint weights to MLX...")
    print(f"Source: {args.model_path}")
    print(f"Target: {output_dir}")
    
    if not os.path.exists(args.model_path):
        print(f"Error: Path does not exist: {args.model_path}")
        sys.exit(1)

    convert_and_save(args.model_path, output_dir=output_dir, profile=PROFILE_PAINT_20)
    print(f"Done.")

if __name__ == "__main__":
    main()
