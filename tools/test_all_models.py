from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
OUT = ROOT / "outputs" / "validation"
SHAPE_DIR = OUT / "shape"
PAINT_DIR = OUT / "paint"
RENDER_DIR = OUT / "renders"
LOG_DIR = OUT / "logs"

for d in [OUT, SHAPE_DIR, PAINT_DIR, RENDER_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class RunResult:
    name: str
    cmd: List[str]
    output_path: str
    ok: bool
    seconds: float
    log_path: str
    error: Optional[str] = None


SHAPE_CASES = [
    {"name": "sv_mini", "preset": "mini", "selector": []},
    {"name": "sv_mini_turbo", "preset": "mini-turbo", "selector": []},
    {"name": "sv_20", "preset": "2.0", "selector": []},
    {"name": "sv_20_turbo", "preset": "2.0-turbo", "selector": []},
    {"name": "sv_21", "preset": "2.1", "selector": []},
    {"name": "mv", "preset": "mv", "selector": ["mv", "1"]},
    {"name": "mv_turbo", "preset": "mv-turbo", "selector": ["mv", "1"]},
]

PAINT_CASES = [
    {"name": "paint_20", "preset": "2.0"},
    {"name": "paint_20_turbo", "preset": "2.0-turbo"},
    {"name": "paint_21", "preset": "2.1"},
]


def run_cmd(name: str, cmd: List[str], log_path: Path, timeout: int = 7200) -> RunResult:
    env = os.environ.copy()
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    t0 = time.time()
    p = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    dt = time.time() - t0
    log_path.write_text(p.stdout)
    ok = p.returncode == 0
    err = None if ok else f"exit={p.returncode}"
    return RunResult(name=name, cmd=cmd, output_path="", ok=ok, seconds=dt, log_path=str(log_path), error=err)


def mesh_stats(path: Path) -> Dict:
    mesh = trimesh.load(path, force="mesh")
    bounds = mesh.bounds
    extents = (bounds[1] - bounds[0]).tolist()
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "area": float(mesh.area),
        "extents": [float(x) for x in extents],
        "is_watertight": bool(mesh.is_watertight),
    }


def sample_points_normed(path: Path, n: int = 3000) -> np.ndarray:
    mesh = trimesh.load(path, force="mesh")
    pts = mesh.sample(n).astype(np.float32)
    center = pts.mean(axis=0, keepdims=True)
    pts = pts - center
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    if diag > 1e-8:
        pts = pts / diag
    return pts


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    ta = torch.from_numpy(a)
    tb = torch.from_numpy(b)
    d = torch.cdist(ta, tb)
    return float((d.min(dim=1).values.mean() + d.min(dim=0).values.mean()) * 0.5)


def texture_image(path: Path) -> np.ndarray:
    mesh = trimesh.load(path, force="mesh")
    mat = getattr(mesh.visual, "material", None)
    img = None
    if mat is not None and hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None:
        img = mat.baseColorTexture
    elif mat is not None and hasattr(mat, "image") and mat.image is not None:
        img = mat.image
    if img is None:
        raise RuntimeError(f"No texture image found in {path}")
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return arr


def compare_textures(a: np.ndarray, b: np.ndarray) -> Dict:
    if a.shape != b.shape:
        b = np.array(Image.fromarray((b * 255).astype(np.uint8)).resize((a.shape[1], a.shape[0]), Image.BILINEAR), dtype=np.float32) / 255.0
    diff = a - b
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    psnr = 99.0 if mse == 0 else float(10.0 * math.log10(1.0 / mse))
    return {"mae": mae, "mse": mse, "psnr_db": psnr}


def render_contact_sheet(path: Path, out_png: Path, textured: bool) -> None:
    mesh = trimesh.load(path, force="mesh")
    renderer = MeshRender(default_resolution=512, texture_size=1024, raster_mode="auto", device="auto")
    renderer.load_mesh(mesh)
    views = [(10, 0), (10, 120), (10, 240)]
    imgs = []
    for elev, azim in views:
        if textured:
            img = renderer.render(elev=elev, azim=azim, keep_alpha=False, bgcolor=[1, 1, 1], return_type="np")
        else:
            img = renderer.render_normal(elev=elev, azim=azim, return_type="np")
        arr = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        imgs.append(arr)
    sheet = np.concatenate(imgs, axis=1)
    Image.fromarray(sheet).save(out_png)


def main() -> None:
    summary = {
        "shape_runs": [],
        "paint_runs": [],
        "shape_stats": {},
        "shape_chamfer": {},
        "paint_texture_compare": {},
        "renders": {},
    }

    print("[1/4] Running shape models...")
    for case in SHAPE_CASES:
        out_path = SHAPE_DIR / f"{case['name']}.glb"
        log_path = LOG_DIR / f"shape_{case['name']}.log"
        cmd = [
            "uv", "run", "python", "main.py", "--seed", "12345", "shape",
            "--shape-preset", case["preset"],
            "--output", str(out_path),
            *case["selector"],
        ]
        print(" ", " ".join(cmd))
        rr = run_cmd(case["name"], cmd, log_path)
        rr.output_path = str(out_path)
        summary["shape_runs"].append(asdict(rr))

    print("[2/4] Running paint models...")
    base_mesh = SHAPE_DIR / "sv_20_turbo.glb"
    for case in PAINT_CASES:
        out_path = PAINT_DIR / f"{case['name']}.glb"
        log_path = LOG_DIR / f"paint_{case['name']}.log"
        cmd = [
            "uv", "run", "python", "main.py", "--seed", "12345", "paint",
            "--paint-preset", case["preset"],
            "--mesh", str(base_mesh),
            "--output", str(out_path),
        ]
        print(" ", " ".join(cmd))
        rr = run_cmd(case["name"], cmd, log_path)
        rr.output_path = str(out_path)
        summary["paint_runs"].append(asdict(rr))

    print("[3/4] Comparing outputs...")
    ok_shape = {x["name"]: Path(x["output_path"]) for x in summary["shape_runs"] if x["ok"]}
    for name, path in ok_shape.items():
        summary["shape_stats"][name] = mesh_stats(path)

    if "sv_20_turbo" in ok_shape:
        ref = sample_points_normed(ok_shape["sv_20_turbo"])
        for name in ["sv_mini", "sv_mini_turbo", "sv_20", "sv_21"]:
            if name in ok_shape:
                summary["shape_chamfer"][f"{name}_vs_sv_20_turbo"] = chamfer_distance(ref, sample_points_normed(ok_shape[name]))

    if "mv_turbo" in ok_shape and "mv" in ok_shape:
        summary["shape_chamfer"]["mv_vs_mv_turbo"] = chamfer_distance(
            sample_points_normed(ok_shape["mv_turbo"]), sample_points_normed(ok_shape["mv"])
        )

    ok_paint = {x["name"]: Path(x["output_path"]) for x in summary["paint_runs"] if x["ok"]}
    if "paint_20_turbo" in ok_paint:
        ref_tex = texture_image(ok_paint["paint_20_turbo"])
        for name in ["paint_20", "paint_21"]:
            if name in ok_paint:
                summary["paint_texture_compare"][f"{name}_vs_paint_20_turbo"] = compare_textures(ref_tex, texture_image(ok_paint[name]))

    print("[4/4] Rendering preview sheets...")
    for name, path in ok_shape.items():
        png = RENDER_DIR / f"shape_{name}.png"
        render_contact_sheet(path, png, textured=False)
        summary["renders"][f"shape_{name}"] = str(png)

    for name, path in ok_paint.items():
        png = RENDER_DIR / f"paint_{name}.png"
        render_contact_sheet(path, png, textured=True)
        summary["renders"][f"paint_{name}"] = str(png)

    summary_path = OUT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    md = ["# Validation Summary", "", "## Shape runs"]
    for r in summary["shape_runs"]:
        md.append(f"- {r['name']}: {'OK' if r['ok'] else 'FAIL'} ({r['seconds']:.1f}s) -> `{r['output_path']}`")
    md.append("")
    md.append("## Paint runs")
    for r in summary["paint_runs"]:
        md.append(f"- {r['name']}: {'OK' if r['ok'] else 'FAIL'} ({r['seconds']:.1f}s) -> `{r['output_path']}`")
    md.append("")
    md.append("## Shape comparability (Chamfer, lower is closer)")
    for k, v in summary["shape_chamfer"].items():
        md.append(f"- {k}: {v:.6f}")
    md.append("")
    md.append("## Paint texture comparability (vs 2.0-turbo)")
    for k, v in summary["paint_texture_compare"].items():
        md.append(f"- {k}: MAE={v['mae']:.6f}, PSNR={v['psnr_db']:.2f} dB")
    md.append("")
    md.append("## Render previews")
    for k, v in summary["renders"].items():
        md.append(f"- {k}: `{v}`")

    (OUT / "summary.md").write_text("\n".join(md) + "\n")

    print(f"Done. Wrote {summary_path} and {OUT / 'summary.md'}")


if __name__ == "__main__":
    main()
