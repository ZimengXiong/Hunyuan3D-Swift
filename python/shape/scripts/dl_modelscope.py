"""Resumable ModelScope download of Hunyuan3D-2mini safetensors.

ModelScope serves the LFS blobs (hf-mirror 308-redirects them to blocked
huggingface.co). Honors HTTPS_PROXY from env. Pure stdlib.
"""
import os
import sys
import time
import urllib.request

BASE = ("https://modelscope.cn/api/v1/models/Tencent-Hunyuan/"
        "Hunyuan3D-2mini/repo?Revision=master&FilePath=")
ROOT = os.environ.get("HY3D_WEIGHTS_ROOT", "weights/Hunyuan3D-2mini")

FILES = [
    ("hunyuan3d-vae-v2-mini/model.fp16.safetensors", 428_455_666),
    ("hunyuan3d-dit-v2-mini/model.fp16.safetensors", 3_819_958_234),
]

opener = urllib.request.build_opener(
    urllib.request.ProxyHandler(urllib.request.getproxies())
)


def download(path, approx):
    out = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for attempt in range(1, 61):
        have = os.path.getsize(out) if os.path.exists(out) else 0
        if have >= approx:
            print(f"  complete {path} ({have} bytes)", flush=True)
            return True
        req = urllib.request.Request(BASE + path)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            t0 = time.time()
            with opener.open(req, timeout=120) as r, open(out, "ab") as f:
                got = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if got % (64 << 20) < (1 << 20):
                        cur = have + got
                        mbps = got / 1e6 / max(1e-3, time.time() - t0)
                        print(f"  {path}: {cur/1e6:.0f} MB  ({mbps:.1f} MB/s)", flush=True)
            print(f"  segment done {path}: +{got/1e6:.0f} MB", flush=True)
        except urllib.error.HTTPError as e:
            if e.code == 416:  # range past EOF => file already complete
                print(f"  complete {path} (416, {os.path.getsize(out)} bytes)", flush=True)
                return True
            print(f"  attempt {attempt} {path}: HTTP {e.code}", flush=True)
            time.sleep(min(20, 2 ** (attempt % 6)))
        except Exception as e:  # noqa: BLE001
            print(f"  attempt {attempt} {path}: {type(e).__name__}: {str(e)[:100]}", flush=True)
            time.sleep(min(20, 2 ** (attempt % 6)))
    return os.path.getsize(out) if os.path.exists(out) else 0 >= approx - 4096


if __name__ == "__main__":
    ok = True
    for path, approx in FILES:
        print(f">>> {path}", flush=True)
        if not download(path, approx):
            ok = False
    print("DOWNLOADS_DONE" if ok else "DOWNLOADS_INCOMPLETE", flush=True)
    sys.exit(0 if ok else 1)
