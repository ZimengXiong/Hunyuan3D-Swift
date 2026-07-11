"""Generic resumable ModelScope downloader. Reads exact size from the repo API.
Usage: python scripts/dl_any.py <repo> <filepath> <outpath>
"""
import os
import sys
import time
import json
import urllib.request

REPO, FILEPATH, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
API = f"https://modelscope.cn/api/v1/models/{REPO}/repo/files?Revision=master&Recursive=true"
BASE = f"https://modelscope.cn/api/v1/models/{REPO}/repo?Revision=master&FilePath={FILEPATH}"
opener = urllib.request.build_opener(urllib.request.ProxyHandler(urllib.request.getproxies()))

# exact size from API
size = None
with opener.open(API, timeout=60) as r:
    for f in json.load(r)["Data"]["Files"]:
        if f.get("Path") == FILEPATH:
            size = f["Size"]
print(f"target {FILEPATH}: {size} bytes ({size/1e9:.2f} GB)", flush=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)

for attempt in range(1, 81):
    have = os.path.getsize(OUT) if os.path.exists(OUT) else 0
    if have >= size:
        print(f"complete ({have} bytes)", flush=True)
        break
    req = urllib.request.Request(BASE)
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        t0 = time.time()
        with opener.open(req, timeout=120) as r, open(OUT, "ab") as fo:
            got = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fo.write(chunk); got += len(chunk)
                if got % (256 << 20) < (1 << 20):
                    print(f"  {(have+got)/1e9:.2f} GB ({got/1e6/max(1e-3,time.time()-t0):.1f} MB/s)", flush=True)
    except urllib.error.HTTPError as e:
        if e.code == 416:
            print("complete (416)", flush=True); break
        print(f"  attempt {attempt}: HTTP {e.code}", flush=True); time.sleep(min(20, 2 ** (attempt % 6)))
    except Exception as e:  # noqa: BLE001
        print(f"  attempt {attempt}: {type(e).__name__} {str(e)[:80]}", flush=True); time.sleep(min(20, 2 ** (attempt % 6)))
print("DONE" if (os.path.exists(OUT) and os.path.getsize(OUT) >= size) else "INCOMPLETE", flush=True)
