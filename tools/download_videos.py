import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen

WORK = r"C:\Users\Alejandro\AppData\Local\Temp\opencode\media_work"
OUT = os.path.join(WORK, "videos")
os.makedirs(OUT, exist_ok=True)

TARGET = 300
MAX_WORKERS = 24
FAILED = int(sys.argv[1]) if len(sys.argv) > 1 else 0

random.seed(42)

data = json.load(open(r"C:\Users\Alejandro\Desktop\FENCINGLY.AI\src\json\brutoVideos.json", encoding="utf-8"))

by_weapon = {}
for v in data:
    by_weapon.setdefault(v.get("weapon"), []).append(v)

for w in by_weapon:
    random.shuffle(by_weapon[w])

pool = []
for w, items in by_weapon.items():
    pool.extend(items)

if FAILED > 0:
    pool = pool[:-FAILED]

lock = threading.Lock()
success = 0
attempts = 0
manifest = []
downloaded = set()


def worker(v):
    global success
    url = v["download_url"]
    if success >= TARGET:
        return None
    name = f"{v['gfycat_gfy_id']}.mp4"
    dest = os.path.join(OUT, name)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return v
    last_err = None
    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                while True:
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
            return v
        except Exception as e:
            last_err = e
    with lock:
        global attempts
        attempts += 1
    return None


def main():
    global success
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(worker, v): v for v in pool}
        for fut in as_completed(futs):
            if success >= TARGET:
                break
            r = fut.result()
            if r is None:
                continue
            with lock:
                if success >= TARGET:
                    continue
                success += 1
                manifest.append(r)
                if success % 10 == 0:
                    print(success, len(manifest), flush=True)

    with open(os.path.join(WORK, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("DONE success=", len(manifest), "output=", OUT, flush=True)


if __name__ == "__main__":
    main()