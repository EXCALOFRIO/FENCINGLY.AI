"""Descarga y empaqueta N videos aleatorios de una misma arma (epee/foil/sabre).

Uso:
    python tools/download_weapon.py --weapon foil --count 300 --out media/foil
"""
import argparse
import json
import os
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen

REPO = r"C:\Users\Alejandro\Desktop\FENCINGLY.AI"
DATA = os.path.join(REPO, "src", "json", "brutoVideos.json")
WORK = r"C:\Users\Alejandro\AppData\Local\Temp\opencode\weapon_work"
TARGET_MB = 80


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weapon", required=True, choices=["epee", "foil", "sabre"])
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.load(open(DATA, encoding="utf-8"))
    clips = [v for v in data if v.get("weapon") == args.weapon]
    random.seed(args.seed)
    random.shuffle(clips)
    print(f"Clips {args.weapon}: {len(clips)}")

    dl = os.path.join(WORK, args.weapon)
    os.makedirs(dl, exist_ok=True)

    def worker(v):
        dest = os.path.join(dl, f"{v['gfycat_gfy_id']}.mp4")
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            return v
        for _ in range(3):
            try:
                req = Request(v["download_url"], headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                    while True:
                        chunk = r.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
                return v
            except Exception:
                continue
        return None

    ok = []
    i = 0
    while len(ok) < args.count and i < len(clips):
        batch = clips[i : i + args.workers * 4]
        i += len(batch)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(worker, v) for v in batch]
            for fut in as_completed(futs):
                r = fut.result()
                if r and len(ok) < args.count:
                    ok.append(r)
    ok = ok[: args.count]
    print(f"Descargados {args.weapon}: {len(ok)}")

    outabs = os.path.join(REPO, args.out)
    os.makedirs(outabs, exist_ok=True)
    parts, current, current_bytes = [], [], 0
    for v in ok:
        name = f"{v['gfycat_gfy_id']}.mp4"
        size = os.path.getsize(os.path.join(dl, name))
        if current and current_bytes + size > TARGET_MB * 1024 * 1024:
            parts.append(current)
            current, current_bytes = [], 0
        current.append(name)
        current_bytes += size
    if current:
        parts.append(current)

    for n, part in enumerate(parts, 1):
        zip_path = os.path.join(outabs, f"{args.weapon}_part{n:02d}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
            for name in part:
                z.write(os.path.join(dl, name), arcname=name)
        print(" ", zip_path.rsplit("\\", 1)[-1], f"{os.path.getsize(zip_path)/1e6:.1f}MB")

    with open(os.path.join(outabs, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(ok, f, ensure_ascii=False, indent=1)
    print("LISTO:", outabs)


if __name__ == "__main__":
    main()