import json
import os
import zipfile

WORK = r"C:\Users\Alejandro\AppData\Local\Temp\opencode\media_work"
VIDS = os.path.join(WORK, "videos")
OUT_DIR = os.path.join(WORK, "zips")
os.makedirs(OUT_DIR, exist_ok=True)
TARGET_MB = 80

manifest = json.load(open(os.path.join(WORK, "manifest.json"), encoding="utf-8"))
print("manifest:", len(manifest))

parts = []
current = []
current_bytes = 0
for v in manifest:
    name = f"{v['gfycat_gfy_id']}.mp4"
    path = os.path.join(VIDS, name)
    size = os.path.getsize(path)
    if current and current_bytes + size > TARGET_MB * 1024 * 1024:
        parts.append(current)
        current = []
        current_bytes = 0
    current.append(name)
    current_bytes += size
if current:
    parts.append(current)

print("parts:", len(parts), [len(p) for p in parts])

zip_index = {}
for i, part in enumerate(parts, 1):
    zip_path = os.path.join(OUT_DIR, f"videos_part{i:02d}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        for name in part:
            z.write(os.path.join(VIDS, name), arcname=name)
    tot = sum(os.path.getsize(os.path.join(VIDS, n)) for n in part)
    print(i, zip_path, f"{os.path.getsize(zip_path)/1e6:.1f}MB (raw {tot/1e6:.1f}MB)")

info = {}
for i, part in enumerate(parts, 1):
    item = {"zip": f"videos_part{i:02d}.zip", "videos": []}
    for v in manifest:
        name = f"{v['gfycat_gfy_id']}.mp4"
        if name in part:
            item["videos"].append(v)
    info[i] = item

with open(os.path.join(WORK, "zips_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(info, f, indent=1, ensure_ascii=False)
print("zips done")