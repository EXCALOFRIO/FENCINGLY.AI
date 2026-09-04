import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

BASE = r"C:\Users\Alejandro\AppData\Local\Temp\opencode\models"
os.makedirs(BASE, exist_ok=True)

URLS = {
    "rtmlib_checkpoints/yolox_m_8xb8-300e_humanart-c2c7a14a.zip": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
    "rtmlib_checkpoints/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
    "rtmlib_checkpoints/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip",
    "motionbert/FT_MB_lite_MB_ft_h36m_global_lite.bin": "https://huggingface.co/walterzhu/MotionBERT/resolve/main/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin",
    "depth/depth_anything_v2_vits.pth": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true",
}


def download(key, url):
    dest = os.path.join(BASE, key)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("skip exists", key, flush=True)
        return
    for attempt in range(4):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                total = None
                n = 0
                while True:
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    n += len(chunk)
            os.replace(tmp, dest)
            print("OK", key, f"{n/1e6:.1f}MB", flush=True)
            return
        except Exception as e:
            print("RETRY", key, type(e).__name__, str(e)[:120], flush=True)
            time.sleep(5)
    print("FAIL", key, flush=True)


def main():
    with ThreadPoolExecutor(max_workers=5) as ex:
        for k, u in URLS.items():
            ex.submit(download, k, u)
    print("MODELS DONE", flush=True)


if __name__ == "__main__":
    main()