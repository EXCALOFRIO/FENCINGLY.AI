# models/

Pesos de modelos descargados automáticamente. Ningún fichero supera 100 MB, por lo que **no hace falta Git LFS** (decidido tras comprobar tamaños: todos < 100 MB).

## Prioridad 1 — Pose 2D (compara frame a frame con YOLOv11-pose)

Zips de rtmlib (formato ONNX, listos para `rtmlib`). Van a `~/.cache/rtmlib/hub/checkpoints/` (ya copiados en esta máquina).

| Fichero | Modelo | Fuente |
|---|---|---|
| `rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip` | RTMO-M (detector+pose one-stage, body7, 640x640) | download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/ |
| `rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip` | RTMPose-M 256x192 (26 keypoints halpe) | download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/ |
| `yolox_m_8xb8-300e_humanart-c2c7a14a.zip` | YOLOX-M (detector humano) | download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/ |

Cada zip contiene `end2end.onnx` + metadatos `deploy.json`, `pipeline.json`, `detail.json`.

## Prioridad 2 — Pose 3D y profundidad

| Fichero | Modelo | Fuente |
|---|---|---|
| `motionbert/FT_MB_lite_MB_ft_h36m_global_lite.bin` | MotionBERT-Lite pose 3D desde cámara (H36M) | HuggingFace `walterzhu/MotionBERT` (renombrado desde `best_epoch.bin`; verificados `model_pos`) |
| `depth/depth_anything_v2_vits.pth` | Depth-Anything-V2-Small (perspectiva) | HuggingFace `depth-anything/Depth-Anything-V2-Small` (24,78M params, verificado) |

## Prioridad 3 — VideoMAEv2-Base (NO incluido)

El modelo base de VideoMAEv2 en HuggingFace está **restringido (gated, HTTP 401 sin token)**. No se pudo descargar sin credenciales. Si lo necesitas, acepta la licencia en https://huggingface.co/MCG-NJU/videomaev2-base y enlaza tu token:

```bash
huggingface-cli download MCG-NJU/videomaev2-base --local-dir models/videomaev2_base
```

## Notas

- No se incluyen SAM, CLIP, DINOv2, RAFT ni PANNs (no requeridos). El OCR (`rapidocr`, ya instalado) y `silero-vad` se instalan por pip.
- Script de descarga: `tools/download_models.py`.
- Los `.bin`/`.pth` están además en zip (`.zip`) para transporte si los necesitas fuera de git.