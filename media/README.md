# media/

Carpeta con una muestra aleatoria de **300 vídeos** de esgrima extraídos de `src/json/brutoVideos.json` (140.739 clips).

## Contenido

- `videos_part01.zip` … `videos_part07.zip` — 7 zips (ZIP_STORED, sin recompresión) con los 300 `.mp4`. Cada zip < 100 MB para cumplir el límite de GitHub.
- `manifest.json` — JSON con la ficha completa de cada vídeo (`gfycat_gfy_id`, `tournament_id`, `weapon`, `gender`, nombres, marcador, `touch`, `download_url`).

## Muestreo

- Muestreo aleatorio (semilla 42) sobre `brutoVideos.json`, repartido entre espada/florete/sable y ambos sexos.
- Fuente: `https://clips.fencingdatabase.com/<tournament>/<gfycat_gfy_id>.mp4`.
- ~10% de URLs devuelven 404; por eso se descargaron ~330 y se quedaron los 300 válidos.

## Cómo extraer

```bash
# PowerShell
Expand-Archive media\videos_part01.zip -DestinationPath media\extracted
```

Los zips contienen los `.mp4` con nombre `gfycat_gfy_id.mp4`. El script de descarga/zip está en `tools/download_videos.py` y `tools/zip_videos.py`.