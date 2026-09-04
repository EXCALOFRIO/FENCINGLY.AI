# fencer_Yannick_Borel/

100 vídeos aleatorios del mismo tirador: **Yannick Borel** (espadista francés, flèche, diestro, 1636 clips en `brutoVideos.json`).

- Muestra pseudoaleatoria (semilla 42) de sus 1636 apariciones, mezclando toques: izquierda/derecha/doble (~2/2/1), 24 torneos distintos.
- 3 zips (ZIP_STORED, <100 MB): `videos_part01.zip` (82,1 MB), `videos_part02.zip` (82,1 MB), `videos_part03.zip` (12,5 MB).
- `manifest.json`: ficha de cada clip (torneo, rivales, marcador, `touch`, `download_url`).

Regenerar o cambiar de tirador: `python tools/download_fencer.py --name "NOMBRE" --count 100 --out media/fencer_NOMBRE`.