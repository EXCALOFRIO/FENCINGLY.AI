# media/

Vídeos de esgrima descargados de `https://clips.fencingdatabase.com/<tournament>/<gfycat_gfy_id>.mp4`, separados **por arma** (campos `weapon` verificados en `manifest.json` de cada carpeta). Zips `ZIP_STORED` (sin recompresión, cada uno < 100 MB).

## Estructura

| Carpeta | Contenido | Zips |
|---|---|---|
| `epee/` | 300 clips aleatorios de **espada** | `epee_part01.zip` … `epee_part07.zip` |
| `foil/` | 300 clips aleatorios de **florete** | `foil_part01.zip` … `foil_part09.zip` |
| `sabre/` | 300 clips aleatorios de **sable** | `sabre_part01.zip` … `sabre_part08.zip` |
| `foil_Alice_Volpi/` | 100 clips de un **mismo tirador de florete** (Alice Volpi, 1216 clips en el JSON) | `videos_part01.zip` … `videos_part04.zip` |
| `fencer_Yannick_Borel/` | 100 clips de un mismo tirador de **espada** (Yannick Borel) | `videos_part01.zip` … `videos_part03.zip` |

Cada carpeta incluye `manifest.json` con la ficha completa de cada clip (`gfycat_gfy_id`, `tournament_id`, `weapon`, `gender`, rivales, marcador, `touch`, `download_url`).

## Muestreo

- Aleatorio (semilla 42) sobre `brutoVideos.json` (140.739 clips), mezclando toques (izquierda/derecha/doble) y torneos.
- ~10% de URLs dan 404; los scripts descargan de más hasta completar el número objetivo.

## Cómo extraer y regenerar

```bash
Expand-Archive media\epee\epee_part01.zip -DestinationPath media\epee\extracted
```

Regenerar por arma:
```bash
python tools/download_weapon.py --weapon foil --count 300 --out media/foil
python tools/download_fencer.py --name "Alice Volpi" --count 100 --out media/foil_Alice_Volpi
```