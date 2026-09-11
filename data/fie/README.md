# data/fie/

Material de la FIE (federación internacional de esgrima) para construir el parser de atletas **sin depender de la red**.

## Páginas HTML guardadas (SSR, contienen los datos reales)

| Fichero | Contenido |
|---|---|
| `fencer_41862.html` | Ficha completa del atleta FAVARETTO Martina (482 KB) — biografía, medallas, mano, club, rankings |
| `ranking_foil_senior.html` | Ranking **Senior Women's Individual Foil** (2027) — 1 MB |
| `ranking_foil_senior_men.html` | Ranking **Senior Men's Individual Foil** (2027) — 1,3 MB |
| `athletes_search.html` | Página de búsqueda de atletas de `https://fie.org/athletes` |

## JSON de la API interna

`fencer_41862.json` — respuesta completa de `GET https://fie.org/api/fie/fencer/41862?category=S`.
Campos clave: `id, name, country, countryCode, weapon, gender, hand, date, rank, points, licenseNumber, image, biography, graceNoteBiography, ranking, medals, fencerRanks, availableWeapons, availableCategories, futureCompetitions ...`

Con esto se puede construir el parser de biografía completa (la API evita parsear HTML para datos estructurados).

## Muestra de datos descargados

`atletas_fie_sample.jsonl` — 281 atletas (IDs 41700–41980) en JSON Lines, descargados con el script. Sirve de ejemplo de salida y para reanudar/ampliar.

## Descarga masiva

Script: `tools/descargar_fie.py` (vía rápida con IDs del ranking o barrido por rango, ThreadPool 5 hilos, reanudable):

```bash
# Barrido completo de IDs históricos (ignora los 404, reanuda lo ya bajado)
python tools/descargar_fie.py --start 1 --end 60000 --category S --workers 5 --output atletas_fie_2026.jsonl

# O con IDs concretos desde un CSV/Excel del ranking de 2026
python tools/descargar_fie.py --ids-file ids_atletas_2026.txt --category S --output atletas_fie_2026.jsonl
```

Después: convertir a CSV con pandas (aplana `graceNoteBiography`, `ranking`, `medals`) — ejemplo en comentarios del script.

## Nota de reproducción

- `fie.org` es accesible desde esta red: se obtuvieron HTML y JSON reales (2026-09-05).
- Los IDs son enteros secuenciales; los inexistentes devuelven 404 y se omiten.
- Ritmo seguro ≈ 15-20 atletas/s con 5 hilos (se evita HTTP 429).