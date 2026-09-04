"""
Descarga masiva de fichas de atletas de la FIE usando su API REST interna.

Uso:
    python descargar_fie.py --start 1 --end 60000 --category S [--workers 5] [--output atletas_fie_2026.jsonl]

Vías para conseguir IDs:
  a) Barrido por rango: --start 1 --end 60000 (los IDs 404 se ignoran; se puede reanudar).
  b) Lista desde CSV/Excel del ranking (--ids-file ids.txt, un ID por linea).
  c) IDs concretos: --ids 41862,51359

El fichero .jsonl permite pausar y reanudar: los IDs ya presentes se omiten.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://fie.org/athletes",
    "Accept": "application/json, text/plain, */*",
    "DNT": "1",
}


def load_downloaded(output_file):
    ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "id" in data:
                        ids.add(data["id"])
                except json.JSONDecodeError:
                    continue
    return ids


def fetch_fencer(fencer_id, category, session):
    url = f"https://fie.org/api/fie/fencer/{fencer_id}?category={category}"
    for attempt in range(3):
        try:
            response = session.get(url, headers=HEADERS, timeout=15)
            if response.status_code == 200:
                return response.json()
            if response.status_code == 404:
                return None
            if response.status_code == 429:
                wait = (attempt + 1) * 5
                print(f"[!] Rate limit (429) en ID {fencer_id}. Esperando {wait}s...", flush=True)
                time.sleep(wait)
            else:
                return None
        except requests.exceptions.RequestException:
            time.sleep(2)
    return None


def build_target_ids(args, downloaded):
    ids = []
    if args.ids:
        for chunk in args.ids.split(","):
            chunk = chunk.strip()
            if chunk and chunk.isdigit():
                ids.append(int(chunk))
    elif args.ids_file:
        with open(args.ids_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.isdigit():
                    ids.append(int(line))
    else:
        ids = list(range(args.start, args.end + 1))
    return [i for i in ids if i not in downloaded]


def main():
    ap = argparse.ArgumentParser(description="Descarga masiva de fichas FIE")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=60000)
    ap.add_argument("--ids", type=str, default="", help="IDs concretos separados por coma")
    ap.add_argument("--ids-file", type=str, default="", help="Fichero con un ID por linea")
    ap.add_argument("--category", type=str, default="S", help="S=Senior, J=Junior, ...")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--output", type=str, default="atletas_fie_2026.jsonl")
    args = ap.parse_args()

    downloaded = load_downloaded(args.output)
    print(f"IDs ya descargados previamente: {len(downloaded)}")

    target_ids = build_target_ids(args, downloaded)
    print(f"Total de atletas pendientes por consultar: {len(target_ids)}")

    session = requests.Session()
    ok = 0
    with open(args.output, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_id = {
                executor.submit(fetch_fencer, fid, args.category, session): fid
                for fid in target_ids
            }
            for future in as_completed(future_to_id):
                fid = future_to_id[future]
                try:
                    result = future.result()
                    if result and "id" in result:
                        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f_out.flush()
                        ok += 1
                        print(f"[OK] {result.get('id')} - {result.get('name')} ({result.get('countryCode')})", flush=True)
                    else:
                        print(f"[404/-] {fid}", flush=True)
                except Exception as e:
                    print(f"[ERROR] Fallo al procesar ID {fid}: {e}", flush=True)

    print(f"Fin. Descargados {ok} atletas nuevos -> {args.output}")


if __name__ == "__main__":
    main()