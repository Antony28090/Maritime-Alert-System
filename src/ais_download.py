"""
Download NOAA / BOEM MarineCadastre daily AIS files (2024 GeoParquet release).

Source:  https://ocmgeodatastor1.blob.core.windows.net/marinecadastre/ais2024/
Licence: CC0 1.0 Universal (stated in the dataset readme at
         github.com/ocm-marinecadastre/ais-vessel-traffic,
         data/ais-broadcast-points-2024-readme.md).
Citation: Martin, D. R., J. Brass, M. Dornback and J. Fontenault (2025).
         Nationwide Automatic Identification System 2024. NOAA Office for
         Coastal Management.  Acknowledgement: U.S. Coast Guard Navigation Center.

Each daily file is ~300 MB.  The server throttles single connections to a few
hundred kB/s but scales with parallel range requests, so each file is fetched
as N_CONN concurrent byte ranges and assembled on disk.  A file is kept only
if every range succeeded; files that already exist with the right size are
skipped, so the script is safe to re-run after a network outage.

    python -m src.ais_download --start 2024-07-01 --end 2024-07-31 --out data/real_ais/raw
"""

import argparse
import datetime as dt
import os
import sys
import threading
import time
import urllib.request

BASE = "https://ocmgeodatastor1.blob.core.windows.net/marinecadastre/ais2024/"
N_CONN = 16
CHUNK = 1 << 18


def remote_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def fetch_range(url, start, end, dest, offset, progress, lock, errors, retries=8):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "r+b") as fh:
                fh.seek(offset)
                pos = 0
                while True:
                    b = r.read(CHUNK)
                    if not b:
                        break
                    fh.write(b)
                    pos += len(b)
                    with lock:
                        progress[0] += len(b)
            if pos == end - start + 1:
                return
        except Exception as e:  # noqa: BLE001
            time.sleep(5 * (attempt + 1))
    errors.append(f"range {start}-{end} failed after {retries} attempts")


def download(url, dest, n_conn=N_CONN):
    size = remote_size(url)
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        print(f"  skip (complete): {os.path.basename(dest)} {size/1e6:.0f} MB", flush=True)
        return
    tmp = dest + ".part"
    with open(tmp, "wb") as fh:
        fh.truncate(size)
    bounds = [(i * size // n_conn, (i + 1) * size // n_conn - 1) for i in range(n_conn)]
    progress, lock, errors = [0], threading.Lock(), []
    threads = [threading.Thread(target=fetch_range, args=(url, s, e, tmp, s, progress, lock, errors))
               for s, e in bounds]
    t0 = time.time()
    for th in threads:
        th.start()
    while any(th.is_alive() for th in threads):
        time.sleep(15)
        done = progress[0]
        rate = done / max(time.time() - t0, 1e-6) / 1e6
        print(f"    {os.path.basename(dest)}: {done/1e6:.0f}/{size/1e6:.0f} MB  {rate:.2f} MB/s", flush=True)
    for th in threads:
        th.join()
    if errors or os.path.getsize(tmp) != size:
        os.remove(tmp)
        raise RuntimeError("; ".join(errors) or "size mismatch after download")
    os.replace(tmp, dest)
    print(f"  done: {os.path.basename(dest)} in {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-07-01")
    ap.add_argument("--end", default="2024-07-31")
    ap.add_argument("--out", default="data/real_ais/raw")
    ap.add_argument("--conn", type=int, default=N_CONN)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d0 = dt.date.fromisoformat(args.start)
    d1 = dt.date.fromisoformat(args.end)
    day = d0
    while day <= d1:
        name = f"ais-{day.isoformat()}.parquet"
        print(f"[{time.strftime('%H:%M:%S')}] {name}", flush=True)
        for attempt in range(1, 4):
            try:
                download(BASE + name, os.path.join(args.out, name), args.conn)
                break
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED {name} (attempt {attempt}): {e}", file=sys.stderr, flush=True)
                time.sleep(60 * attempt)
        day += dt.timedelta(days=1)


if __name__ == "__main__":
    main()
