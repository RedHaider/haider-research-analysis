"""Cache the first 16 participants' unilateral imagery recordings on the data drive."""

import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path(os.environ.get("PORTFOLIO_DATA", Path(__file__).resolve().parent / "data"))
DATA = DATA / "eegmmidb"
BASE = "https://physionet.org/files/eegmmidb/1.0.0"


def fetch(key):
    subject, run = key
    tag = f"S{subject:03d}"
    name = f"{tag}R{run:02d}.edf"
    target = DATA / name
    if target.exists():
        return target
    for attempt in range(3):
        try:
            with urllib.request.urlopen(f"{BASE}/{tag}/{name}", timeout=45) as response:
                data = response.read()
                assert len(data) == int(
                    response.headers.get("Content-Length", len(data))
                )
                assert data.startswith(b"0       ") and len(data) > 1000000
            temp = target.with_suffix(".part")
            temp.write_bytes(data)
            temp.replace(target)
            return target
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2)


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(4) as pool:
        for path in pool.map(fetch, [(i, r) for i in range(1, 17) for r in (4, 8)]):
            print(path.name, flush=True)
