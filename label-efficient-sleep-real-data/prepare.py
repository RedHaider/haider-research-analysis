import hashlib
import os
from pathlib import Path

DATA = Path(os.environ.get("PORTFOLIO_DATA", Path(__file__).resolve().parent / "data"))
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen

base = "https://physionet.org/files/sleep-accel/1.0.0/"
manifest = urlopen(base + "SHA256SUMS.txt").read().decode().splitlines()


def fetch(line):
    checksum, name = line.split()
    name = name.lstrip("*").removeprefix("./")
    if not name.startswith(("heart_rate/", "labels/")):
        return
    p = DATA / "sleep-accel" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_bytes(urlopen(base + name, timeout=90).read())
    assert hashlib.sha256(p.read_bytes()).hexdigest() == checksum


with ThreadPoolExecutor(4) as pool:
    list(pool.map(fetch, manifest))
