from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from time import sleep
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[3]
SOURCE = "https://physionet.org/files/eegmmidb/1.0.0/"
directory = ROOT / "datasets/eegmmidb"
directory.mkdir(parents=True, exist_ok=True)
with urlopen(SOURCE + "SHA256SUMS.txt", timeout=20) as response:
    checks = {
        name.lstrip("*"): checksum
        for checksum, name in [
            row.split(None, 1)
            for row in response.read().decode().splitlines()
            if row.strip()
        ]
    }


def download(p):
    name = f"S{p:03}/S{p:03}R04.edf"
    local = directory / name.split("/")[-1]
    expected = checks.get(name, checks.get("./" + name))
    if expected is None:
        raise ValueError(f"No published checksum for {name}")
    if local.exists() and sha256(local.read_bytes()).hexdigest() == expected:
        return p
    for retry in range(3):
        try:
            with urlopen(SOURCE + name, timeout=25) as response:
                raw = response.read()
            if sha256(raw).hexdigest() != expected:
                raise ValueError(f"Checksum mismatch for {name}")
            stage = local.with_suffix(".part")
            stage.write_bytes(raw)
            stage.replace(local)
            return p
        except (OSError, ValueError):
            if retry == 2:
                raise
            sleep(retry + 1)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as pool:
        for ready, _ in enumerate(pool.map(download, range(1, 110)), start=1):
            if ready % 10 == 0 or ready == 109:
                print(f"Verified {ready}/109 EDF recordings", flush=True)
