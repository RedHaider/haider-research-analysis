"""Fetch the official 28px release and check the publisher's checksum."""

import hashlib
from pathlib import Path
from urllib.request import urlopen


def main():
    target = Path(__file__).resolve().parent / "data" / "pneumoniamnist.npz"
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        payload = target.read_bytes()
    else:
        with urlopen(
            "https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1",
            timeout=90,
        ) as response:
            payload = response.read()
    if hashlib.md5(payload).hexdigest() != "28209eda62fecd6e6a2d98b1501bb15f":
        raise ValueError("PneumoniaMNIST checksum mismatch")
    if not target.exists():
        target.write_bytes(payload)
    print(target)


if __name__ == "__main__":
    main()
