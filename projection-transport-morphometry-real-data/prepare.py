from pathlib import Path
import hashlib
import os
import urllib.request
import zipfile

root = Path(os.environ.get("PORTFOLIO_DATA", Path(__file__).resolve().parent / "data"))
path = root / 'medmnist/pneumoniamnist.npz'
path.parent.mkdir(parents=True, exist_ok=True)
if not path.exists():
    with urllib.request.urlopen('https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1',timeout=120) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != 'e1792d3f03751cb101e99f19a63b3c1941436c988665f47853417b05be250cd8':
        raise ValueError("Downloaded release checksum mismatch")
    path.write_bytes(payload)
if hashlib.sha256(path.read_bytes()).hexdigest() != 'e1792d3f03751cb101e99f19a63b3c1941436c988665f47853417b05be250cd8':
    raise ValueError("Local dataset checksum mismatch")
