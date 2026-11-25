from pathlib import Path
import hashlib
import os
import urllib.request
import zipfile

root = Path(os.environ.get("PORTFOLIO_DATA", Path(__file__).resolve().parent / "data"))
path = root / 'medmnist/breastmnist.npz'
path.parent.mkdir(parents=True, exist_ok=True)
if not path.exists():
    with urllib.request.urlopen('https://zenodo.org/records/10519652/files/breastmnist.npz?download=1',timeout=120) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != '3a22d2a4625e7dd8903f0f59030e81a3d0c222bece2e5e10927456cb50efdec1':
        raise ValueError("Downloaded release checksum mismatch")
    path.write_bytes(payload)
if hashlib.sha256(path.read_bytes()).hexdigest() != '3a22d2a4625e7dd8903f0f59030e81a3d0c222bece2e5e10927456cb50efdec1':
    raise ValueError("Local dataset checksum mismatch")
