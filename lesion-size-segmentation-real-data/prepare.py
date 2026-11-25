from pathlib import Path
import hashlib
import os
import urllib.request
import zipfile

root = Path(os.environ.get("PORTFOLIO_DATA", Path(__file__).resolve().parent / "data"))
path = root / 'kvasir-seg/kvasir-seg.zip'
path.parent.mkdir(parents=True, exist_ok=True)
if not path.exists():
    with urllib.request.urlopen('https://datasets.simula.no/downloads/kvasir-seg.zip',timeout=120) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != '03b30e21d584e04facf49397a2576738fd626815771afbbf788f74a7153478f7':
        raise ValueError("Downloaded release checksum mismatch")
    path.write_bytes(payload)
if hashlib.sha256(path.read_bytes()).hexdigest() != '03b30e21d584e04facf49397a2576738fd626815771afbbf788f74a7153478f7':
    raise ValueError("Local dataset checksum mismatch")
with zipfile.ZipFile(path) as archive:
    archive.extractall(path.parent / "extracted")
