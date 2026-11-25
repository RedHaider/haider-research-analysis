from pathlib import Path
import hashlib
import os
import urllib.request
import zipfile

root = Path(os.environ.get("PORTFOLIO_DATA", Path(__file__).resolve().parent / "data"))
path = root / 'lung-survival.csv'
path.parent.mkdir(parents=True, exist_ok=True)
if not path.exists():
    with urllib.request.urlopen('https://raw.githubusercontent.com/CamDavidsonPilon/lifelines/master/lifelines/datasets/lung.csv',timeout=120) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != '16ebd02c38bbb5708186d600cff80ad84d0764974619436f83ba4a605e77eb44':
        raise ValueError("Downloaded release checksum mismatch")
    path.write_bytes(payload)
if hashlib.sha256(path.read_bytes()).hexdigest() != '16ebd02c38bbb5708186d600cff80ad84d0764974619436f83ba4a605e77eb44':
    raise ValueError("Local dataset checksum mismatch")
