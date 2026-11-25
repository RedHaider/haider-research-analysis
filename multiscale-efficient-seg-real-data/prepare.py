import hashlib
import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

root = (
    Path(os.environ.get("PORTFOLIO_DATA", Path(__file__).resolve().parent / "data"))
    / "kvasir-seg"
)
root.mkdir(parents=True, exist_ok=True)
target = root / "kvasir-seg.zip"
if not target.exists():
    temp = target.with_suffix(".part")
    urlretrieve("https://datasets.simula.no/downloads/kvasir-seg.zip", temp)
    temp.replace(target)
assert (
    hashlib.sha256(target.read_bytes()).hexdigest()
    == "03b30e21d584e04facf49397a2576738fd626815771afbbf788f74a7153478f7"
)
with zipfile.ZipFile(target) as archive:
    archive.extractall(root / "extracted")
