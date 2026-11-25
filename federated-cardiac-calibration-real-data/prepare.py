import os
from pathlib import Path

DATA = Path(os.environ.get("PORTFOLIO_DATA", Path(__file__).resolve().parent / "data"))
import zipfile
from urllib.request import urlretrieve

p = DATA / "heart-sites/heart.zip"
p.parent.mkdir(parents=True, exist_ok=True)
if not p.exists():
    urlretrieve("https://archive.ics.uci.edu/static/public/45/heart+disease.zip", p)
with zipfile.ZipFile(p) as archive:
    for name in archive.namelist():
        if Path(name).name.startswith("processed.") and name.endswith(".data"):
            (p.parent / Path(name).name).write_bytes(archive.read(name))
