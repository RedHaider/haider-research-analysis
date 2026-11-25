import hashlib
import os
from pathlib import Path

DATA = Path(os.environ.get("PORTFOLIO_DATA", Path(__file__).resolve().parent / "data"))
from urllib.request import urlretrieve

p = DATA / "medmnist/pathmnist.npz"
p.parent.mkdir(parents=True, exist_ok=True)
if not p.exists():
    urlretrieve("https://zenodo.org/records/10519652/files/pathmnist.npz?download=1", p)
assert hashlib.md5(p.read_bytes()).hexdigest() == "a8b06965200029087d5bd730944a56c1"
