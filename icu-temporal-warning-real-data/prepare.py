import os
from pathlib import Path
from urllib.request import urlretrieve

root = (
    Path(os.environ.get("PORTFOLIO_DATA", Path(__file__).resolve().parent / "data"))
    / "icu2012"
)
root.mkdir(parents=True, exist_ok=True)
for name in ["set-a.tar.gz", "Outcomes-a.txt"]:
    target = root / name
    if not target.exists():
        temp = target.with_suffix(".part")
        urlretrieve("https://physionet.org/files/challenge-2012/1.0.0/" + name, temp)
        temp.replace(target)
