import os
from pathlib import Path
from urllib.request import urlretrieve

data = Path(os.environ.get("PORTFOLIO_DATA", Path(__file__).resolve().parent / "data"))
data.mkdir(parents=True, exist_ok=True)
target = data / "diabetes130.zip"
if not target.exists():
    temp = target.with_suffix(".part")
    urlretrieve(
        "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip",
        temp,
    )
    temp.replace(target)
