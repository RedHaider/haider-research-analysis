import os
from pathlib import Path
from urllib.request import urlretrieve

data = Path(os.environ.get("PORTFOLIO_DATA", Path(__file__).resolve().parent / "data"))
data.mkdir(parents=True, exist_ok=True)
target = data / "oasis2.xlsx"
if not target.exists():
    temp = target.with_suffix(".part")
    urlretrieve(
        "https://sites.wustl.edu/oasisbrains/files/2024/03/oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx",
        temp,
    )
    temp.replace(target)
