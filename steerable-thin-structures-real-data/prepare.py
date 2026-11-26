from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen
import os
import time

ROOT = (
    Path(os.environ.get("PORTFOLIO_DATA", Path(__file__).resolve().parent / "data"))
    / "drive"
)
REPO = "https://huggingface.co/datasets/Zomba/DRIVE-digital-retinal-images-for-vessel-extraction/resolve/e43e3f8a4b1146db231648786912d0ac882067b2/"


def obtain(name):
    target = ROOT / name
    if target.exists() and target.stat().st_size > 5000:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            with urlopen(REPO + name, timeout=35) as reply:
                data = reply.read()
            if len(data) < 5000:
                raise ValueError("Incomplete image " + name)
            temp = target.with_suffix(".part")
            temp.write_bytes(data)
            temp.replace(target)
            return
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2)


if __name__ == "__main__":
    files = [
        f"{split}/{kind}/{i:02}{'_manual1' if split == 'val' and kind == 'label' else ''}.{ext}"
        for split, ids in [("train", range(21, 41)), ("val", range(1, 21))]
        for i in ids
        for kind, ext in [("input", "tif"), ("label", "png")]
    ]
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(obtain, files))
