"""Validate the links and reports in the published training portfolio."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDIES = json.loads(Path(__file__).with_name("studies.json").read_text(encoding="utf-8"))

assert len(STUDIES) == len({study["professor"] for study in STUDIES}) == 32
for study in STUDIES:
    assert (ROOT / study["trainer"]).is_file()
    for key in ("aligned", "public_data"):
        path = ROOT / study[key]
        assert path.with_name("README.md").is_file()
        assert json.loads(path.read_text(encoding="utf-8"))
print("Verified 32 professor pairs and 64 public experiment reports.")
