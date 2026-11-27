import json
from pathlib import Path
import numpy as np
from data import load_task

root=Path(__file__).resolve().parents[2]
records=json.loads((root/"training-logs/training-batch-79.json").read_text())
for record in records:
    if record["mode"]!="flow":continue
    folder=root/record["folder"]
    source=folder/"results.json"
    report=json.loads(source.read_text())
    y=load_task(root/"datasets","flow",record["real"])["y"]
    for seed,run in report["runs"].items():
        with np.load(folder/"checkpoints"/f"split-{seed}.npz") as split:
            error=float(np.sqrt(((y[split["test"]]*8)**2).sum(axis=1)).mean())
        run["metrics"]["fixed_reference"]={"endpoint_error_px":error}
    report["means"]["fixed_reference"]={"endpoint_error_px":float(np.mean([r["metrics"]["fixed_reference"]["endpoint_error_px"] for r in report["runs"].values()]))}
    source.write_text(json.dumps(report,indent=2))
    print(record["real"],report["means"]["fixed_reference"])
