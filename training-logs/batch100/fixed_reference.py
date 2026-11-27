from pathlib import Path
import json
import numpy as np

from train import scores
from data import load_task

root=Path(__file__).resolve().parents[2]
manifest=root/"training-logs/training-batch-100.json"
for rec in json.loads(manifest.read_text()):
    i=rec["i"]
    if i not in (4,5,6,14):continue
    folder=root/rec["folder"]
    report=json.loads((folder/"results.json").read_text())
    data=load_task(root/"datasets",i,rec["real"])
    for seed,run in report["runs"].items():
        with np.load(folder/"checkpoints"/f"split-{seed}.npz") as stats:
            test=stats["test"]
        x=data["x"][test];y=data["y"][test];kind=data["kind"]
        if i==4:
            # Last glucose reading persists into the next eligible sample.
            margin=(x[:,0,-1]-1.8)*3
            pred=np.stack((-margin,margin),axis=1)
        elif i==6:
            pred=(x[:,:1]-.7)*10
        else:
            pred=np.zeros_like(y)
        run["metrics"]["fixed_reference"]=scores(y,pred,kind)
    metrics=next(iter(report["runs"].values()))["metrics"]["fixed_reference"]
    report["means"]["fixed_reference"]={k:float(np.mean([r["metrics"]["fixed_reference"][k] for r in report["runs"].values()])) for k in metrics}
    (folder/"results.json").write_text(json.dumps(report,indent=2))
    print(rec["prof"],rec["real"],report["means"]["fixed_reference"])
