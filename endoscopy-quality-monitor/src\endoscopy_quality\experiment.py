import argparse,json;from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
def evaluate(seed=27):
 r=np.random.default_rng(seed);truth=[];pred=[]
 for _ in range(120):
  y=np.zeros(100,int);start=r.integers(20,70);y[start:start+r.integers(8,20)]=1;signal=r.normal(size=100)+y*1.4;rolling=np.convolve(signal,np.ones(5)/5,'same');p=rolling>.65;truth.extend(y);pred.extend(p)
 return {'frame_f1':float(f1_score(truth,pred)),'alert_rate':float(np.mean(pred)),'event_prevalence':float(np.mean(truth))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/demo'));p.add_argument('--seed',type=int,default=27);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r=evaluate(a.seed);(a.output/'metrics.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
