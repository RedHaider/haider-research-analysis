import argparse,json;from pathlib import Path
import numpy as np
def evaluate(seed=27):
 r=np.random.default_rng(seed);t=np.arange(500);pos=np.c_[np.sin(t/28),np.cos(t/35)]+r.normal(scale=.015,size=(500,2));out={}
 for delay in [1,3,5]:
  direct=pos[:-delay];truth=pos[delay:];vel=np.diff(pos,axis=0,prepend=pos[:1]);pred=pos[:-delay]+delay*vel[:-delay];out[f'delay_{delay}']={'direct_rmse':float(np.sqrt(np.mean((direct-truth)**2))),'velocity_rmse':float(np.sqrt(np.mean((pred-truth)**2)))}
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/demo'));p.add_argument('--seed',type=int,default=27);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r=evaluate(a.seed);(a.output/'metrics.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
