import argparse,json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter,rotate
def phantom(seed=27,size=48):
 r=np.random.default_rng(seed);y,x=np.mgrid[:size,:size];im=np.zeros((size,size))
 for _ in range(3):cx,cy=r.uniform(12,36,2);rx,ry=r.uniform(4,10,2);im+=r.uniform(.5,1)*(((x-cx)/rx)**2+((y-cy)/ry)**2<1)
 return np.clip(im,0,1)
def project(im,angles):return np.array([rotate(im,a,reshape=False,order=1).sum(0) for a in angles])
def backproject(sino,angles):
 out=np.zeros((sino.shape[1],sino.shape[1]))
 for p,a in zip(sino,angles):out+=rotate(np.tile(p,(len(p),1)),-a,reshape=False,order=1)
 out/=len(angles);return (out-out.min())/(out.max()-out.min()+1e-9)
def metrics(a,b):return {"rmse":float(np.sqrt(np.mean((a-b)**2))),"correlation":float(np.corrcoef(a.ravel(),b.ravel())[0,1])}
def evaluate(seed=27):
 vals={"backprojection":[],"physics_smoothed":[]};angles=np.linspace(-55,55,18)
 for k in range(30):im=phantom(seed+k);raw=backproject(project(im,angles),angles);smooth=gaussian_filter(raw,.8);vals["backprojection"].append(metrics(im,raw));vals["physics_smoothed"].append(metrics(im,smooth))
 return {k:{m:float(np.mean([z[m] for z in v])) for m in ["rmse","correlation"]} for k,v in vals.items()}
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("outputs/demo"));p.add_argument("--seed",type=int,default=27);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r=evaluate(a.seed);(a.output/"metrics.json").write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=="__main__":main()
