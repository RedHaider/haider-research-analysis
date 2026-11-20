import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
def data(n=2600,seed=27):
 r=np.random.default_rng(seed);x=r.normal(size=(n,24));w=r.normal(size=24);p=1/(1+np.exp(-(x@w/4)));return x,r.binomial(1,p)
def evaluate(seed=27):
 x,y=data(seed=seed);cut=1800;m=LogisticRegression(max_iter=500).fit(x[:cut],y[:cut]);w=m.coef_[0];b=m.intercept_[0];base=(x[cut:]@w+b)>0;out={}
 for bits in [32,16,8,4]:
  if bits==32:q=w
  else:levels=2**(bits-1)-1;scale=np.max(np.abs(w))/levels;q=np.round(w/scale)*scale
  score=x[cut:]@q+b;out[f"int{bits}" if bits<32 else "float32"]={"auroc":float(roc_auc_score(y[cut:],score)),"model_bytes":int((len(w)+1)*bits/8),"prediction_disagreement":float(np.mean((score>0)!=base))}
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("outputs/demo"));p.add_argument("--seed",type=int,default=27);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r=evaluate(a.seed);(a.output/"metrics.json").write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=="__main__":main()
