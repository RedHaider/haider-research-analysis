import argparse,json;from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
def data(n=2200,seed=27):
 r=np.random.default_rng(seed);base=r.normal(size=(n,4));trend=r.normal(scale=.18,size=(n,4));seq=base[:,None,:]+np.arange(12)[None,:,None]*trend[:,None,:]+r.normal(scale=.3,size=(n,12,4));p=1/(1+np.exp(-(1.1*trend[:,0]-.9*trend[:,1]+.5*base[:,2])));return seq,r.binomial(1,p)
def evaluate(seed=27):
 s,y=data(seed=seed);cut=1500;out={}
 for w in [3,6,12]:
  x=np.c_[s[:,:w].mean(1),(s[:,w-1]-s[:,0])/max(1,w-1)];m=LogisticRegression(max_iter=500).fit(x[:cut],y[:cut]);out[f"{w}_visits"]={"auroc":float(roc_auc_score(y[cut:],m.predict_proba(x[cut:])[:,1])),"lead_visits":12-w}
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/demo'));p.add_argument('--seed',type=int,default=27);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r=evaluate(a.seed);(a.output/'metrics.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
