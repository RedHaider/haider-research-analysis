import argparse,json;from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score
def data(n,seed,shift=0):
 r=np.random.default_rng(seed);x=r.normal(size=(n,12));x[:,:4]+=shift;p=1/(1+np.exp(-(x[:,0]-.8*x[:,2]+.6*x[:,5])));return x,r.binomial(1,p)
def evaluate(seed=27):
 x,y=data(1800,seed);xt,yt=data(900,seed+1,.4);m=LogisticRegression(max_iter=500).fit(x,y);p=m.predict_proba(xt)[:,1];out={'auroc':float(roc_auc_score(yt,p))}
 for threshold in [.2,.4,.6]:keep=np.abs(p-.5)*2>=threshold;out[f'abstain_{threshold}']={'coverage':float(keep.mean()),'accuracy':float(accuracy_score(yt[keep],p[keep]>=.5))}
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/demo'));p.add_argument('--seed',type=int,default=27);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r=evaluate(a.seed);(a.output/'metrics.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
