import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
def data(n=2200,seed=27):
 r=np.random.default_rng(seed);y=r.binomial(1,.42,n);x=r.normal(size=(n,12));x[:,:4]+=y[:,None]*.65;return x,y
def evaluate(seed=27):
 x,y=data(seed=seed);lab=120;un=1500;test=x[un:];yt=y[un:];base=LogisticRegression(max_iter=500).fit(x[:lab],y[:lab]);centers=np.array([x[:lab][y[:lab]==v].mean(0) for v in [0,1]]);d=((x[lab:un,None,:]-centers[None,:,:])**2).mean(2);margin=np.abs(d[:,0]-d[:,1]);keep=margin>np.quantile(margin,.65);pseudo=np.argmin(d[keep],1);semi=LogisticRegression(max_iter=500).fit(np.vstack([x[:lab],x[lab:un][keep]]),np.r_[y[:lab],pseudo]);return {"supervised_auroc":float(roc_auc_score(yt,base.predict_proba(test)[:,1])),"prototype_semisupervised_auroc":float(roc_auc_score(yt,semi.predict_proba(test)[:,1])),"pseudo_label_accuracy":float(np.mean(pseudo==y[lab:un][keep])),"pseudo_label_coverage":float(keep.mean())}
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("outputs/demo"));p.add_argument("--seed",type=int,default=27);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r=evaluate(a.seed);(a.output/"metrics.json").write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=="__main__":main()
