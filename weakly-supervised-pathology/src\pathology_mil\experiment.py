import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
def bags(n=1000,seed=27):
 r=np.random.default_rng(seed);y=r.binomial(1,.5,n);x=r.normal(size=(n,30,6));
 for i in np.where(y==1)[0]:x[i,r.choice(30,4,False),:2]+=2.2
 return x,y
def evaluate(seed=27):
 x,y=bags(seed=seed);cut=700;patch=LogisticRegression(max_iter=500).fit(x[:cut].reshape(-1,6),np.repeat(y[:cut],30));s=patch.predict_proba(x.reshape(-1,6))[:,1].reshape(len(x),30);features={"mean_pool":s.mean(1),"max_pool":s.max(1),"top4_attention":np.sort(s,axis=1)[:,-4:].mean(1)};return {k:{"auroc":float(roc_auc_score(y[cut:],v[cut:]))} for k,v in features.items()}
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("outputs/demo"));p.add_argument("--seed",type=int,default=27);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r=evaluate(a.seed);(a.output/"metrics.json").write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=="__main__":main()
