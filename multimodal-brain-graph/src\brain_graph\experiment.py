import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
def data(n=1800,seed=27):
 r=np.random.default_rng(seed);y=r.binomial(1,.5,n);latent=r.normal(size=(n,6));latent[:,:2]+=y[:,None]*.55;struct=latent@r.normal(size=(6,12))+r.normal(size=(n,12));func=latent@r.normal(size=(6,12))+r.normal(size=(n,12));return struct,func,y
def evaluate(seed=27):
 s,f,y=data(seed=seed);cut=1200;sets={"structural":s,"functional":f,"multimodal":np.c_[s,f]};out={}
 for name,x in sets.items():m=LogisticRegression(max_iter=500).fit(x[:cut],y[:cut]);out[name]={"auroc":float(roc_auc_score(y[cut:],m.predict_proba(x[cut:])[:,1]))}
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("outputs/demo"));p.add_argument("--seed",type=int,default=27);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r=evaluate(a.seed);(a.output/"metrics.json").write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=="__main__":main()
