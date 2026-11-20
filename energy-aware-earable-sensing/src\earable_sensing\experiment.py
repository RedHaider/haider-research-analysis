import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
def data(n=2200,seed=27):
 r=np.random.default_rng(seed);x=r.normal(size=(n,9));p=1/(1+np.exp(-(.9*x[:,0]-.7*x[:,3]+.6*x[:,6]+.45*x[:,7])));return x,r.binomial(1,p)
def evaluate(seed=27):
 x,y=data(seed=seed);cut=1500;configs={"motion_only":([0,1,2],1),"motion_acoustic":(list(range(6)),3),"all_sensors":(list(range(9)),6)};out={}
 for name,(cols,cost) in configs.items():m=LogisticRegression(max_iter=500).fit(x[:cut,cols],y[:cut]);out[name]={"auroc":float(roc_auc_score(y[cut:],m.predict_proba(x[cut:,cols])[:,1])),"relative_energy_cost":cost}
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("outputs/demo"));p.add_argument("--seed",type=int,default=27);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r=evaluate(a.seed);(a.output/"metrics.json").write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=="__main__":main()
