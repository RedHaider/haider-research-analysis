import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression,Ridge
def data(n=5000,seed=27):
 r=np.random.default_rng(seed);x=r.normal(size=(n,6));tau=.4+1.1*(x[:,2]>0)-.7*x[:,3];prop=1/(1+np.exp(-(.9*x[:,0]-.8*x[:,1])));a=r.binomial(1,prop);base=.4*x[:,0]+.3*x[:,4];reward=base+a*tau+r.normal(scale=.7,size=n);return x,a,reward,tau,base
def evaluate(seed=27):
 x,a,r,tau,base=data(seed=seed);cut=3500;prop=LogisticRegression(max_iter=500).fit(x[:cut],a[:cut]).predict_proba(x[:cut])[:,1];out={}
 for name,weights in {"naive":None,"propensity_weighted":np.where(a[:cut]==1,1/np.clip(prop,.05,.95),1/np.clip(1-prop,.05,.95))}.items():
  models=[Ridge(alpha=1).fit(x[:cut][a[:cut]==v],r[:cut][a[:cut]==v],sample_weight=None if weights is None else weights[a[:cut]==v]) for v in [0,1]];policy=models[1].predict(x[cut:])>models[0].predict(x[cut:]);oracle=tau[cut:]>0;value=np.mean(base[cut:]+policy*tau[cut:]);oracle_value=np.mean(base[cut:]+oracle*tau[cut:]);out[name]={"policy_value":float(value),"regret":float(oracle_value-value),"oracle_agreement":float(np.mean(policy==oracle))}
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("outputs/demo"));p.add_argument("--seed",type=int,default=27);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r=evaluate(a.seed);(a.output/"metrics.json").write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=="__main__":main()
