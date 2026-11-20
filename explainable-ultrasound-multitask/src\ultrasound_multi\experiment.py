import argparse,json;from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score
def data(n=2200,seed=27):
 r=np.random.default_rng(seed);x=r.normal(size=(n,10));risk=1.1*x[:,0]+.8*x[:,1]-.6*x[:,3]+.5*x[:,6];y=r.binomial(1,1/(1+np.exp(-risk)));cat=np.digitize(risk+r.normal(size=n),[-1,0,1]);return x,y,cat
def evaluate(seed=27):
 x,y,c=data(seed=seed);cut=1500;m=LogisticRegression(max_iter=500).fit(x[:cut],y[:cut]);mc=LogisticRegression(max_iter=500).fit(x[:cut],c[:cut]);coef=np.abs(m.coef_[0])+np.abs(mc.coef_).mean(0);return {'malignancy_auroc':float(roc_auc_score(y[cut:],m.predict_proba(x[cut:])[:,1])),'category_accuracy':float(accuracy_score(c[cut:],mc.predict(x[cut:]))),'top_shared_feature':int(np.argmax(coef))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/demo'));p.add_argument('--seed',type=int,default=27);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r=evaluate(a.seed);(a.output/'metrics.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
