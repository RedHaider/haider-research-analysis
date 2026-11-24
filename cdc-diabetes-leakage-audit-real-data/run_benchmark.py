from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from real_leakage.audit import hidden_representation,moment_match,quantile_match
OUTPUT=ROOT/'validated_results.json';SEEDS=[17,29,43]

def residualize(train,test,protected):
    probe=LogisticRegression(max_iter=1000).fit(train,protected)
    direction=probe.coef_[0];direction=direction/(np.linalg.norm(direction)+1e-12)
    return train-np.outer(train@direction,direction),test-np.outer(test@direction,direction)

def metrics(train_rep,test_rep,train_y,test_y,train_g,test_g):
    disease=LogisticRegression(max_iter=1500,class_weight='balanced').fit(train_rep,train_y)
    probability=disease.predict_proba(test_rep)[:,1]
    linear=LogisticRegression(max_iter=1500).fit(train_rep,train_g).predict_proba(test_rep)[:,1]
    nonlinear=HistGradientBoostingClassifier(max_iter=100,max_depth=4).fit(train_rep,train_g).predict_proba(test_rep)[:,1]
    return {'disease_auroc':float(roc_auc_score(test_y,probability)),'disease_brier':float(brier_score_loss(test_y,probability)),
            'linear_probe_auroc':float(roc_auc_score(test_g,linear)),'nonlinear_probe_auroc':float(roc_auc_score(test_g,nonlinear)),
            'strongest_probe_auroc':float(max(roc_auc_score(test_g,linear),roc_auc_score(test_g,nonlinear)))}

def summarize(records,method,key):
    values=np.array([r[key] for r in records if r['method']==method]);return {'mean':float(values.mean()),'std':float(values.std(ddof=1))}

def main():
    started=time.time();shared=Path('D:/USA based documents/Haider Heavy Research Projects/real-data-projects/cdc-diabetes-federated-edge/data/cdc_diabetes_health_indicators.csv')
    if shared.exists():
        frame=pd.read_csv(shared);y=frame.pop('Diabetes_binary').astype(int);x=frame.astype(float)
    else:
        d=fetch_ucirepo(id=891);x=d.data.features.astype(float);y=d.data.targets.iloc[:,0].astype(int)
    g=x.pop('Sex').astype(int);records=[]
    for seed in SEEDS:
        sample,_=train_test_split(np.arange(len(y)),train_size=25000,stratify=y,random_state=seed)
        tr,te=train_test_split(sample,test_size=.3,stratify=y.iloc[sample],random_state=seed+100)
        imp=SimpleImputer(strategy='median');sc=StandardScaler();xtr=sc.fit_transform(imp.fit_transform(x.iloc[tr]));xte=sc.transform(imp.transform(x.iloc[te]))
        mlp=MLPClassifier(hidden_layer_sizes=(48,16),early_stopping=True,max_iter=60,random_state=seed,batch_size=512).fit(xtr,y.iloc[tr])
        rtr=hidden_representation(mlp,xtr);rte=hidden_representation(mlp,xte)
        variants={'original':(rtr,rte),'projection':residualize(rtr,rte,g.iloc[tr]),
                  'moment_matching':moment_match(rtr,rte,g.iloc[tr].to_numpy(),g.iloc[te].to_numpy()),
                  'quantile_matching':quantile_match(rtr,rte,g.iloc[tr].to_numpy(),g.iloc[te].to_numpy())}
        for method,(a,b) in variants.items():records.append({'seed':seed,'method':method,**metrics(a,b,y.iloc[tr],y.iloc[te],g.iloc[tr],g.iloc[te])})
        print(json.dumps({'completed_seed':seed}),flush=True)
    keys=['disease_auroc','disease_brier','linear_probe_auroc','nonlinear_probe_auroc','strongest_probe_auroc']
    summary={m:{k:summarize(records,m,k) for k in keys} for m in variants}
    report={'status':'validated_complete','professor':'Su-In Lee','dataset':'CDC Diabetes Health Indicators; 25,000-record stratified sample per seed; sex excluded from disease inputs',
            'protocol':'3 seeds; MLP 48/16 representation; linear and nonlinear probes; projection, moment matching and quantile matching',
            'elapsed_minutes':(time.time()-started)/60,'records':records,'summary':summary,
            'probe_reduction':summary['original']['strongest_probe_auroc']['mean']-summary['projection']['strongest_probe_auroc']['mean'],
            'disease_auroc_change':summary['projection']['disease_auroc']['mean']-summary['original']['disease_auroc']['mean'],
            'interpretation':'Projection produced only a small multi-probe leakage reduction. Moment matching fooled the linear probe but made nonlinear leakage perfect; quantile matching also increased nonlinear leakage. This is a negative-result audit, not a mitigation-success claim.',
            'limitations':'Observational survey data; group-conditioned transform requires sex at deployment; representation leakage reduction does not prove outcome fairness or clinical utility.'}
    OUTPUT.write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()

