from __future__ import annotations
import json,sys,time
from pathlib import Path
import cv2,numpy as np
from scipy.ndimage import binary_dilation,binary_erosion
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from kvasir_prob.features import pixel_features
DATA=Path('D:/USA based documents/Haider Heavy Research Projects/datasets/kvasir-seg/extracted/Kvasir-SEG');SEEDS=[17,29,43]
def pairs():
    images={p.stem:p for p in (DATA/'images').glob('*')};masks={p.stem:p for p in (DATA/'masks').glob('*')};return [(images[k],masks[k]) for k in sorted(images.keys()&masks.keys())]
def load(pair,size=64):
    image=cv2.imread(str(pair[0]));mask=cv2.resize(cv2.imread(str(pair[1]),0),(size,size),interpolation=cv2.INTER_NEAREST)>127
    return pixel_features(image,size),mask
def dice(a,b):return float((2*np.sum(a&b)+1e-6)/(a.sum()+b.sum()+1e-6))
def main():
    started=time.time();all_pairs=pairs();records=[]
    for seed in SEEDS:
        train,test=train_test_split(all_pairs,test_size=200,random_state=seed);rng=np.random.default_rng(seed);xs=[];ys=[]
        for pair in train[:500]:
            x,y=load(pair);idx=rng.choice(len(y.ravel()),350,replace=False);xs.append(x[idx]);ys.append(y.ravel()[idx])
        xtrain=np.concatenate(xs);ytrain=np.concatenate(ys)
        baseline=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced')).fit(xtrain,ytrain)
        ensemble=[]
        for member in range(5):
            idx=np.random.default_rng(seed*10+member).choice(len(ytrain),len(ytrain),replace=True)
            ensemble.append(ExtraTreesClassifier(n_estimators=60,min_samples_leaf=5,n_jobs=-1,class_weight='balanced',random_state=seed*10+member).fit(xtrain[idx],ytrain[idx]))
        stats={m:{'dice':[],'brier':[],'boundary_uncertainty':[]} for m in ('baseline','proposed')}
        for pair in test:
            x,y=load(pair);base=baseline.predict_proba(x)[:,1].reshape(y.shape);members=np.stack([m.predict_proba(x)[:,1] for m in ensemble]);proposed=members.mean(0).reshape(y.shape)
            boundary=binary_dilation(y)^binary_erosion(y)
            for method,p in [('baseline',base),('proposed',proposed)]:
                stats[method]['dice'].append(dice(p>=.5,y));stats[method]['brier'].append(brier_score_loss(y.ravel(),p.ravel()));stats[method]['boundary_uncertainty'].append(float((p*(1-p))[boundary].mean()))
        records.append({'seed':seed,**{m:{k:float(np.mean(v)) for k,v in d.items()} for m,d in stats.items()}});print(json.dumps({'completed_seed':seed}),flush=True)
    summary={m:{k:float(np.mean([r[m][k] for r in records])) for k in ('dice','brier','boundary_uncertainty')} for m in ('baseline','proposed')}
    report={'status':'validated_complete','professor':'Abdullah-Al-Zubaer Imran','dataset':'Kvasir-SEG real images/masks; 500 training and 200 held-out test images per seed at 64px','protocol':'3 seeds; balanced logistic baseline; five-member bootstrap Extra-Trees probabilistic ensemble','elapsed_minutes':(time.time()-started)/60,'records':records,'summary':summary,'dice_gain':summary['proposed']['dice']-summary['baseline']['dice'],'brier_reduction':summary['baseline']['brier']-summary['proposed']['brier'],'limitations':'Classical low-resolution pixel model; image-level split because patient IDs unavailable; not clinical or professor-model claim.'}
    (ROOT/'validated_results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()

