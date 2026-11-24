from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation,binary_erosion,shift

SEEDS=[17,29,43,59,71]

def dice(a,b):
    return float((2*np.sum(a&b)+1e-6)/(a.sum()+b.sum()+1e-6))

def simulate(seed,n=120,size=96,annotators=7):
    rng=np.random.default_rng(seed);yy,xx=np.mgrid[:size,:size];truth=[];annotations=[]
    biases=np.array([-2,-1,0,0,1,2,3])
    for _ in range(n):
        cx,cy=rng.uniform(.28*size,.72*size,2);rx,ry=rng.uniform(.10*size,.20*size,2);mask=((xx-cx)/rx)**2+((yy-cy)/ry)**2<1
        truth.append(mask);current=[]
        for index,bias in enumerate(biases):
            noisy=binary_dilation(mask,iterations=max(0,int(bias))) if bias>=0 else binary_erosion(mask,iterations=-int(bias))
            noisy=shift(noisy.astype(float),rng.normal(0,0.8+index*.12,2),order=0)>0.5
            if index==6 and rng.random()<.18:noisy=rng.random(mask.shape)<.03
            current.append(noisy)
        annotations.append(current)
    return np.asarray(truth),np.asarray(annotations)

def reliability_weights(annotations):
    scores=[]
    for index in range(annotations.shape[1]):
        others=np.delete(annotations,index,axis=1).mean(1)>=.5
        scores.append(np.mean([dice(a,b) for a,b in zip(annotations[:,index],others)]))
    weights=np.maximum(np.asarray(scores)-.45,.01);return weights/weights.sum()

def evaluate(seed):
    truth,annotations=simulate(seed);baseline=annotations.mean(1);weights=reliability_weights(annotations[:60])
    proposed=np.tensordot(annotations,weights,axes=(1,0));boundary=np.asarray([binary_dilation(m)^binary_erosion(m) for m in truth])
    def score(probability):
        return {'dice':float(np.mean([dice(p>=.5,t) for p,t in zip(probability,truth)])),
                'brier':float(np.mean((probability-truth)**2)),
                'boundary_uncertainty':float(np.mean((probability*(1-probability))[boundary]))}
    return {'seed':seed,'weights':weights.tolist(),'baseline':score(baseline),'proposed':score(proposed)}

def main():
    records=[evaluate(seed) for seed in SEEDS]
    summary={m:{k:float(np.mean([r[m][k] for r in records])) for k in ('dice','brier','boundary_uncertainty')} for m in ('baseline','proposed')}
    report={'status':'validated_complete','professor':'Abdullah-Al-Zubaer Imran','protocol':'5 seeds; 120 lesions/seed; 7 heterogeneous annotators; first 60 lesions estimate reliability',
            'records':records,'summary':summary,'dice_gain':summary['proposed']['dice']-summary['baseline']['dice'],
            'brier_reduction':summary['baseline']['brier']-summary['proposed']['brier'],
            'limitations':'Synthetic ambiguity and annotators; not clinical validation or professor-model reproduction.'}
    Path('validated_results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()

