from __future__ import annotations

import argparse, csv, json, random, time
from pathlib import Path
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .data import SegmentationDataset, assert_split_isolation, discover_pairs, fold_indices, transforms
from .metrics import metrics_from_statistics, segmentation_statistics, soft_dice_loss


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def build_model(name):
    if name == "baseline": return smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    return smp.DeepLabV3Plus("efficientnet-b0", encoder_weights="imagenet", in_channels=3, classes=1)


def evaluate(model, loader, device, threshold):
    model.eval(); tp=fp=fn=0; confidence=[]; correctness=[]
    with torch.no_grad():
        for image, mask, _ in loader:
            stats=segmentation_statistics(model(image.to(device)), mask.to(device), threshold)
            tp += stats["tp"]; fp += stats["fp"]; fn += stats["fn"]
            confidence.append(stats["confidence"]); correctness.append(stats["correctness"])
    return metrics_from_statistics(tp, fp, fn, np.concatenate(confidence), np.concatenate(correctness))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,required=True);parser.add_argument("--fold",type=int,required=True)
    parser.add_argument("--model",choices=["baseline","proposed"],required=True);args=parser.parse_args()
    cfg=yaml.safe_load(args.config.read_text()); seed_everything(cfg["seed"]+args.fold)
    pairs=discover_pairs(Path(cfg["data_root"])); train_idx,val_idx,test_idx=fold_indices(len(pairs),cfg["folds"],cfg["seed"],args.fold)
    pick=lambda idx:[pairs[i] for i in idx]
    train_pairs,val_pairs,test_pairs=pick(train_idx),pick(val_idx),pick(test_idx)
    assert_split_isolation(train_pairs,val_pairs,test_pairs)
    robust=args.model=="proposed" and cfg["corruption_augmentation"]
    train=DataLoader(SegmentationDataset(train_pairs,transforms(cfg["image_size"],True,robust)),batch_size=cfg["batch_size"],shuffle=True,num_workers=cfg["num_workers"])
    val=DataLoader(SegmentationDataset(val_pairs,transforms(cfg["image_size"],False,False)),batch_size=cfg["batch_size"],num_workers=cfg["num_workers"])
    test=DataLoader(SegmentationDataset(test_pairs,transforms(cfg["image_size"],False,False)),batch_size=cfg["batch_size"],num_workers=cfg["num_workers"])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model=build_model(args.model).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    scaler=torch.amp.GradScaler("cuda",enabled=cfg["amp"] and device.type=="cuda");out=Path(cfg["output_root"])/args.model/f"fold_{args.fold}";out.mkdir(parents=True,exist_ok=True)
    manifest={"train":[p[0].name for p in pick(train_idx)],"validation":[p[0].name for p in pick(val_idx)],"test":[p[0].name for p in pick(test_idx)]};(out/"split.json").write_text(json.dumps(manifest,indent=2))
    best=-1;stale=0;history=[]
    for epoch in range(1,cfg["epochs"]+1):
        model.train();losses=[];start=time.time()
        for image,mask,_ in train:
            image,mask=image.to(device),mask.to(device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=cfg["amp"] and device.type=="cuda"):
                logits=model(image);loss=F.binary_cross_entropy_with_logits(logits,mask)+soft_dice_loss(logits,mask)
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();losses.append(float(loss))
        metrics=evaluate(model,val,device,cfg["threshold"]);history.append({"epoch":epoch,"loss":np.mean(losses),"seconds":time.time()-start,**metrics})
        if metrics["dice"]>best: best=metrics["dice"];stale=0;torch.save({"model":model.state_dict(),"config":cfg},out/"best.pt")
        else: stale+=1
        if stale>=cfg["patience"]: break
    checkpoint=torch.load(out/"best.pt",map_location=device);model.load_state_dict(checkpoint["model"]);result=evaluate(model,test,device,cfg["threshold"])
    (out/"test_metrics.json").write_text(json.dumps(result,indent=2));
    with (out/"history.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()

