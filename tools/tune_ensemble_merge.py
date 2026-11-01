"""Tune ensemble merge params on the robustness val/test images.

Loads each member once, sweeps conf-floor per member + per-member weights,
reports fallen-class metrics. Image-level (no temporal engine).
"""
import argparse, json
from pathlib import Path
import numpy as np
from ultralytics import YOLO

IOU = 0.5
FALLEN = 0

def iou(a, b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    ua=(a[2]-a[0])*(a[3]-a[1]); ub=(b[2]-b[0])*(b[3]-b[1])
    return inter/(ua+ub-inter+1e-9)

def nms(boxes, scores, thr=0.5):
    if len(boxes)==0: return np.zeros(0,dtype=int)
    order=np.argsort(scores)[::-1]; keep=[]
    while order.size:
        i=order[0]; keep.append(i)
        rest=order[1:]
        ious=np.array([iou(boxes[i],boxes[j]) for j in rest])
        order=rest[ious<=thr]
    return np.array(keep)

def predict_one(model, path, imgsz, conf_floor, device):
    r=model.predict(str(path), imgsz=imgsz, conf=conf_floor, device=device, verbose=False)
    boxes=[]; classes=[]; scores=[]
    if r and r[0].boxes is not None:
        for b in r[0].boxes:
            boxes.append(b.xyxy[0].cpu().numpy()); classes.append(int(b.cls[0])); scores.append(float(b.conf[0]))
    if not boxes: return np.zeros((0,4)),np.zeros(0,dtype=int),np.zeros(0)
    boxes=np.array(boxes); classes=np.array(classes); scores=np.array(scores)
    keep=nms(boxes, scores, 0.5)
    return boxes[keep], classes[keep], scores[keep]

def make_members(weights, imgsz, device):
    out=[]
    for w in weights:
        m=YOLO(str(w)); m.to(device)
        out.append(m)
    return out

def merge_avg(member_preds, weights, floors):
    # member_preds: list of (boxes,classes,scores) at conf=0.01
    # conf floors are applied BEFORE NMS: a box only becomes a candidate if the member
    # that produced it passes that member's floor
    cand=[]
    for mi,(b,c,s) in enumerate(member_preds):
        for j in range(len(b)):
            if s[j] < floors[mi]:
                continue
            cand.append((b[j], c[j], s[j], mi))
    if not cand: return np.zeros((0,4)),np.zeros(0,dtype=int),np.zeros(0)
    cand_boxes=np.array([x[0] for x in cand]); cand_cls=np.array([x[1] for x in cand])
    cand_sc=np.array([x[2] for x in cand]); cand_mi=np.array([x[3] for x in cand])
    # agnostic NMS on raw union
    order=np.argsort(cand_sc)[::-1]; keep=[]
    while order.size:
        i=order[0]; keep.append(i)
        rest=order[1:]
        ious=np.array([iou(cand_boxes[i],cand_boxes[j]) for j in rest])
        order=rest[ious<=0.5]
    keep=np.array(keep)
    kb=cand_boxes[keep]; kc=cand_cls[keep]; ks=cand_sc[keep]
    # per surviving box, compute weighted fallen_label across members that overlap
    out_boxes=[]; out_cls=[]; out_sc=[]
    for i in range(len(kb)):
        # members overlapping this box
        wsum=0.0; wtot=0.0; cls=kc[i]
        for mi in range(len(member_preds)):
            b,c,s=member_preds[mi]
            # find best overlap box in member mi
            best=-1; bo=0.0
            for j in range(len(b)):
                o=iou(kb[i],b[j])
                if o>bo: bo=o; best=j
            if best>=0 and bo>=0.5:
                # member saw this object; fallen prob
                mem_fallen = 1.0 if c[best]==FALLEN else 0.0
                # use member's score as confidence gate
                mem_conf = float(s[best])
                # conf floor
                if mem_conf < floors[mi]: 
                    mem_fallen = 0.0
                w = weights[mi]
                wsum += w*mem_fallen*mem_conf
                wtot += w*mem_conf
        if wtot>0:
            fused = wsum/wtot
        else:
            fused = 0.0 if cls!=FALLEN else float(ks[i])
        # class: fallen if fused >= 0.5 else normal; score = fused score for fallen
        # box is kept only if at least one member saw it (guaranteed by construction)
        if cls==FALLEN:
            if fused>=0.5:
                final_cls = FALLEN
                final_sc = fused
            else:
                final_cls = 1
                final_sc = float(ks[i])
        else:
            final_cls = 1
            final_sc = float(ks[i])
        out_boxes.append(kb[i]); out_cls.append(final_cls); out_sc.append(final_sc)
    return np.array(out_boxes), np.array(out_cls), np.array(out_sc)

def load_split(dataset, split):
    # dataset is robustness_v1_trainval-style: images + labels under split
    root=Path(dataset)
    imgdir=root/'images' if (root/'images').exists() else root
    # support yolo layout: dataset/<split>/images + labels
    if (root/split/'images').exists():
        imgdir=root/split/'images'; labdir=root/split/'labels'
    elif (root/'images').exists():
        imgdir=root/'images'; labdir=root/'labels'
    else:
        raise SystemExit('bad dataset layout')
    pairs=[]
    for ip in sorted(imgdir.glob('*.jpg'))+sorted(imgdir.glob('*.png')):
        lp=labdir/(ip.stem+'.txt')
        boxes=[]
        if lp.exists():
            for line in lp.read_text().split():
                pass
            for line in lp.read_text().splitlines():
                p=line.split()
                if not p: continue
                cls=int(p[0]); cx,cy,w,h=map(float,p[1:5])
                # yolo normalized -> pixels using image size
                import cv2
                im=cv2.imread(str(ip)); H,W=im.shape[:2]
                x1=(cx-w/2)*W; y1=(cy-h/2)*H; x2=(cx+w/2)*W; y2=(cy+h/2)*H
                boxes.append([cls,x1,y1,x2,y2])
        pairs.append((ip, boxes))
    return pairs

def evaluate_config(members, floors, weights, dataset, imgsz, device):
    import cv2
    root=Path(dataset)
    pairs=[]
    # robustness_v1_trainval layout: <root>/images/*.jpg + <root>/labels/*.txt
    for ip in sorted(root.glob('images/*.jpg'))+sorted(root.glob('images/*.png')):
        lp=root/'labels'/(ip.stem+'.txt')
        im=cv2.imread(str(ip)); H,W=im.shape[:2]
        boxes=[]
        if lp.exists():
            for line in lp.read_text().splitlines():
                p=line.split()
                if not p: continue
                cls=int(float(p[0])); cx,cy,w,h=map(float,p[1:5])
                x1=(cx-w/2)*W; y1=(cy-h/2)*H; x2=(cx+w/2)*W; y2=(cy+h/2)*H
                boxes.append([cls,x1,y1,x2,y2])
        pairs.append((ip,boxes))
    tp=fp=gt=0
    for ip, boxes in pairs:
        preds=[predict_one(m, ip, imgsz, 0.01, device) for m in members]
        # apply conf floors inside merge by passing floors separately
        arrs,clss,scs=merge_avg(preds, weights, floors)
        matched=set()
        for label in boxes:
            if label[0]!=0:
                continue
            gt+=1
            best=None;bo=0.0
            for pos,box in enumerate(arrs):
                if clss[pos]!=FALLEN or pos in matched: continue
                o=iou(box,label[1:])
                if o>bo: bo=o; best=pos
            if best is not None and bo>=IOU:
                tp+=1; matched.add(best)
        for pos in range(len(arrs)):
            if pos in matched: continue
            if clss[pos]==FALLEN:
                # false positive if not overlapping any fallen gt
                overlapped=any(iou(arrs[pos],l[1:])>=IOU for l in boxes if l[0]==0)
                if not overlapped: fp+=1
    prec=tp/(tp+fp) if tp+fp else 0.0
    rec=tp/gt if gt else 0.0
    f1=2*prec*rec/(prec+rec) if prec+rec else 0.0
    return tp,fp,gt,prec,rec,f1

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--imgsz', type=int, default=416)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--json-out')
    args=ap.parse_args()
    weights=[w.strip() for w in args.weights.split(',')]
    members=make_members(weights, args.imgsz, args.device)
    import cv2
    root=Path(args.dataset)
    # layout: <root>/images/val/*.jpg + <root>/labels/val/*.txt (YOLO split dirs)
    split='val' if (root/'images'/'val').exists() else ''
    pairs=[]
    for ip in sorted((root/'images'/split).glob('*.jpg')) if split else sorted((root/'images').glob('*.jpg')):
        lp=(root/'labels'/split/(ip.stem+'.txt')) if split else (root/'labels'/(ip.stem+'.txt'))
        im=cv2.imread(str(ip)); H,W=im.shape[:2]
        boxes=[]
        if lp.exists():
            for line in lp.read_text().splitlines():
                p=line.split()
                if not p: continue
                cls=int(float(p[0])); cx,cy,w,h=map(float,p[1:5])
                x1=(cx-w/2)*W; y1=(cy-h/2)*H; x2=(cx+w/2)*W; y2=(cy+h/2)*H
                boxes.append([cls,x1,y1,x2,y2])
        pairs.append((ip,boxes))
    print('images:', len(pairs), 'fallen gt:', sum(1 for _,b in pairs for l in b if l[0]==0))
    # predict each member ONCE per image (conf=0.01 keeps everything); cache
    cache=[]
    for ip,_ in pairs:
        cache.append([predict_one(m, ip, args.imgsz, 0.01, args.device) for m in members])
    floors_grid=[0.05,0.1,0.2,0.3,0.4]
    w_grid=[0.0,0.5,1.0]
    results=[]
    for f0 in floors_grid:
        for f1 in floors_grid:
            for f2 in floors_grid:
                floors=[f0,f1,f2]
                for w0 in w_grid:
                    for w1 in w_grid:
                        for w2 in w_grid:
                            ww=[w0,w1,w2]
                            tp=fp=gt=0
                            for (ip,boxes),mem in zip(pairs, cache):
                                arrs,clss,scs=merge_avg(mem, ww, floors)
                                matched=set()
                                for label in boxes:
                                    if label[0]!=0: continue
                                    gt+=1
                                    best=None;bo=0.0
                                    for pos,box in enumerate(arrs):
                                        if clss[pos]!=FALLEN or pos in matched: continue
                                        o=iou(box,label[1:])
                                        if o>bo: bo=o; best=pos
                                    if best is not None and bo>=IOU:
                                        tp+=1; matched.add(best)
                                for pos in range(len(arrs)):
                                    if pos in matched: continue
                                    if clss[pos]==FALLEN:
                                        overlapped=any(iou(arrs[pos],l[1:])>=IOU for l in boxes if l[0]==0)
                                        if not overlapped: fp+=1
                            prec=tp/(tp+fp) if tp+fp else 0.0
                            rec=tp/gt if gt else 0.0
                            f1m=2*prec*rec/(prec+rec) if prec+rec else 0.0
                            results.append({'floors':floors,'weights':ww,'tp':tp,'fp':fp,'gt':gt,'prec':round(prec,3),'rec':round(rec,3),'f1':round(f1m,3)})
    results.sort(key=lambda r:(-r['f1'], r['fp'], -r['tp']))
    print('top 20 by f1:')
    for r in results[:20]:
        print(r)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results,indent=2),encoding='utf8')

if __name__=='__main__':
    main()
