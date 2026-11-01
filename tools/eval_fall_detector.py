"""Evaluate a fall detector against reviewed YOLO labels.

Works with any ultralytics checkpoint (.pt) or exported model. Ground truth is
read from YOLO label files in a dataset folder. Predictions of the fallen class
are compared against fallen ground truth, every remaining person class counts
as a normal-person prediction. Matching is greedy IoU >= 0.5 per image, and the
reported metric is interpolated AP50 plus precision and recall.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IOU = 0.5


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / area if area > 0 else 0.0


def load_split(dataset, split):
    dataset = Path(dataset).resolve()
    lines = (dataset / 'data.yaml').read_text(encoding='utf8').splitlines()
    names = next(eval(line.split('names:', 1)[1].strip()) for line in lines if line.startswith('names:'))
    folder = next(dataset / line.split(':', 1)[1].strip() for line in lines if line.startswith(split + ':'))
    pairs = []
    for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')):
        label_file = dataset / 'labels' / split / (path.stem + '.txt')
        boxes = []
        if label_file.exists():
            height, width = cv2.imread(str(path)).shape[:2]
            for line in label_file.read_text(encoding='ascii').splitlines():
                if not line.strip():
                    continue
                cls, cx, cy, w, h = (float(value) for value in line.split())
                boxes.append((int(cls), (cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height))
        pairs.append((path, boxes))
    return names, pairs


def average_precision(tp_pairs, fp_pairs, gt_total):
    """Interpolated VOC-style AP50 over all confidence levels."""
    if gt_total == 0:
        return float('nan')
    candidates = sorted(tp_pairs + fp_pairs, key=lambda item: -item[0])
    tp, seen, points = 0, 0, []
    for _, matched in candidates:
        seen += 1
        if matched == 'tp':
            tp += 1
            points.append((tp / gt_total, tp / seen))
    best = 0.0
    for index in range(len(points) - 1, -1, -1):
        best = max(best, points[index][1])
        points[index] = (points[index][0], best)
    area, previous = 0.0, 0.0
    for recall, precision in points:
        area += (recall - previous) * precision
        previous = recall
    return area


def nms_classwise(boxes, classes, scores, threshold=IOU):
    """Mimics the runtime merge: agnostic NMS across classes and members."""
    order = np.argsort(-scores)
    keep = []
    for position in order:
        if all(iou(boxes[position], boxes[other]) < threshold for other, _ in keep):
            keep.append((position, scores[position]))
    keep.sort(key=lambda item: item[0])
    if not keep:
        return np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros(0)
    return (np.stack([boxes[kept] for kept, _ in keep]),
            np.array([classes[kept] for kept, _ in keep], dtype=int),
            np.array([score for _, score in keep]))


def run_member(model, path, imgsz, conf_floor, device):
    result = model.predict(str(path), imgsz=imgsz, conf=conf_floor, iou=IOU,
                           device=device, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.zeros((0, 4), dtype='float32')
    classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else np.zeros(0, dtype=int)
    scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.zeros(0)
    return boxes, classes, scores


def make_detector(weights, merge, imgsz, conf_floor, device):
    models = [YOLO(str(path)) for path in weights]
    fallen_index = next((index for index, name in models[0].names.items()
                         if 'fallen' in str(name).lower()), None)
    if fallen_index is None:
        raise ValueError('No fallen class in model names: ' + str(models[0].names))

    def group_of(cls):
        return 0 if int(cls) == fallen_index else 1

    if merge == 'single':
        def predict(path):
            return run_member(models[0], path, imgsz, conf_floor, device)
    elif merge == 'union':
        def predict(path):
            members = [run_member(model, path, imgsz, conf_floor, device) for model in models]
            boxes = np.concatenate([member[0] for member in members] or [np.zeros((0, 4))])
            classes = np.concatenate([member[1] for member in members] or [np.zeros(0, dtype=int)])
            scores = np.concatenate([member[2] for member in members] or [np.zeros(0)])
            return nms_classwise(boxes, classes, scores)
    elif merge == 'cascade':
        primary, fallback = models[0], models[1]

        def predict(path):
            boxes, classes, scores = run_member(primary, path, imgsz, conf_floor, device)
            # fall back when the precision-tuned member reports no fall at all
            if not any((classes == fallen_index) & (scores > 0.0)):
                boxes2, classes2, scores2 = run_member(fallback, path, imgsz, conf_floor, device)
                keep2 = classes2 == fallen_index
                boxes, classes, scores = np.concatenate((boxes, boxes2[keep2]), axis=0), \
                    np.concatenate((classes, classes2[keep2])), np.concatenate((scores, scores2[keep2]))
                boxes, classes, scores = nms_classwise(boxes, classes, scores)
            return boxes, classes, scores
    else:
        raise ValueError('Unknown merge mode: ' + merge)

    def group_names(path):
        return predict(path)

    return group_names, group_of


def evaluate(weights, dataset, splits, imgsz, conf_floor, device, merge='single'):
    predict, group_of = make_detector(weights, merge, imgsz, conf_floor, device)
    summary = {'weights': [str(w) for w in weights], 'merge': merge, 'dataset': str(dataset),
               'iou': IOU, 'conf_floor': conf_floor, 'imgsz': imgsz, 'device': device}
    for split in splits:
        names, pairs = load_split(dataset, split)
        tp_pairs = ([], [])  # index 0 fallen, 1 normal
        fp_pairs = ([], [])
        missed = ([], [])
        gt = [0, 0]
        images = 0
        for path, boxes in pairs:
            images += 1
            arrays, classes, scores = predict(path)
            matched = set()
            for label in boxes:
                group = 0 if label[0] == 0 else 1
                gt[group] += 1
                best, best_overlap = None, 0.0
                for position, box in enumerate(arrays):
                    if group_of(classes[position]) != group or position in matched:
                        continue
                    overlap = iou(box, label[1:])
                    if overlap > best_overlap:
                        best, best_overlap = position, overlap
                if best is not None and best_overlap >= IOU:
                    matched.add(best)
                    tp_pairs[group].append((float(scores[best]), 'tp'))
                else:
                    missed[group].append({'image': path.name, 'best_iou': round(float(best_overlap), 3)})
            # count each unmatched prediction exactly once; anything overlapping a
            # ground-truth bubble of the same group is a duplicate localization,
            # not a false alarm
            for position, score in enumerate(scores):
                if position in matched:
                    continue
                group = group_of(classes[position])
                overlapped = any(iou(arrays[position], label[1:]) >= IOU for label in boxes
                                 if (0 if label[0] == 0 else 1) == group)
                if overlapped:
                    continue
                fp_pairs[group].append((float(score), 'fp'))
        per_class = {}
        for group, name in ((0, 'fallen'), (1, 'normal')):
            tp, fp = len(tp_pairs[group]), len(fp_pairs[group])
            ap50 = average_precision(tp_pairs[group], fp_pairs[group], gt[group]) if gt[group] else None
            per_class[name] = {
                'gt': gt[group], 'tp': tp, 'fp': fp, 'fn': gt[group] - tp,
                'precision': round(tp / (tp + fp), 4) if tp + fp else None,
                'recall': round(tp / gt[group], 4) if gt[group] else None,
                'ap50': round(ap50, 4) if ap50 is not None and ap50 == ap50 else None,
            }
        summary[split] = {'images': images, 'fallen': per_class['fallen'], 'normal': per_class['normal'],
                          'missed_fallen': missed[0], 'missed_normal': missed[1]}
        fallen = per_class['fallen']
        weights_name = '+'.join(Path(w).parts[-2] for w in weights)
        print(f"{weights_name} [{merge}] {Path(dataset).name}/{split}: fallen P={fallen['precision']} "
              f"R={fallen['recall']} AP50={fallen['ap50']} (gt {fallen['gt']}, tp {fallen['tp']}, fp {fallen['fp']})", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description='Fall detector robustness evaluator.')
    parser.add_argument('--weights', required=True, help='one weight path, or comma-separated two weights')
    parser.add_argument('--merge', default='single', choices=['single', 'union', 'cascade'],
                        help='single = one model; union = both models pooled + class NMS; '
                             'cascade = primary model, fall back to second when primary reports no fallen box')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--splits', default='val,test')
    parser.add_argument('--imgsz', type=int, default=416)
    parser.add_argument('--conf-floor', type=float, default=0.05)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--json-out')
    args = parser.parse_args()
    splits = [item.strip() for item in args.splits.split(',') if item.strip()]
    weights = [item.strip() for item in args.weights.split(',') if item.strip()]
    summary = evaluate(weights, args.dataset, splits, args.imgsz, args.conf_floor, args.device, merge=args.merge)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
