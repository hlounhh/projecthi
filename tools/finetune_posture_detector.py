"""Fine-tune the deployed fall detector for blur / dim / low-quality / occlusion.

Starts from the pinned deployed weights, trains on the robustness dataset
(see build_robustness_dataset.py), then republishes the best checkpoint and an
ONNX export under models/fall-finetuned-v1/.
"""

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description='Fine-tune the fall detector for degraded footage.')
    parser.add_argument('--weights', default=r'models\human-fall-detection\97261b03632a9715a27481b2cfdac6a78e26d2d9\best.pt')
    parser.add_argument('--data', default=r'data\datasets\robustness_v1\data.yaml')
    parser.add_argument('--project', default=r'data\eval\finetune')
    parser.add_argument('--name', default='robust_v1')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--imgsz', type=int, default=416)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--lr0', type=float, default=0.0015)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    model = YOLO(str(PROJECT / args.weights))
    model.train(
        data=str(PROJECT / args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer='AdamW',
        lr0=args.lr0,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        patience=args.patience,
        seed=args.seed,
        verbose=False,
        project=str(PROJECT / args.project),
        name=args.name,
    )

    run_dir = Path(model.trainer.save_dir)
    trained = YOLO(str(run_dir / 'weights' / 'best.pt'))
    metrics = trained.val(data=str(PROJECT / args.data), imgsz=args.imgsz, device=args.device, verbose=False)

    publish = PROJECT / 'models' / 'fall-finetuned-v1'
    publish.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_dir / 'weights' / 'best.pt', publish / 'best.pt')
    onnx_path = trained.export(format='onnx', imgsz=args.imgsz, device=args.device)
    exported = Path(onnx_path)
    shutil.copyfile(exported, publish / exported.name)

    provenance = {
        'source_weights': str(PROJECT / args.weights),
        'dataset': str(PROJECT / args.data),
        'run_dir': str(run_dir),
        'epochs_requested': args.epochs,
        'epochs_done': getattr(getattr(model, 'trainer', None), 'epoch', None),
        'imgsz': args.imgsz,
        'batch': args.batch,
        'optimizer': 'AdamW',
        'lr0': args.lr0,
        'seed': args.seed,
        'published_best': str(publish / 'best.pt'),
        'published_onnx': str(publish / exported.name),
    }
    results_csv = run_dir / 'results.csv'
    if results_csv.exists():
        provenance['results_csv'] = str(results_csv)
        rows = results_csv.read_text(encoding='utf8').strip().splitlines()
        provenance['last_row'] = rows[-1]
    (publish / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'published': str(publish), 'map50': getattr(metrics, 'box', None) and round(metrics.box.map50, 4), 'export': str(publish / exported.name)}, indent=2))


if __name__ == '__main__':
    main()
