"""Build a YOLO dataset tuned for blur / dim light / low quality / occlusion.

Reads the reviewed fall-posture annotations (schema v1, classes fallen/normal)
and emits two dataset folders next to each other:

<output>                 clean train/val/test originals plus, for train only,
                         deterministic degraded copies (blur, dim light, low
                         quality, obstacle occlusion).
<output>_eval_degraded   only degraded copies of the val and test records, for
                         measuring robustness of a detector before and after a
                         fine-tune.

Everything is deterministic for a given annotation file, source videos, and
seed, and every source is hash-verified exactly like build_posture_dataset.py.
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np


def seed_rng(*parts):
    """Deterministic rng seeded from the record identity, variant, and seed."""
    text = '|'.join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode('utf8')).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'big'))


def blur_copy(image, rng):
    """Blurry focus or sensor motion smear."""
    pick = rng.random()
    if pick < .4:
        # defocus: circular disc kernel
        size = rng.integers(5, 10) * 2 + 1
        kernel = np.zeros((size, size), 'float32')
        cv2.circle(kernel, (size // 2, size // 2), size // 2, 1.0, -1)
        kernel /= kernel.sum()
        return cv2.filter2D(image, -1, kernel)
    if pick < .75 :
        # sensor motion smear
        size = int(rng.integers(5, 14)) * 2 + 1
        kernel = np.zeros((size, size), 'float32')
        kernel[:, :] = 1.0 / size
        return cv2.filter2D(image, -1, kernel)
    return cv2.GaussianBlur(image, (0, 0), float(rng.uniform(1.1, 2.4)))


def dim_copy(image, rng):
    """Dim rooms, blown backlight, and big soft shadows across the frame."""
    pick = rng.random()
    if pick < .4:
        # dim exposure
        lut = (255.0 * (np.arange(256, dtype='float32') / 255.0) ** float(rng.uniform(.4, .62))).astype('uint8')
        return cv2.LUT(image, lut)
    if pick < .7:
        # backlit: blown highlights, weaker colours
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv[:, :, 2] = hsv[:, :, 2] * float(rng.uniform(1.5, 2.1))
        hsv[:, :, 1] = hsv[:, :, 1] * float(rng.uniform(.55, .85))
        hsv = np.clip(hsv, 0, 255).astype('uint8')
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # soft shadow band over one side of the frame
    height, width = image.shape[:2]
    mask = np.zeros((height, width), 'float32')
    band = max(1, round(height * float(rng.uniform(.2, .4))))
    if rng.random() < .5:
        mask[:band, :] = 1.0
    else:
        mask[height - band:, :] = 1.0
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, band / 4.0))
    strength = float(rng.uniform(.5, .72))
    return np.clip(image.astype('float32') * (1.0 - mask.astype('float32')[..., None] * strength), 0, 255).astype('uint8')


def grain_copy(image, rng):
    """Low-quality compression: pixelation, jpeg artefacts, sensor noise."""
    result = image
    if rng.random() < .8:
        factor = int(rng.integers(3, 9))
        height, width = result.shape[:2]
        small = cv2.resize(result, (max(1, width // factor), max(1, height // factor)), interpolation=cv2.INTER_LINEAR)
        result = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    encoded, buffer = cv2.imencode('.jpg', result, [int(cv2.IMWRITE_JPEG_QUALITY), int(rng.integers(8, 26))])
    if encoded:
        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if decoded is not None:
            result = decoded
    noise = rng.normal(0.0, float(rng.uniform(5.0, 12.0)), result.shape).astype('float32')
    return np.clip(result.astype('float32') + noise, 0, 255).astype('uint8')


def degraded_copy(image, kind, rng, box=None):
    from build_posture_dataset import occluded_copy
    if kind == 'blur':
        return blur_copy(image, rng)
    if kind == 'dim':
        return dim_copy(image, rng)
    if kind == 'grain':
        return grain_copy(image, rng)
    if kind == 'occluded':
        if box is None:
            raise ValueError('Occlusion needs the person box.')
        return occluded_copy(image, box, seed=int(rng.integers(0, 1 << 31)))
    raise ValueError(f'Unknown degradation kind: {kind}')


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_posture_dataset import label_text, sha256, validate_annotations

    parser = argparse.ArgumentParser(description='Build the blur/dim/low-quality/occlusion robustness dataset.')
    parser.add_argument('--annotations', default='data/annotations/occluded_postures_v1.json')
    parser.add_argument('--source-root', default='.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=20261024)
    parser.add_argument('--train-copies', type=int, default=1, help='degrade copies per kind for train')
    parser.add_argument('--eval-copies', type=int, default=1, help='degrade copies per kind for eval_degraded values')
    parser.add_argument('--occlusion-copies', type=int, default=1, help='extra occlusion copies on top of --train-copies')
    args = parser.parse_args()

    annotations, output = Path(args.annotations), Path(args.output).resolve()
    records = validate_annotations(json.loads(annotations.read_text(encoding='utf8')))
    for value, limit, name in ((args.train_copies, 3, 'train_copies'), (args.eval_copies, 3, 'eval_copies'), (args.occlusion_copies, 3, 'occlusion_copies')):
        if type(value) is not int or not 0 <= value <= limit:
            raise ValueError(f'{name} must be an integer from 0 to {limit}.')

    eval_output = Path(str(output) + '_eval_degraded')
    if output.exists() or eval_output.exists():
        raise ValueError('Dataset output already exists.')

    source_root = Path(args.source_root).resolve()
    sources = {}
    for record in records:
        path = (source_root / record['video']).resolve()
        if path.parent != source_root or not path.is_file():
            raise ValueError('Source video must be a local file in source_root.')
        if path not in sources:
            sources[path] = sha256(path)
        if sources[path] != record['source_sha256']:
            raise ValueError('Source video changed since annotation.')

    for root in (output, eval_output):
        root.mkdir(parents=True)
    for kind in ('images', 'labels'):
        for split in ('train', 'val', 'test'):
            (output / kind / split).mkdir(parents=True)
    for kind in ('images', 'labels'):
        for split in ('val', 'test'):
            (eval_output / kind / split).mkdir(parents=True)

    train_kinds = ('blur', 'dim', 'grain')
    eval_kinds = ('blur', 'dim', 'grain')
    manifest, captures = [], {}
    try:
        for record in records:
            video, split = record['video'], record['split']
            if video not in captures:
                captures[video] = cv2.VideoCapture(str(source_root / video), cv2.CAP_FFMPEG)
            capture = captures[video]
            capture.set(cv2.CAP_PROP_POS_FRAMES, record['frame_index'])
            ok, image = capture.read()
            if not ok or image is None:
                raise ValueError(f'Frame {record["frame_index"]} of {video} could not be read.')
            if image.shape[0] != record['height'] or image.shape[1] != record['width']:
                raise ValueError(f'Frame {record["frame_index"]} of {video} does not match annotated dimensions.')
            box = record['xyxy']
            base = record['id']
            stem = f'{base}.jpg'

            # clean original
            destination = output / 'images' / split / stem
            cv2.imwrite(str(destination), image, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
            manifest.append({**record, 'dataset': 'main', 'variant': 'clean',
                             'image': f'images/{split}/{stem}', 'labels': f'labels/{split}/'+(f'{base}.txt')})
            (output / 'labels' / split / f'{base}.txt').write_text(label_text(record), encoding='ascii')

            if split == 'train':
                rng_kinds = []
                if record['class_id'] is not None:
                    rng_kinds.extend(('occluded',) * args.occlusion_copies)
                rng_kinds.extend(kind for kind in train_kinds for _ in range(args.train_copies))
                for kind in rng_kinds:
                    variant = f'{base}_{kind}'
                    copy = degraded_copy(image, kind, seed_rng(record['id'], kind, args.seed), box)
                    target = output / 'images' / 'train' / f'{variant}.jpg'
                    cv2.imwrite(str(target), copy, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
                    manifest.append({**record, 'dataset': 'main', 'variant': kind,
                                     'image': f'images/train/{variant}.jpg', 'labels': f'labels/train/{variant}.txt'})
                    (output / 'labels' / 'train' / f'{variant}.txt').write_text(label_text(record), encoding='ascii')
            elif args.eval_copies > 0:
                rng_kinds = ('occluded',) * args.eval_copies if record['class_id'] is not None else ()
                rng_kinds = tuple(rng_kinds) + tuple(kind for kind in eval_kinds for _ in range(args.eval_copies))
                for kind in rng_kinds:
                    variant = f'{base}_{kind}'
                    copy = degraded_copy(image, kind, seed_rng(record['id'], kind, args.seed), box)
                    target = eval_output / 'images' / split / f'{variant}.jpg'
                    cv2.imwrite(str(target), copy, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
                    manifest.append({**record, 'dataset': 'eval_degraded', 'variant': kind,
                                     'image': f'images/{split}/{variant}.jpg', 'labels': f'labels/{split}/{variant}.txt'})
                    (eval_output / 'labels' / split / f'{variant}.txt').write_text(label_text(record), encoding='ascii')
    finally:
        for capture in captures.values():
            capture.release()

    names = json.loads(annotations.read_text(encoding='utf8'))['classes']
    (output / 'data.yaml').write_text(f'train: images/train\nval: images/val\ntest: images/test\nnames: {names}\n', encoding='utf8')
    (eval_output / 'data.yaml').write_text(f'train: images/val\nval: images/val\ntest: images/test\nnames: {names}\n', encoding='utf8')
    stats = {}
    for entry in manifest:
        stats[(entry['dataset'], entry['split'], entry['variant'])] = stats.get((entry['dataset'], entry['split'], entry['variant']), 0) + 1
    info = {'annotations_version': '1', 'classes': names, 'seed': args.seed,
            'train_copies': args.train_copies, 'occlusion_copies': args.occlusion_copies,
            'eval_copies': args.eval_copies, 'counts': {f'{dataset}/{split}/{variant}': count for (dataset, split, variant), count in sorted(stats.items())}}
    (output / 'manifest.json').write_text(json.dumps(info, indent=2, sort_keys=True) + '\n', encoding='utf8')
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
