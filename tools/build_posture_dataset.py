"""Build a scene-separated YOLO dataset from reviewed image annotations."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_annotations(manifest):
    if manifest.get('schema_version') != 1 or manifest.get('classes') != ['fallen', 'normal']:
        raise ValueError('Expected the reviewed binary posture schema.')
    records = manifest.get('records', [])
    if not records:
        raise ValueError('No reviewed annotations.')
    scenes, videos, frames, identifiers, source_hashes = {}, {}, set(), set(), {}
    counts = Counter()
    for record in records:
        split, scene, video = record['split'], record['scene'], record['video']
        if split not in ('train', 'val', 'test') or not scene or Path(video).name != video:
            raise ValueError('Invalid split, scene or local video name.')
        if record.get('reviewed') is not True:
            raise ValueError('Every annotation must be visually reviewed.')
        for groups, key in ((scenes, scene), (videos, video), (source_hashes, record['source_sha256'])):
            if key in groups and groups[key] != split:
                raise ValueError('A scene, video or source hash crosses dataset partitions.')
            groups[key] = split
        identity = (record['source_sha256'], record['frame_index'])
        if identity in frames or record['id'] in identifiers:
            raise ValueError('Duplicate frame or annotation ID.')
        frames.add(identity)
        identifiers.add(record['id'])
        if type(record['frame_index']) is not int or record['frame_index'] < 0:
            raise ValueError('Invalid source frame index.')
        width, height, box = record['width'], record['height'], record['xyxy']
        if type(width) is not int or type(height) is not int or min(width, height) <= 0:
            raise ValueError('Invalid source dimensions.')
        if record['class_id'] is None:
            if box is not None:
                raise ValueError('Empty frames cannot contain a person box.')
        else:
            if type(record['class_id']) is not int or record['class_id'] not in (0, 1):
                raise ValueError('Invalid posture class.')
            if (not isinstance(box, list) or len(box) != 4
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)
                    or not 0 <= box[0] < box[2] <= width
                    or not 0 <= box[1] < box[3] <= height):
                raise ValueError('Person box must lie inside its source image.')
            counts[(split, record['class_id'])] += 1
    if any(counts[(split, label)] == 0 for split in ('train', 'val', 'test') for label in (0, 1)):
        raise ValueError('Each partition needs positive and negative person examples.')
    return records


def label_text(record):
    if record['class_id'] is None:
        return ''
    x1, y1, x2, y2 = record['xyxy']
    width, height = record['width'], record['height']
    return (f'{record["class_id"]} {(x1+x2)/2/width:.7f} {(y1+y2)/2/height:.7f} '
            f'{(x2-x1)/width:.7f} {(y2-y1)/height:.7f}\n')


def occluded_copy(image, box, seed):
    import numpy as np

    result = image.copy()
    rng = random.Random(seed)
    x1, y1, x2, y2 = map(round, box)
    width, height = x2 - x1, y2 - y1
    cover = max(1, round(height * rng.uniform(.12, .25)))
    top = rng.randint(y1 + round(height * .25), max(y1 + round(height * .25), y2 - cover))
    patch = image[max(0, y1 - 12):max(1, y1), max(0, x1 - 12):min(image.shape[1], x2 + 12)]
    color = np.median(patch.reshape(-1, 3), axis=0).astype('uint8') if patch.size else np.array([110]*3, dtype='uint8')
    result[top:top+cover, x1:x2] = color
    return result


def build_dataset(annotations, source_root, output, occlusion_copies=1):
    import cv2
    import yaml

    annotations, source_root, output = Path(annotations), Path(source_root).resolve(), Path(output).resolve()
    manifest = json.loads(annotations.read_text(encoding='utf8'))
    records = validate_annotations(manifest)
    if type(occlusion_copies) is not int or not 0 <= occlusion_copies <= 3:
        raise ValueError('Use zero to three training occlusion copies.')
    if output.exists():
        raise ValueError('Dataset output already exists.')
    sources = {}
    for record in records:
        path = (source_root / record['video']).resolve()
        if path.parent != source_root or not path.is_file():
            raise ValueError('Source video must be a local file in source_root.')
        if path not in sources:
            sources[path] = sha256(path)
        if sources[path] != record['source_sha256']:
            raise ValueError('Source video changed since annotation.')
    output.mkdir(parents=True)
    for kind in ('images', 'labels'):
        for split in ('train', 'val', 'test'):
            (output/kind/split).mkdir(parents=True)
    captures, generated = {}, []
    try:
        for record in records:
            video, split = record['video'], record['split']
            if video not in captures:
                captures[video] = cv2.VideoCapture(str(source_root/video), cv2.CAP_FFMPEG)
            cap = captures[video]
            cap.set(cv2.CAP_PROP_POS_FRAMES, record['frame_index'])
            ok, image = cap.read()
            if not ok or image.shape[:2] != (record['height'], record['width']):
                raise ValueError('Source frame cannot be decoded at annotated dimensions.')
            key = f'{Path(video).stem}_{record["frame_index"]:06}_{record["id"]:03}'
            variants = [(key, image, False)]
            if split == 'train' and record['xyxy'] is not None:
                variants.extend((f'{key}_occluded_{i}', occluded_copy(image, record['xyxy'], record['id']*11+i), True)
                                for i in range(occlusion_copies))
            for name, pixels, synthetic in variants:
                image_path = output/'images'/split/f'{name}.jpg'
                if not cv2.imwrite(str(image_path), pixels, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                    raise ValueError('Cannot write dataset image.')
                (output/'labels'/split/f'{name}.txt').write_text(label_text(record), encoding='utf8')
                generated.append(dict(id=record['id'], image=str(image_path.relative_to(output)).replace('\\','/'),
                                      sha256=sha256(image_path), synthetic=synthetic, split=split,
                                      scene=record['scene'], class_id=record['class_id'], xyxy=record['xyxy']))
    finally:
        for cap in captures.values():
            cap.release()
    yaml_path = output/'dataset.yaml'
    yaml_path.write_text(yaml.safe_dump(dict(path=str(output), train='images/train', val='images/val',
                                           test='images/test', names={0:'fallen', 1:'normal'})), encoding='utf8')
    (output/'dataset_manifest.json').write_text(json.dumps(dict(schema_version=1,
        annotations_sha256=sha256(annotations), split_policy=manifest['split_policy'],
        records=records, images=generated), ensure_ascii=False, indent=2), encoding='utf8')
    return dict(path=str(yaml_path), images=dict(Counter(item['split'] for item in generated)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--occlusion-copies', type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.annotations, args.source_root, args.output, args.occlusion_copies)))


if __name__ == '__main__':
    main()