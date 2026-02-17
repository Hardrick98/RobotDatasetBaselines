import json
import os
import shutil
from pathlib import Path


def convert_coco_to_yolo(coco_json_path, images_src_dir, output_dir, split_name):
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    img_out = Path(output_dir) / 'images' / split_name
    lbl_out = Path(output_dir) / 'labels' / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    images_info = {img['id']: img for img in coco['images']}

    img_to_anns = {}
    for ann in coco['annotations']:
        img_to_anns.setdefault(ann['image_id'], []).append(ann)

    cat_ids = sorted([c['id'] for c in coco['categories']])
    cat_id_to_yolo = {cid: idx for idx, cid in enumerate(cat_ids)}

    print(f"Categories ({len(cat_ids)}):")
    for c in coco['categories']:
        print(f"  {c['id']} -> {cat_id_to_yolo[c['id']]}: {c['name']}")

    linked = 0
    skipped = 0

    for img_id, img_info in images_info.items():
        file_name = img_info['file_name']
        img_w = img_info['width']
        img_h = img_info['height']

        src_path = Path(images_src_dir) / file_name
        if not src_path.exists():
            skipped += 1
            if skipped <= 5:
                print(f"  [SKIP] Image not found: {src_path}")
            continue

        flat_name = file_name.replace('/', '_').replace('\\', '_')
        stem = Path(flat_name).stem

        # Symlink instead of copy
        dst_img = img_out / flat_name
        if not dst_img.exists():
            os.symlink(src_path.resolve(), dst_img)

        # Convert annotations to YOLO format
        anns = img_to_anns.get(img_id, [])
        label_lines = []
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            cat_idx = cat_id_to_yolo[ann['category_id']]
            x, y, w, h = ann['bbox']

            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))

            label_lines.append(f"{cat_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        lbl_path = lbl_out / f"{stem}.txt"
        with open(lbl_path, 'w') as f:
            f.write('\n'.join(label_lines))

        linked += 1

    print(f"\n[{split_name}] Symlinked {linked} images, skipped {skipped}")
    return cat_ids, coco['categories']


def create_yaml(output_dir, categories, yaml_name='dataset.yaml'):
    cat_names = {idx: c['name'] for idx, c in enumerate(
        sorted(categories, key=lambda x: x['id'])
    )}

    yaml_content = f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
test: images/test

nc: {len(categories)}
names: {cat_names}
"""
    yaml_path = Path(output_dir) / yaml_name
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\nYAML config written to: {yaml_path}")
    return yaml_path


if __name__ == '__main__':
    DATASET_ROOT = './exo_dataset'
    YOLO_OUTPUT = './exo_dataset_yolo'

    SPLITS = {
        'train': os.path.join(DATASET_ROOT, 'train_coco_detection.json'),
        'val':   os.path.join(DATASET_ROOT, 'val_coco_detection.json'),
        'test':  os.path.join(DATASET_ROOT, 'test_coco_detection.json'),
    }

    for split in ['train', 'val', 'test']:
        for sub in ['images', 'labels']:
            p = Path(YOLO_OUTPUT) / sub / split
            if p.exists():
                shutil.rmtree(p)

    categories = None
    for split_name, json_path in SPLITS.items():
        if not os.path.exists(json_path):
            print(f"[WARN] {json_path} not found, skipping {split_name}")
            continue
        _, cats = convert_coco_to_yolo(
            coco_json_path=json_path,
            images_src_dir=DATASET_ROOT,
            output_dir=YOLO_OUTPUT,
            split_name=split_name
        )
        categories = cats

    if categories:
        yaml_path = create_yaml(YOLO_OUTPUT, categories)
        print(f"\nDone! Train with:")
        print(f"  yolo detect train data={yaml_path} model=yolov8n.pt epochs=100 imgsz=640")