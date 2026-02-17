import json
import os
import shutil
from pathlib import Path


def convert_coco_to_yolo_pose(coco_json_path, images_src_dir, output_dir, split_name):
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    img_out = Path(output_dir) / 'images' / split_name
    lbl_out = Path(output_dir) / 'labels' / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    images_info = {img['id']: img for img in coco['images']}

    print(f"Original categories (all mapped to class 0 'robot'):")
    for c in coco['categories']:
        print(f"  {c['id']}: {c['name']} -> 0: robot")

    img_to_anns = {}
    for ann in coco['annotations']:
        img_to_anns.setdefault(ann['image_id'], []).append(ann)

    # Determine number of keypoints from first annotation that has them
    num_kpts = 0
    for ann in coco['annotations']:
        if 'keypoints' in ann and len(ann['keypoints']) > 0:
            num_kpts = len(ann['keypoints']) // 3
            break

    print(f"Keypoints per instance: {num_kpts}")

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

        # Convert annotations to YOLO pose format
        anns = img_to_anns.get(img_id, [])
        label_lines = []
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue

            # All categories -> class 0 (robot)
            cat_idx = 0

            # Bounding box: COCO [x, y, w, h] -> YOLO [x_center, y_center, w, h] normalized
            x, y, w, h = ann['bbox']
            x_center = max(0, min(1, (x + w / 2) / img_w))
            y_center = max(0, min(1, (y + h / 2) / img_h))
            w_norm = max(0, min(1, w / img_w))
            h_norm = max(0, min(1, h / img_h))

            # Keypoints: COCO [x1, y1, v1, x2, y2, v2, ...] -> YOLO [x1, y1, v1, ...] normalized
            parts = [f"{cat_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"]

            kpts = ann.get('keypoints', [])
            for i in range(0, len(kpts), 3):
                kx = kpts[i] / img_w
                ky = kpts[i + 1] / img_h
                kv = int(kpts[i + 2])

                kx = max(0, min(1, kx))
                ky = max(0, min(1, ky))
                parts.append(f"{kx:.6f} {ky:.6f} {kv}")

            label_lines.append(' '.join(parts))

        lbl_path = lbl_out / f"{stem}.txt"
        with open(lbl_path, 'w') as f:
            f.write('\n'.join(label_lines))

        linked += 1

    print(f"\n[{split_name}] Symlinked {linked} images, skipped {skipped}")
    return coco['categories'], num_kpts


def create_yaml(output_dir, categories, num_kpts, yaml_name='dataset.yaml'):
    yaml_content = f"path: {os.path.abspath(output_dir)}\n"
    yaml_content += "train: images/train\n"
    yaml_content += "val: images/val\n"
    yaml_content += "test: images/test\n\n"

    yaml_content += f"kpt_shape: [{num_kpts}, 3]\n"

    # Try to get flip_idx from category info
    for c in categories:
        if 'flip_idx' in c:
            yaml_content += f"flip_idx: {c['flip_idx']}\n"
            break

    yaml_content += f"\nnc: 1\n"
    yaml_content += f"names:\n  0: robot\n"

    # Keypoint names if available
    for c in categories:
        if 'keypoints' in c:
            yaml_content += f"\n# Keypoint names: {c['keypoints']}\n"
        if 'skeleton' in c:
            skeleton = [[s[0] - 1, s[1] - 1] for s in c['skeleton']]
            yaml_content += f"# Skeleton: {skeleton}\n"
        break  # only need info from first category

    yaml_path = Path(output_dir) / yaml_name
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\nYAML config written to: {yaml_path}")
    return yaml_path


if __name__ == '__main__':
    DATASET_ROOT = './exo_dataset'
    YOLO_OUTPUT = './exo_dataset_yolo_pose'

    SPLITS = {
        'train': os.path.join(DATASET_ROOT, 'train_coco_pose.json'),
        'val':   os.path.join(DATASET_ROOT, 'val_coco_pose.json'),
        'test':  os.path.join(DATASET_ROOT, 'test_coco_pose.json'),
    }

    for split in ['train', 'val', 'test']:
        for sub in ['images', 'labels']:
            p = Path(YOLO_OUTPUT) / sub / split
            if p.exists():
                shutil.rmtree(p)

    categories = None
    num_kpts = 0
    for split_name, json_path in SPLITS.items():
        if not os.path.exists(json_path):
            print(f"[WARN] {json_path} not found, skipping {split_name}")
            continue
        cats, nk = convert_coco_to_yolo_pose(
            coco_json_path=json_path,
            images_src_dir=DATASET_ROOT,
            output_dir=YOLO_OUTPUT,
            split_name=split_name
        )
        if cats:
            categories = cats
            num_kpts = nk

    if categories:
        yaml_path = create_yaml(YOLO_OUTPUT, categories, num_kpts)
        print(f"\nDone! Train with:")
        print(f"  yolo pose train data={yaml_path} model=yolov8n-pose.pt epochs=100 imgsz=640")