import os.path as osp
from collections import defaultdict
from typing import List, Tuple

import numpy as np
from mmengine.fileio import get_local_path
from xtcocotools.coco import COCO

from mmpose.datasets.datasets import BaseMocapDataset
from mmpose.registry import DATASETS


@DATASETS.register_module()
class RobotDataset3D(BaseMocapDataset):
    
    
    def __init__(self,
                 multiple_target: int = 0,
                 multiple_target_step: int = 0,
                 seq_step: int = 1,
                 pad_video_seq: bool = False,
                 **kwargs):
        self.seq_step = seq_step
        self.pad_video_seq = pad_video_seq

        if multiple_target > 0 and multiple_target_step == 0:
            multiple_target_step = multiple_target
        self.multiple_target_step = multiple_target_step

        super().__init__(multiple_target=multiple_target, **kwargs)

    METAINFO: dict = dict(from_file='configs/_base_/datasets/robot_dataset.py')

    def _load_ann_file(self, ann_file: str) -> dict:
        """Load annotation file."""
        with get_local_path(ann_file) as local_path:
            self.ann_data = COCO(local_path)

    def get_sequence_indices(self) -> List[List[int]]:
        img_ids = self.ann_data.getImgIds()
        sequence_indices = []

        for img_id in img_ids:
            ann_ids = self.ann_data.getAnnIds(imgIds=[img_id])
            if len(ann_ids) > 0:
                sequence_indices.append(ann_ids)

        # Applica subset_frac se necessario
        subset_size = int(len(sequence_indices) * self.subset_frac)
        start = np.random.randint(0, len(sequence_indices) - subset_size + 1)
        end = start + subset_size
        sequence_indices = sequence_indices[start:end]

        return sequence_indices


    def _parse_image_name(self, image_path: str) -> Tuple[str, int]:
        """Parse image name to get video name and frame index.

        Args:
            image_name (str): Image name.

        Returns:
            tuple[str, int]: Video name and frame index.
        """
        print(image_path)
        trim, file_name = image_path.split('/')[-2:]
        frame_id, suffix = file_name.split('.')
        return trim, frame_id, suffix

    def _load_annotations(self):
        """Load data from annotations in COCO format."""
        num_keypoints = 14
        self._metainfo['CLASSES'] = self.ann_data.loadCats(
            self.ann_data.getCatIds())

        instance_list = []
        image_list = []

        for i, _ann_ids in enumerate(self.sequence_indices):
            expected_num_frames = self.seq_len
            if self.multiple_target:
                expected_num_frames = self.multiple_target

            assert len(_ann_ids) == (expected_num_frames), (
                f'Expected `frame_ids` == {expected_num_frames}, but '
                f'got {len(_ann_ids)} ')

            anns = self.ann_data.loadAnns(_ann_ids)
            num_anns = len(anns)
            img_ids = []
            kpts = np.zeros((num_anns, num_keypoints, 2), dtype=np.float32)
            kpts_3d = np.zeros((num_anns, num_keypoints, 3), dtype=np.float32)
            keypoints_visible = np.zeros((num_anns, num_keypoints),
                                         dtype=np.float32)
            scales = np.zeros((num_anns, 2), dtype=np.float32)
            centers = np.zeros((num_anns, 2), dtype=np.float32)
            bboxes = np.zeros((num_anns, 4), dtype=np.float32)
            bbox_scores = np.zeros((num_anns, ), dtype=np.float32)
            bbox_scales = np.zeros((num_anns, 2), dtype=np.float32)

            for j, ann in enumerate(anns):
                img_ids.append(ann['image_id'])
                kpts[j] = np.array(ann['keypoints'], dtype=np.float32).reshape(14, 2)
                kpts_3d[j] = np.array(ann['keypoints_3d'], dtype=np.float32).reshape(14, 3)
                keypoints_visible[j] = np.array(
                    ann['keypoints_valid'], dtype=np.float32).reshape(14)
                if 'scale' in ann:
                    scales[j] = np.array(ann['scale'])
                if 'center' in ann:
                    centers[j] = np.array(ann['center'])
                bboxes[j] = np.array(ann['bbox'], dtype=np.float32)
                bbox_scores[j] = np.array([1], dtype=np.float32)
                bbox_scales[j] = np.array([1, 1], dtype=np.float32)

            imgs = self.ann_data.loadImgs(img_ids)

            img_paths = np.array([
                osp.join(self.data_root,img['file_name']) for img in imgs
            ])

            factors = np.zeros((kpts_3d.shape[0], ), dtype=np.float32)

            target_idx = [-1] if self.causal else [int(self.seq_len // 2)]
            if self.multiple_target:
                target_idx = list(range(self.multiple_target))

            

            instance_info = {
                'num_keypoints': num_keypoints,
                'keypoints': kpts,
                'keypoints_3d': kpts_3d,
                'keypoints_visible': keypoints_visible,
                'scale': scales,
                'center': centers,
                'id': i,
                'category_id': 1,
                'iscrowd': 0,
                'img_paths': list(img_paths),
                'img_path': img_paths[-1],
                'img_ids': [img['id'] for img in imgs],
                'factor': factors,
                'bbox': bboxes,
                'bbox_scales': bbox_scales,
                'bbox_scores': bbox_scores
            }

            instance_list.append(instance_info)

        if self.data_mode == 'bottomup':
            for img_id in self.ann_data.getImgIds():
                img = self.ann_data.loadImgs(img_id)[0]
                img.update({
                    'img_id':
                    img_id,
                    'img_path':
                    osp.join(self.data_prefix['img'], img['file_name']),
                })
                image_list.append(img)
        del self.ann_data
        return instance_list, image_list

    def load_data_list(self) -> List[dict]:
        data_list = super().load_data_list()
        self.ann_data = None
        return data_list
