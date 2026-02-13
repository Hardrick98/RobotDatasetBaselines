import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmpose.utils.tensor_utils import to_numpy

from mmpose.registry import METRICS
from ..functional import keypoint_mpjpe


@METRICS.register_module()
class SimpleMPJPE3D(BaseMetric):
    """Evaluator 3D MPJPE / P-MPJPE per dataset custom."""

    def __init__(self, mode='mpjpe', **kwargs):
        """
        Args:
            mode (str): 'mpjpe' o 'p-mpjpe'
        """
        super().__init__(**kwargs)
        assert mode in ['mpjpe', 'p-mpjpe']
        self.mode = mode
        self.reset()

    def reset(self):
        self.errors = []
        self.counts = 0

    def process(self, data_batch, data_samples):
        """
        Args:
            data_batch (dict): batch originale, non necessario qui
            data_samples (list[PoseDataSample]): predizioni e GT
        """
        for sample in data_samples:
            
            pred = sample["pred_instances"]["keypoints"][0]      # (K, 3) pred 3D
            gt = sample["gt_instances"]["lifting_target"][0]      # (K, 3) gt 3D
        
            mask = np.ones((pred.shape[0])) # (K,) boolean

            pred = to_numpy(pred)
            gt = to_numpy(gt)
            #mask = np.array(mask).astype(bool)



            if self.mode == 'p-mpjpe':
                # allineamento procrustes (rigid + scale)
                pred_centered = pred - pred.mean(axis=0)
                gt_centered = gt - gt.mean(axis=0)
                U, s, Vt = np.linalg.svd(gt_centered.T @ pred_centered)
                R = U @ Vt
                pred = (pred_centered @ R.T) * (s.sum() / (pred_centered ** 2).sum())
                pred += gt.mean(axis=0)

            err = np.linalg.norm(pred - gt, axis=-1)
            self.errors.append(err)
            self.counts += mask.sum()

    def compute_metrics(self, reset=True):
        all_errors = np.concatenate(self.errors, axis=0)
        mpjpe = all_errors.mean() if self.counts > 0 else 0.0
        if reset:
            self.reset()
        return {f'{self.mode}': float(mpjpe)}
