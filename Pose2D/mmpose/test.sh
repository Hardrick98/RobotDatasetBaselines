#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/test_DWPose.out
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --account=vezzani_fakegs

source activate /homes/rcatalini/.conda/envs/mmpose
export PYTHONPATH=/work/ToyotaHPE/rcatalini/EventRobotPose/mmdetection:$PYTHONPATH

srun python -u tools/test.py configs/robot_2d_keypoint/dwpose_l_dis_m_coco-256x192.py baselines/DWPose/epoch_10.pth --work-dir=./baselines/DWPose/ 