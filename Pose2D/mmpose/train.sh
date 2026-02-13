#!/bin/bash
#SBATCH --job-name=pose2d_train
#SBATCH --output="logs/train_2DWPose.out"
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=vezzani_fakegs
#SBATCH --time=1-00:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-gpu=8

#cd /work/ToyotaHPE/rcatalini/EventRobotPose/3DRobotPose/mmpose/projects/rtmpose3d/


source activate /homes/rcatalini/.conda/envs/mmpose
export PYTHONPATH=$(pwd):$PYTHONPATH

srun python -u tools/train.py configs/robot_2d_keypoint/dwpose_l_dis_m_coco-256x192.py --work-dir=./baselines/DWPose/ #--launcher="slurm"
#srun python -u tools/train.py configs/robot_2d_keypoint/rtmpose-l_8xb256-420e_robots-256x192.py --work-dir=./baselines/RTMPose/ --launcher="slurm"

#cd /work/ToyotaHPE/rcatalini/EventRobotPose/3DRobotPose/mmpose/
#python tools/train.py /work/ToyotaHPE/rcatalini/EventRobotPose/3DRobotPose/mmpose/projects/rtmpose3d/configs/rtmw3d_robots.py
#python tools/train.py configs/robot_2d_keypoint/rtmpose-l_8xb256-420e_robots-256x192.py