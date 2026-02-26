#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/test_RTMPose3D.out
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --account=vezzani_fakegs

cd /work/ToyotaHPE/rcatalini/EventRobotPose/mmpose/projects/rtmpose3d/


source activate /homes/rcatalini/.conda/envs/mmpose
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=/work/ToyotaHPE/rcatalini/EventRobotPose/mmdetection:$PYTHONPATH
cd /work/ToyotaHPE/rcatalini/EventRobotPose/mmpose/
#srun python -u tools/test.py configs/robot_2d_keypoint/dwpose_l_dis_m_coco-256x192.py baselines/DWPose/epoch_10.pth --work-dir=./baselines/DWPose/ 

#srun python -u tools/test.py configs/robot_2d_keypoint/td-hm_hrnet-w32_8xb64-210e_robot-256x192.py baselines/HRNet/epoch_8.pth --work-dir=./baselines/HRNet/ 
#srun python -u tools/test.py configs/robot_2d_keypoint/rtmpose-l_8xb256-420e_robots-256x192.py baselines/RTMPose/epoch_10.pth --work-dir=./baselines/RTMPose/ --out ./baselines/RTMPose/results.pkl
srun python -u tools/test.py /work/ToyotaHPE/rcatalini/EventRobotPose/mmpose/projects/rtmpose3d/configs/rtmw3d_robots.py ./baselines/RTMPose3D/epoch_1.pth --work-dir=./baselines/RTMPose3D/ --out ./baselines/RTMPose3D/results.pkl #--launcher="slurm"