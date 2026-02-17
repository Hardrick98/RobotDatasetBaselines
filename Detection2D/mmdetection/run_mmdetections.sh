export PYTHONPATH=./:$PYTHONPATH
python -u tools/train.py configs/robots/faster-rcnn_r50_fpn_1x_robots.py --work-dir=./baselines/ 2>&1 | tee "RESULTS/results_faster.txt"
# python -u tools/train.py configs/robots/dino-4scale_r50_8xb2-12e_robots.py --work-dir=./baselines/ 2>&1 | tee "RESULTS/results_dino.txt"
# python -u tools/train.py configs/robots/detr_r50_8xb2-150e_robots.py--work-dir=./baselines/ 2>&1 | tee "RESULTS/results_Detr.txt"
# python -u tools/train.py configs/robots/deformable-detr_r50_16xb2-50e_robots.py  --work-dir=./baselines/  2>&1 | tee "RESULTS/results_defDetr.txt"
