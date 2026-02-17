#!/bin/bash
source /andromeda/personal/gmagrini/miniconda3/etc/profile.d/conda.sh 
conda activate mmpose

MODELS=("yolo11l-pose.yaml" "yolo26s-pose.yaml")
DATA="../../exo_dataset_yolo_pose/dataset.yaml"
EPOCHS=10
IMGSZ=640

for MODEL in "${MODELS[@]}"; do
  NAME="${MODEL%.yaml}"
  echo "===== Training $NAME ====="

  python3 -c "
from ultralytics import YOLO
model = YOLO('$MODEL')
results = model.train(data='$DATA', epochs=$EPOCHS, imgsz=$IMGSZ, name='$NAME', batch=128)
" 2>&1 | tee "results_${NAME}.txt"

  echo "===== Done $NAME — saved to results_${NAME}.txt ====="
done

echo "All trainings complete!"
