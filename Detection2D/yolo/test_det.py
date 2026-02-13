from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description='Train a YOLO model for robot detection')
parser.add_argument('--model', type=str, default="yolo26n.yaml", help='Path to model YAML file')
parser.add_argument('--weights', type=str, default=None, help='Path to weights file')
parser.add_argument('--name', type=str, default=None, help='Name for the test run')
args = parser.parse_args()
model_config  = args.model


# Load a model
model = YOLO(model_config)  # build a new model from YAML
#model = YOLO("yolo26n-pose.pt")  # load a pretrained model (recommended for training)

model = YOLO(args.weights) 
# Customize validation settings
metrics = model.val(data="robots_det.yaml", imgsz=640, batch=16, conf=0.25, iou=0.7, device="0", save_json=True, split="test", name=args.name)  # val/test a model, save JSON results to file
 # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps