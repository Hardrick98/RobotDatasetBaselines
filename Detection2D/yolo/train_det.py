from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description='Train a YOLO model for robot detection')
parser.add_argument('--model', type=str, default="yolo26n.yaml", help='Path to model YAML file')
args = parser.parse_args()
model_config  = args.model


# Load a model
model = YOLO(model_config)  # build a new model from YAML
#model = YOLO("yolo26n-pose.pt")  # load a pretrained model (recommended for training)
#model = YOLO("yolo26n-pose.yaml").load("yolo26n-pose.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="robots_det.yaml", epochs=10, imgsz=640, batch=16)