import torch
from ultralytics import YOLO

model = YOLO('yolov8m.pt')
device = torch.device("cuda:1")
results = model.train(data="deep_scores.yaml", imgsz=640, rect=True, epochs = 10, batch=4)