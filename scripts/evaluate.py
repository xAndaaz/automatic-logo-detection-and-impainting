# scripts/evaluate.py
from ultralytics import YOLO
import os
import torch
def evaluate_yolo(weights_path, data_dir):
    model = YOLO(weights_path)
    results = model.val(
        data=os.path.join(data_dir, 'data.yaml'),
        imgsz=640,
        batch=8,
        device=0 if torch.cuda.is_available() else 'cpu',
        project='runs/val',
        name='yolov11_flickrlogos47',
        exist_ok=True
    )
    print("Evaluation Results:", results)

if __name__ == "__main__":
    data_dir = "E:/LOGO-detection and removal/data/dataset/FlickrLogos_47"
    weights_path = "E:/LOGO-detection and removal/runs/train/yolov11_flickrlogos47/weights/best.pt"
    evaluate_yolo(weights_path, data_dir)