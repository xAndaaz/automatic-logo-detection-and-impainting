import os
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO

def train_yolo(data_dir, weights_path=r"E:\LOGO-detection and removal\models\yolo11m.pt", epochs=50, batch_size=8):
    model = YOLO(weights_path)
    model.train(
        data=os.path.join(data_dir, 'data.yaml'),
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        device=0 if torch.cuda.is_available() else 'cpu',
        project='runs/train',
        name='yolov11_flickrlogos47',
        exist_ok=True
    )

if __name__ == "__main__":
    data_dir = "E:/LOGO-detection and removal/data/dataset/FlickrLogos_47"
    train_yolo(data_dir, epochs=50, batch_size=8)