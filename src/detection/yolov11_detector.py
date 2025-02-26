from src.detection.detector import Detector
import numpy as np
from ultralytics import YOLO

class YOLOv11Detector(Detector):
    def __init__(self, weights_path: str, conf_threshold: float = 0.5):
        """Initialize the YOLOv11m detector.
        
        Args:
            weights_path: Path to YOLOv11m weights file (e.g., yolov11m.pt).
            conf_threshold: Confidence threshold for detections.
        """
        self.model = YOLO(weights_path)  # Load pre-trained YOLOv11m
        self.conf_threshold = conf_threshold

    def detect(self, image: np.ndarray) -> list:
        """Detect logos in an image using YOLOv11m.
        
        Args:
            image: Input image (H, W, C) in BGR format.
        Returns:
            List of (x_min, y_min, x_max, y_max, conf, class_name).
        """
        results = self.model.predict(image, conf=self.conf_threshold, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                conf = float(box.conf)
                class_name = result.names[int(box.cls)]
                detections.append((x_min, y_min, x_max, y_max, conf, class_name))
        return detections