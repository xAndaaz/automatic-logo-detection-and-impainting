from src.detection.detector import Detector
from src.inpainting.inpainter import Inpainter
from src.utils.image_utils import create_mask_from_boxes
import numpy as np
import cv2

class LogoRemover:
    def __init__(self, detector: Detector, inpainter: Inpainter):
        self.detector = detector
        self.inpainter = inpainter

    def remove_logos(self, image: np.ndarray, draw_boxes: bool = False) -> np.ndarray:
        detections = self.detector.detect(image)
        if not detections:
            return image
        
        mask = create_mask_from_boxes(image.shape[:2], detections)
        result = self.inpainter.inpaint(image, mask)
        
        if draw_boxes:
            for x_min, y_min, x_max, y_max, conf, class_name in detections:
                cv2.rectangle(result, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green box
                label = f"{class_name} {conf:.2f}"
                cv2.putText(result, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result