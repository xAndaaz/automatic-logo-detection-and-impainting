from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class Detector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float, str]]:
        """Detect logos in an image.
        
        Args:
            image: Input image (H, W, C) in BGR format.
        Returns:
            List of detections: (x_min, y_min, x_max, y_max, confidence, class_name).
        """
        pass