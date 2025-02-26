from abc import ABC, abstractmethod
import numpy as np

class Inpainter(ABC):
    @abstractmethod
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint the masked region of an image.
        
        Args:
            image: Input image (H, W, C) in BGR format.
            mask: Binary mask (H, W), 1 for area to inpaint.
        Returns:
            Inpainted image (H, W, C) in BGR format.
        """
        pass