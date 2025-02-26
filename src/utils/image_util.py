import cv2
import numpy as np
from typing import List, Tuple

def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk in BGR format.
    
    Args:
        image_path: Path to the image file.
    Returns:
        Image as a numpy array (H, W, C) in BGR format.
    Raises:
        FileNotFoundError: If the image file doesn't exist.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    return image

def save_image(image: np.ndarray, output_path: str) -> None:
    """Save an image to disk.
    
    Args:
        image: Image as a numpy array (H, W, C).
        output_path: Path where the image will be saved.
    """
    cv2.imwrite(output_path, image)

def create_mask_from_boxes(image_shape: Tuple[int, int], 
                          detections: List[Tuple[int, int, int, int, float, str]], 
                          dilation: int = 5) -> np.ndarray:
    """Create a binary mask from detection bounding boxes.
    
    Args:
        image_shape: Tuple of (height, width) of the image.
        detections: List of (x_min, y_min, x_max, y_max, conf, class_name).
        dilation: Number of pixels to dilate the mask for better coverage.
    Returns:
        Binary mask (H, W) where 1 indicates area to inpaint, 0 otherwise.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    for x_min, y_min, x_max, y_max, _, _ in detections:
        mask[y_min:y_max, x_min:x_max] = 1
    if dilation > 0:
        kernel = np.ones((dilation, dilation), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask