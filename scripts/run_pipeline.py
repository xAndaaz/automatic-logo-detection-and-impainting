import cv2
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.detection.yolov11_detector import YOLOv11Detector
from src.inpainting.lama_inpainter import LaMaInpainter
from src.pipeline.logo_remover import LogoRemover
from src.utils.image_utils import load_image, save_image
from src.utils.logging import setup_logger

def main():
    # Setup logger at the start of main
    logger = setup_logger(__name__)
    logger.info("Starting logo removal pipeline")

    # Initialize models
    logger.info("Initializing YOLOv11 detector")
    detector = YOLOv11Detector(
        weights_path=r"E:\LOGO-detection and removal\runs\train\yolov11_flickrlogos47\weights\best.pt",
        conf_threshold=0.5
    )
    
    logger.info("Initializing LaMa inpainter")
    inpainter = LaMaInpainter(
        weights_path=r"models\lama\big-lama\models\best.ckpt",
        config_path=r"configs\inpainting_config.yaml"
    )
    
    logger.info("Initializing LogoRemover pipeline")
    remover = LogoRemover(detector, inpainter)

    # Load and process image
    logger.info("Loading image from data/flickrlogos32/sample.jpg")
    image = load_image(r"E:\LOGO-detection and removal\data\dataset\FlickrLogos_47\test\000001\000001004.png")
    
    logger.info("Running logo removal")
    result = remover.remove_logos(image)
    #commnet this out if bounding box needed 
    #result = remover.remove_logos(image, draw_boxes=True)
    
    logger.info("Saving result to output/result.jpg")
    save_image(result, "output/result.jpg")
    logger.info("Processed image saved to output/result.jpg")

if __name__ == "__main__":
    main()

    