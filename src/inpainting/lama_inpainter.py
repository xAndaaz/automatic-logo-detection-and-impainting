from .inpainter import Inpainter
import torch
import numpy as np
import cv2
import yaml
import sys
import os

lama_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../lama'))
sys.path.append(lama_dir)

from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.utils import move_to_device
from omegaconf import OmegaConf

class LaMaInpainter(Inpainter):
    def __init__(self, weights_path: str, config_path: str = "configs/inpainting_config.yaml"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(config_path, 'r') as f:
            self.config = OmegaConf.create(yaml.safe_load(f))
            print("Loaded config:\n", OmegaConf.to_yaml(self.config))
        
        self.model = load_checkpoint(
            train_config=self.config,
            path=weights_path,
            map_location=self.device,
            strict=False,
            predict_only=True  # Ensure skipping training components
        )
        self.model.to(self.device)
        self.model.eval()
        self.model.freeze()

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        orig_height, orig_width = image.shape[:2]
        pad_to = 32
        new_height = ((orig_height + pad_to - 1) // pad_to) * pad_to
        new_width = ((orig_width + pad_to - 1) // pad_to) * pad_to
        pad_height = new_height - orig_height
        pad_width = new_width - orig_width
        
        image_padded = cv2.copyMakeBorder(
            image, 0, pad_height, 0, pad_width, cv2.BORDER_REFLECT
        )
        mask_padded = cv2.copyMakeBorder(
            mask, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=0
        )
        
        image_rgb = cv2.cvtColor(image_padded, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask_padded).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        batch = {'image': image_tensor, 'mask': mask_tensor}
        batch = move_to_device(batch, self.device)
        
        with torch.no_grad():
            batch = self.model(batch)
            inpainted = batch['inpainted'].squeeze(0).permute(1, 2, 0).cpu().numpy()
            inpainted = inpainted[:orig_height, :orig_width, :]  # Crop to original size
            inpainted = np.clip(inpainted * 255, 0, 255).astype(np.uint8)
        
        inpainted_bgr = cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR)
        return inpainted_bgr