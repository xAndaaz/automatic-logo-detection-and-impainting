import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class FlickrLogos47Dataset(Dataset):
    def __init__(self, data_dir, mode='train', transform=None):
        """
        Args:
            data_dir (str): Root directory of FlickrLogos_47 (e.g., 'data/datasets/FlickrLogos_47').
            mode (str): 'train' or 'test'.
            transform: Optional transform for images (e.g., augmentation).
        """
        self.data_dir = os.path.join(data_dir, mode)
        self.transform = transform
        self.classes = self._load_classes(os.path.join(data_dir, r'className2ClassID.txt'))
        self.image_files, self.annotations = self._load_dataset()

    def _load_classes(self, class_file):
        """Load class names and IDs from className2ClassID.txt."""
        class_map = {}
        with open(class_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_name, class_id = line.strip().split()
                    class_map[int(class_id)] = class_name
        return class_map

    def _load_dataset(self):
        """Load image paths and annotations from subfolders."""
        image_files = []
        annotations = []
        
        for class_folder in os.listdir(self.data_dir):
            folder_path = os.path.join(self.data_dir, class_folder)
            if not os.path.isdir(folder_path):
                continue
            for img_file in os.listdir(folder_path):
                if img_file.endswith('.jpg') or img_file.endswith('.png'):
                    img_path = os.path.join(folder_path, img_file)
                    gt_file = img_path.replace('.jpg', '.gt_data.txt').replace('.png', '.gt_data.txt')
                    if os.path.exists(gt_file):
                        boxes = self._parse_gt_file(gt_file)
                        image_files.append(img_path)
                        annotations.append(boxes)
        
        return image_files, annotations

    def _parse_gt_file(self, gt_file):
        """Parse groundtruth file into list of [x1, y1, x2, y2, class_id]."""
        boxes = []
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:  # x1, y1, x2, y2, class_id, ...
                    x1, y1, x2, y2, class_id = map(int, parts[:5])
                    difficult = int(parts[7]) if len(parts) > 7 else 0
                    if not difficult:  # Exclude difficult instances
                        boxes.append([x1, y1, x2, y2, class_id])
        return boxes

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        boxes = self.annotations[idx]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        # Convert to YOLO format (x_center, y_center, width, height, class_id) normalized
        h, w = image.shape[:2]
        targets = []
        for box in boxes:
            x1, y1, x2, y2, class_id = box
            x_center = (x1 + x2) / 2 / w
            y_center = (y1 + y2) / 2 / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            targets.append([x_center, y_center, width, height, class_id])
        
        return image, np.array(targets, dtype=np.float32) if targets else np.zeros((0, 5), dtype=np.float32)

if __name__ == "__main__":
    dataset = FlickrLogos47Dataset(r"E:/LOGO-detection and removal/data/datasets/FlickrLogos_47")
    print(f"Loaded {len(dataset)} images")
    img, targets = dataset[0]
    print(f"Sample image shape: {img.shape}, Targets: {targets}")