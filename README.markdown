# StreamClear: AI-Driven Logo Detection and Removal for Streamlined Content Control

## Overview
StreamClear is an AI-driven system designed to detect and remove logos from images, enabling seamless content repurposing for content creators, media agencies, e-learning platforms, and journalists. The project leverages **YOLOv11** for logo detection and **LaMa** for inpainting, achieving high accuracy and near-real-time performance. The system was trained and evaluated on the **FlickrLogos-47** dataset, addressing challenges such as copyright compliance and content reusability.

This project was developed as part of a Bachelor of Engineering in Computer Science with Specialization in Artificial Intelligence and Machine Learning at Chandigarh University under the supervision of Mr. Harjot Singh.

---

## Features
- **Logo Detection**: Utilizes YOLOv11 to accurately detect logos in images with bounding boxes.
- **Logo Removal**: Employs LaMa to inpaint the logo area, reconstructing the background seamlessly.
- **Near-Real-Time Processing**: Processes images in 12.5 ms, suitable for practical applications.
- **Scalable Pipeline**: Modular design allows for independent optimization of detection and inpainting stages.

---

## Dataset
- **FlickrLogos-47**: A dataset containing 8240 training images and 5453 test images across 48 classes (47 logo classes + 1 `no-logo` class).
- **Preprocessing**: Annotations were converted from `.gt_data.txt` to YOLO-compatible `.txt` files using `convert_annotations.py`.

---

## Requirements
To run this project, ensure you have the following installed:
- Python 3.8+
- PyTorch 2.0+
- Ultralytics YOLOv11 (`ultralytics` package)
- LaMa (pre-trained weights: `best.ckpt`)
- NVIDIA GPU (e.g., RTX 3060) for accelerated computation
- Additional dependencies listed in `requirements.txt`

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/[YourGitHubUsername]/StreamClear.git
   cd StreamClear
   ```

2. **Download the Dataset**:
   - Download the FlickrLogos-47 dataset from its official source.
   - Place the dataset in the `data/` directory with the following structure:
     ```
     data/
     ├── train/
     └── test/
     ```

3. **Download Pre-trained Models**:
   - Download the pre-trained LaMa weights (`best.ckpt`) and place them in `lama/models/`.
   - YOLOv11 weights will be automatically downloaded during training if not present.

4. **Prepare the Dataset**:
   - Run the annotation conversion script:
     ```bash
     python convert_annotations.py --dataset_path data/
     ```

---

## Usage
1. **Train the YOLOv11 Model**:
   - Fine-tune YOLOv11 on the FlickrLogos-47 dataset:
     ```bash
     python train.py --data data/flickrlogos47.yaml --epochs 50 --batch-size 8 --img-size 640
     ```
   - Training outputs will be saved in `runs/train/yolov11_flickrlogos47/`.

2. **Run the Pipeline**:
   - Process images using the full pipeline (detection + inpainting):
     ```bash
     python run_pipeline.py --image_path test/000000/00000007.jpg --draw_boxes True --output_dir output/
     ```
   - Options:
     - `--draw_boxes True`: Draws bounding boxes on detected logos for visualization.
     - `--output_dir`: Directory to save processed images.

3. **Evaluate the Model**:
   - Evaluate detection performance:
     ```bash
     python evaluate.py --weights runs/train/yolov11_flickrlogos47/weights/best.pt --data data/flickrlogos47.yaml
     ```

---

## Project Structure
```
StreamClear/
├── data/                    # Dataset directory (FlickrLogos-47)
├── lama/                    # LaMa model and weights
├── output/                  # Processed images
├── runs/                    # Training outputs
├── convert_annotations.py   # Script to preprocess dataset annotations
├── evaluate.py              # Script to evaluate detection performance
├── run_pipeline.py          # Main script to run the pipeline
├── train.py                 # Script to train YOLOv11
├── image_utils.py           # Utility functions for image processing
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

---

## Results
- **Detection Performance (YOLOv11)**:
  - Precision: 0.892
  - Recall: 0.781
  - mAP@0.5: 0.854
  - Speed: 12.1 ms per image
- **Inpainting Performance (LaMa)**:
  - PSNR: 31.2 dB
  - SSIM: 0.943
- **End-to-End Latency**: 12.5 ms per image
- **User Feedback**: Rated 4.4–4.8/5 for accuracy, quality, and usability by 10 content creators.



## Limitations
- Struggles with small or occluded logos due to class imbalance in the dataset.
- Inpainting artifacts may appear in complex, cluttered backgrounds.
- Currently supports static images only (video support planned for future work).

---

## Future Work
- Extend the pipeline to handle video streams for real-time processing.
- Improve detection of small/occluded logos through data augmentation and multi-scale techniques.
- Enhance inpainting for complex backgrounds using hybrid models (e.g., diffusion-based methods).
- Optimize for edge devices (e.g., smartphones) to broaden accessibility.

---

## References
- Romdhane, R., et al. (2015). "LOGO-Net: Large-scale Deep Logo Detection and Brand Recognition." *IEEE Transactions on Multimedia*.
- Suvorov, R., et al. (2021). "Resolution-robust Large Mask Inpainting with Fourier Convolutions." *WACV 2021*.
- Redmon, J., et al. (2016). "You Only Look Once: Unified, Real-Time Object Detection." *CVPR 2016*.
- Ultralytics. (2024). "YOLOv11: Improved Real-Time Object Detection." *GitHub Repository*.
- Romdhane, R., et al. (2012). "FlickrLogos-32: A Realistic Benchmark for Logo Detection." *ICPR 2012*.

---

## Contributors
- **Andaaz (22BAI70129)**  
- **Utkarsh Gottlieb (22BAI70114)**  
- **Aryaman Mohan (22BAI70126)**  
- **Supervisor**: Mr. Harjot Singh, Assistant Professor, Chandigarh University

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
We thank Chandigarh University and Mr. Harjot Singh for their guidance and support throughout this project. We also acknowledge the creators of the FlickrLogos-47 dataset and the open-source communities behind YOLOv11 and LaMa.
