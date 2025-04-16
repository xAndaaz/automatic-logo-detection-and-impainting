The data.yaml file defines the dataset for YOLOv11 training.
present in (.data/datasets/FlickrLogos_47/data.yaml)

YOLOv11 expects annotations in .txt files (one per image) in the format
Current: .gt_data.txt files use <x1> <y1> <x2> <y2> <class_id> ....
Convert these to YOLO format. Update dataset_loader.py to handle this on-the-fly was our approach, but Ultralytics prefers pre-converted files. using convert_annotations.py (Output: Creates .txt files alongside each image in train/ and test/)
