# Autonomous Driving - Car Detection (YOLO)

This project is part of the Deep Learning Specialization (Convolutional Neural Networks course). The main goal of the project is to use the YOLO (You Only Look Once) object detection algorithm to detect cars and other obstacles in road scenarios with high accuracy.

The core functions implemented in this project and their programming/mathematical logic are as follows:

1. YOLO Box Filtering (`yolo_filter_boxes`):
   Computes box scores by performing an element-wise product between box confidence and class probabilities. It filters out weak boxes below a specified threshold, retaining only the most confident object detections.

2. Intersection over Union (`iou`):
   Measures the overlap degree between two bounding boxes. By calculating the intersection area divided by the union area, it determines how closely boxes overlap and whether they cover the same object.

3. Non-Max Suppression (`yolo_non_max_suppression`):
   Eliminates redundant and duplicate boxes generated around the same object. Using IoU and class filtering, it keeps a single box with the highest score and suppresses overlapping redundant boxes.

4. Model Evaluation (`yolo_eval`):
   Takes the raw outputs of the YOLO model and chains all the processing steps together. It converts box coordinates to corners, performs score filtering, scales boxes back to the original image dimensions, and applies Non-Max Suppression to return the final cleaned bounding boxes, classes, and scores.

Required Libraries: TensorFlow, NumPy, Matplotlib, Pillow.
