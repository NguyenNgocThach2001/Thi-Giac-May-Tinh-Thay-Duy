# === piece_detector.py ===
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("runs/detect/xiangqi_fold5/weights/best.pt")
class_names = model.model.names

def detect_pieces(warped_img):
    results = model.predict(source=warped_img, conf=0.5, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    names = class_names
    return warped_img, boxes, classes, names
