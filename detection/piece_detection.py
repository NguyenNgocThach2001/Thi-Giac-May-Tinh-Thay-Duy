from ultralytics import YOLO
import cv2
import numpy as np

def load_piece_model(model_path="runs/detect/model_on_dataset_combined_500_epoch/weights/best.pt"):
    return YOLO(model_path)

def detect_pieces_and_get_positions(model, frame):
    class_names = model.model.names
    results = model.predict(source=frame, conf=0.8, verbose=False)

    pieces = []

    WIDTH, HEIGHT = 640, 640
    SAFE_MARGIN_X = 20  # chỉnh nếu cần
    SAFE_MARGIN_Y = 30

    GRID_X0 = SAFE_MARGIN_X
    GRID_Y0 = SAFE_MARGIN_Y
    GRID_W = WIDTH - 2 * SAFE_MARGIN_X
    GRID_H = HEIGHT - 2 * SAFE_MARGIN_Y

    CELL_W = GRID_W / 9
    CELL_H = GRID_H / 10

    for r in results:
        for box, cls_id in zip(r.boxes.xyxy, r.boxes.cls):
            b = box.cpu().numpy()
            x1, y1, x2, y2 = b
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Xích vào trong vùng chia lưới
            col = int((center_x - GRID_X0) / CELL_W)
            row = int((center_y - GRID_Y0) / CELL_H)

            if 0 <= col <= 8 and 0 <= row <= 9:
                label = class_names.get(int(cls_id.item()), str(int(cls_id.item())))
                pieces.append((label, (col, row)))

    return pieces

