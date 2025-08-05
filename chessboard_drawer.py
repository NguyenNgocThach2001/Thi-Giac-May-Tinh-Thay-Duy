# === chessboard_drawer.py ===
import cv2
import numpy as np

def draw_xiangqi_grid(img, color=(100, 100, 100), thickness=1):
    h, w = img.shape[:2]
    cell_w = w / 8  # 9 lines => 8 cells
    cell_h = h / 9  # 10 lines => 9 cells

    for i in range(9):  # vertical lines
        x = int(i * cell_w)
        cv2.line(img, (x, 0), (x, h), color, thickness)

    for j in range(10):  # horizontal lines
        y = int(j * cell_h)
        cv2.line(img, (0, y), (w, y), color, thickness)

    # Draw river text
    river_top = int(4 * cell_h)
    river_bot = int(5 * cell_h)
    cv2.putText(img, "楚河", (int(w * 0.15), river_top + int(cell_h / 2)), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    cv2.putText(img, "漢界", (int(w * 0.65), river_top + int(cell_h / 2)), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    return img

def draw_pieces(img, boxes, classes, names, font_scale=0.6, font_thickness=2):
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        label = names[cls]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 255, 0), font_thickness)
    return img