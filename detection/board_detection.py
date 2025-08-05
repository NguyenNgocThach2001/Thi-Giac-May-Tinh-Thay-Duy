import cv2
import numpy as np
from ultralytics import YOLO

def extract_quad_from_mask(mask):
    mask = (mask.cpu().numpy() * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        epsilon = 0.03 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2)
    return None

def order_quad_points(pts):
    center = np.mean(pts, axis=0)
    def angle(pt): return np.arctan2(pt[1] - center[1], pt[0] - center[0])
    pts_sorted = sorted(pts, key=angle)
    pts_sorted = np.array(pts_sorted, dtype=np.float32)
    top_left_idx = np.argmin(np.sum(pts_sorted, axis=1))
    pts_sorted = np.roll(pts_sorted, -top_left_idx, axis=0)
    return pts_sorted

def align_board(frame, quad, output_size=(640, 640)):
    quad = order_quad_points(quad)
    dst = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    aligned = cv2.warpPerspective(frame, M, output_size)
    aligned = cv2.rotate(aligned, cv2.ROTATE_90_CLOCKWISE)
    return aligned

def zoomout_after_align(image, pad_ratio=0.1):
    h, w = image.shape[:2]
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    padded = cv2.copyMakeBorder(
        image, pad_y, pad_y, pad_x, pad_x,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # màu đen
    )
    resized = cv2.resize(padded, (w, h))  # resize lại về 640x640
    return resized

def load_board_model(model_path="runs/segment/yolov8_segment_board/weights/best.pt"):
    return YOLO(model_path)

def detect_board(model, frame):
    # Resize về 640x640 (bạn đã train ở size này)
    frame_resized = cv2.resize(frame, (640, 640))

    results = model.predict(source=frame_resized, conf=0.4, verbose=False)
    found_quad = None

    for r in results:
        if r.masks is not None:
            for mask in r.masks.data:
                quad = extract_quad_from_mask(mask)
                if quad is not None:
                    found_quad = quad
                    break

    if found_quad is not None:
        aligned = align_board(frame_resized, found_quad, output_size=(640, 640))
        aligned_zoomout = zoomout_after_align(aligned, pad_ratio=0.1)

        # DEBUG (tuỳ chọn):
        # cv2.imshow("ALIGNED", aligned)
        # cv2.imshow("ZOOMOUT", aligned_zoomout)

        return aligned_zoomout, True

    return frame_resized, False
