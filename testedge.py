import cv2
import numpy as np
from collections import deque

# Làm mượt contour
recent_quads = deque(maxlen=5)

def get_webcam_frame(cap):
    ret, frame = cap.read()
    return frame if ret else None

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur

def apply_canny(image, low=50, high=150):
    return cv2.Canny(image, low, high)

def close_edges(edge_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    return cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE, kernel)

def order_points(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],     # top-left
        pts[np.argmin(diff)],  # top-right
        pts[np.argmax(s)],     # bottom-right
        pts[np.argmax(diff)]   # bottom-left
    ], dtype="float32")

def is_rectangle(pts, angle_thresh=15):
    def angle(p1, p2, p3):
        v1, v2 = p1 - p2, p3 - p2
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))
    pts = order_points(np.array(pts, dtype="float32"))
    angles = [angle(pts[i - 1], pts[i], pts[(i + 1) % 4]) for i in range(4)]
    return all(abs(a - 90) < angle_thresh for a in angles)

def find_largest_rectangle(closed_img):
    contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1500:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            if is_rectangle(pts):
                return order_points(pts)
    return None

def smooth_quad(new_quad):
    if new_quad is None:
        return None
    recent_quads.append(new_quad)
    return np.mean(recent_quads, axis=0)

def draw_rectangle_overlay(frame, quad_pts):
    if quad_pts is not None:
        cv2.polylines(frame, [np.int32(quad_pts)], True, (0, 255, 0), 2)
    return frame

def warp_board(frame, quad_pts, zoom_out_ratio=0.08):
    if quad_pts is None:
        return None

    # Tính kích thước bàn cờ theo chiều dài 2 cạnh
    width = max(np.linalg.norm(quad_pts[0] - quad_pts[1]), np.linalg.norm(quad_pts[2] - quad_pts[3]))
    height = max(np.linalg.norm(quad_pts[1] - quad_pts[2]), np.linalg.norm(quad_pts[3] - quad_pts[0]))

    # Mở rộng thêm tỷ lệ zoom_out
    zoom_factor = 1 + zoom_out_ratio
    center = np.mean(quad_pts, axis=0)
    expanded_quad = center + (quad_pts - center) * zoom_factor

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(expanded_quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(frame, M, (int(width), int(height)))
    return warped

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không mở được webcam.")
        return

    while True:
        frame = get_webcam_frame(cap)
        if frame is None:
            break

        preprocessed = preprocess_frame(frame)
        canny_edges = apply_canny(preprocessed)
        closed_edges = close_edges(canny_edges)

        raw_quad = find_largest_rectangle(closed_edges)
        smoothed_quad = smooth_quad(raw_quad)

        detected_frame = frame.copy()
        detected_frame = draw_rectangle_overlay(detected_frame, smoothed_quad)

        aligned_board = warp_board(frame, smoothed_quad, zoom_out_ratio=0.08)

        # Hiển thị các cửa sổ
        cv2.imshow("webcam", frame)
        cv2.imshow("canny_edges", canny_edges)
        cv2.imshow("closed_edges", closed_edges)
        cv2.imshow("detected_board", detected_frame)
        if aligned_board is not None:
            cv2.imshow("aligned_board", aligned_board)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
