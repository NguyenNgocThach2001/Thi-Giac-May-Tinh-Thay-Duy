import cv2
import numpy as np

# Biến trạng thái toàn cục
last_warped = None
last_contour = None
last_mask = None
stable_counter = 0
STABLE_FRAMES = 10

def order_points(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect

def is_rectangle(pts, angle_thresh=15):
    def angle(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    pts = order_points(np.array(pts, dtype="float32"))
    angles = [angle(pts[i - 1], pts[i], pts[(i + 1) % 4]) for i in range(4)]
    return all(abs(a - 90) < angle_thresh for a in angles)

def detect_chessboard(frame):
    global last_warped, last_contour, last_mask, stable_counter

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    blurred = cv2.medianBlur(filtered, 7)

    h, w = blurred.shape
    margin = 0.1
    roi = blurred[int(h * margin):int(h * (1 - margin)), int(w * margin):int(w * (1 - margin))]
    roi_offset = (int(w * margin), int(h * margin))

    edges = cv2.Canny(roi, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    best_rect = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 4000:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            try:
                pts = approx.reshape(4, 2) + roi_offset
                if is_rectangle(pts):
                    best_rect = pts
                    break
            except:
                continue

    if best_rect is not None:
        ordered = order_points(best_rect)
        center = np.mean(ordered, axis=0)
        ordered = center + (ordered - center) * 1.05  # scale outward

        (tl, tr, br, bl) = ordered
        width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
        height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(frame, M, (int(width), int(height)))

        last_warped = warped
        last_contour = ordered
        last_mask = edges
        stable_counter = STABLE_FRAMES
        return warped, ordered, edges
    else:
        if stable_counter > 0:
            stable_counter -= 1
            return last_warped, last_contour, last_mask
        else:
            last_warped = None
            last_contour = None
            last_mask = edges
            return None, None, edges

def draw_grid_on_board(warped_img, rows=10, cols=9):
    h, w = warped_img.shape[:2]
    img = warped_img.copy()
    for y in range(rows + 1):
        y_pos = int(h * y / rows)
        cv2.line(img, (0, y_pos), (w, y_pos), (0, 255, 255), 1)
    for x in range(cols + 1):
        x_pos = int(w * x / cols)
        cv2.line(img, (x_pos, 0), (x_pos, h), (0, 255, 255), 1)
    return img
