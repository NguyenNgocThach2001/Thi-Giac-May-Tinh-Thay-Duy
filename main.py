import cv2
import numpy as np
from board_detector import detect_chessboard, draw_grid_on_board

# === Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không mở được webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    warped, board_contour, mask = detect_chessboard(frame)

    # Vẽ contour bàn cờ lên ảnh gốc nếu có
    display = frame.copy()
    if board_contour is not None:
        cv2.polylines(display, [board_contour.astype(int)], True, (0, 255, 0), 2)

    # Nếu phát hiện bàn cờ, warp và chia ô
    if warped is not None:
        grid_overlay = draw_grid_on_board(warped)
        cv2.imshow("Warped with Grid", grid_overlay)
    else:
        blank = np.zeros((480, 480, 3), dtype=np.uint8)
        cv2.imshow("Warped with Grid", blank)

    # Hiển thị các ảnh phụ
    cv2.imshow("Camera", display)
    if mask is not None:
        cv2.imshow("Board Mask", mask)

    # Thoát nếu nhấn phím q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
