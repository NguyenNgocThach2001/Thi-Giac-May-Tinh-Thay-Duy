import cv2
import os
import random

# === CẤU HÌNH ===
images_dir = r"Dataset\piece\remapped_dts1\train\images"
labels_dir = r"Dataset\piece\remapped_dts1\train\labels"

# === Đọc danh sách file ảnh ===
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# === Chọn ngẫu nhiên 1 ảnh ===
filename = random.choice(image_files)
image_path = os.path.join(images_dir, filename)
label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")

# === Đọc ảnh ===
img = cv2.imread(image_path)
h, w = img.shape[:2]

# === Vẽ nhãn từ file label ===
if os.path.exists(label_path):
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, box_w, box_h = map(float, parts)

            # YOLO → pixel
            cx, cy = x_center * w, y_center * h
            bw, bh = box_w * w, box_h * h
            x1, y1 = int(cx - bw/2), int(cy - bh/2)
            x2, y2 = int(cx + bw/2), int(cy + bh/2)

            # Vẽ box + ID
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, str(int(class_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

# === Hiển thị ===
cv2.imshow(f"Random Image: {filename}", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
