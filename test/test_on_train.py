from ultralytics import YOLO
import cv2
import os
from pathlib import Path

# === Cấu hình ===
model_path = "runs/detect/xiangqi3/weights/best.pt"  # đường dẫn model đã train
image_dir = "Dataset/piece/Dataset1_800image/train/images"      # ảnh từ tập train
save_output = True                                              # True để lưu kết quả
output_dir = "runs/overfit_check"

# === Load model
model = YOLO(model_path)
os.makedirs(output_dir, exist_ok=True)

# === Lấy danh sách ảnh
image_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))

# === Infer từng ảnh
for img_path in image_paths:
    img = cv2.imread(str(img_path))
    results = model.predict(img, conf=0.5, verbose=False)

    for r in results:
        annotated = r.plot()

    # Lưu nếu cần
    if save_output:
        out_path = Path(output_dir) / img_path.name
        cv2.imwrite(str(out_path), annotated)

cv2.destroyAllWindows()
