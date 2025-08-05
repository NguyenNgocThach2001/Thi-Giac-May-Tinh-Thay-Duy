import os
import shutil
import json
import cv2

# === Cấu hình đường dẫn ===
ROOT = r""  # ← Đặt đường dẫn gốc tại đây
RAW_IMG_DIR = os.path.join(ROOT, "rawdata", "images")
RAW_JSON_DIR = os.path.join(ROOT, "rawdata", "labels")
OUT_IMG_DIR = os.path.join(ROOT, "processed_data_detection", "images")
OUT_LABEL_DIR = os.path.join(ROOT, "processed_data_detection", "labels")

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LABEL_DIR, exist_ok=True)

# === Class mapping đúng thứ tự từ data.yaml ===
class_map = {
    "black-advisor": 0,
    "black-cannon": 1,
    "black-chariot": 2,
    "black-elephant": 3,
    "black-general": 4,
    "black-horse": 5,
    "black-soldier": 6,
    "intersection": 7,
    "red-advisor": 8,
    "red-cannon": 9,
    "red-chariot": 10,
    "red-elephant": 11,
    "red-general": 12,
    "red-horse": 13,
    "red-soldier": 14
}

def polygon_to_bbox(polygon):
    xs = [pt["x"] for pt in polygon]
    ys = [pt["y"] for pt in polygon]
    return min(xs), min(ys), max(xs), max(ys)

def convert_and_copy(image_name):
    name, ext = os.path.splitext(image_name)
    if ext.lower() not in [".jpg", ".jpeg", ".png"]:
        return

    img_path = os.path.join(RAW_IMG_DIR, image_name)
    json_path = os.path.join(RAW_JSON_DIR, name + ".json")
    out_img_path = os.path.join(OUT_IMG_DIR, image_name)
    out_label_path = os.path.join(OUT_LABEL_DIR, name + ".txt")

    if not os.path.exists(img_path):
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Không đọc được ảnh: {image_name}")
        return
    h, w = img.shape[:2]

    # Nếu không có file JSON thì chỉ copy ảnh (ảnh negative)
    if not os.path.exists(json_path):
        shutil.copy(img_path, out_img_path)
        print(f"⚠️ Không có JSON: {image_name} → ảnh negative")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            annotations = json.load(f)
        except Exception as e:
            print(f"❌ Lỗi đọc JSON {json_path}: {e}")
            return

    valid_boxes = []
    for obj in annotations:
        if "content" not in obj or len(obj["content"]) < 4 or "labels" not in obj:
            continue

        label_name = obj["labels"]["labelName"]
        if label_name not in class_map:
            continue

        class_id = class_map[label_name]
        xmin, ymin, xmax, ymax = polygon_to_bbox(obj["content"])

        x_center = ((xmin + xmax) / 2) / w
        y_center = ((ymin + ymax) / 2) / h
        box_w = (xmax - xmin) / w
        box_h = (ymax - ymin) / h

        valid_boxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

    if valid_boxes:
        with open(out_label_path, "w") as out:
            out.write("\n".join(valid_boxes))
        shutil.copy(img_path, out_img_path)
        print(f"✅ {image_name}: {len(valid_boxes)} box → ghi OK")
    else:
        shutil.copy(img_path, out_img_path)
        print(f"⚠️ {image_name}: không có box hợp lệ → chỉ copy ảnh")

def process_all():
    for fname in os.listdir(RAW_IMG_DIR):
        convert_and_copy(fname)

if __name__ == "__main__":
    process_all()
