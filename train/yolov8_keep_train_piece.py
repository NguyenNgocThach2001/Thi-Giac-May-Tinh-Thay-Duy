from ultralytics import YOLO

def train_yolov8_resume():
    # === CẤU HÌNH ===
    pretrained_weights = "runs/detect/model_on_dataset3_continue2_continue_continue/weights/last.pt"  # model đã train trước đó
    data_yaml = "dataset/piece/Dataset4_270image/data.yaml"              # file cấu hình dataset
    epochs = 100        # train thêm 50 epochs nữa
    imgsz = 640
    batch = 8

    # Load model đã huấn luyện
    model = YOLO(pretrained_weights)

    # Train tiếp tục
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="model_on_dataset3_continue2_continue_continue_continue",
        resume=False  # Không dùng checkpoint Ultralytics, chỉ load trọng số từ last.pt
    )

    print("✅ Tiếp tục huấn luyện hoàn tất!")

if __name__ == "__main__":
    train_yolov8_resume()
