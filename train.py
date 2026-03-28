from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="config/data.yaml",
        epochs=10,
        batch=32,
        device="cuda",
        workers=1,
        patience=20,
        # name='my_rps_train_v1_fixed',
    )


if __name__ == "__main__":
    main()
