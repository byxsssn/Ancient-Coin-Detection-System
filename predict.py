from ultralytics import YOLO


def main():
    model = YOLO("weights/v8_0.995.pt")
    metrics = model.val(
        data="config/data.yaml",
        split="test",
        workers=1,
        batch=24,
        # show = True,
        # source=0,
        # stream=True,
    )


if __name__ == "__main__":
    main()
