from ultralytics import YOLO

model = YOLO("best.pt")

model.predict(
    source="tes2.jpg",
    conf=0.5,
    show=True,
    save=True
)
