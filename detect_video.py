from ultralytics import YOLO

print("Mulai load model...")
model = YOLO("best.pt")

print("Mulai deteksi video...")
model.predict(
    source="video.mp4",
    conf=0.5,
    show=True,
    save=True
)

print("Selesai")
