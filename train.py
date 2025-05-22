from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(data = "data.yaml", imgsz=416, batch=8, epochs=50, workers=1, device="cpu")