from ultralytics import YOLO

model = YOLO("custom-trained-model.pt")

model.predict(source = 'test/videos', save = True)