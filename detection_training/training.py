from ultralytics import YOLO

model = YOLO("yolo11l")

results=model.train(data='Ski_Slopes1.yaml', epochs=100, imgsz=1088, freeze=11, fraction= 1.0, device=(0,1,2), batch=36, plots=True, single_cls=True, 
optimizer='AdamW', scale=0.8 ,lr0=0.0000001, mosaic=0.2, erasing=0.0)
