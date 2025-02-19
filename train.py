from ultralytics import YOLOv10, IncrementalYOLOv10

if __name__ == '__main__':
    IncrementalYOLOv10(model=r'ultralytics\cfg\models\v10\yolov10x.yaml').train()
    # YOLOv10(model=r'ultralytics\cfg\models\v10\yolov10x.yaml').train()
