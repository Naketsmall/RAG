from src.model_loaders import YOLOLoader


model = YOLOLoader.load_model('../models/yolo11s.pt')
print(model.names)
