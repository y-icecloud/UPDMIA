import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

def run_train_classifier(args):
    model = YOLO(model=r'./ultralytics/cfg/models/11/yolo11.yaml')
    model.train(data=r'./ultralytics/cfg/datasets/train_mydata.yaml',
                imgsz=640,
                epochs=50,
                batch=64,
                workers=1,
                device='',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                )