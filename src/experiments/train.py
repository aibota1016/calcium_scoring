

import os
import sys
from ultralytics import YOLO
from ultralytics import settings


PROJECT_PATH = os.path.dirname(os.path.dirname(os.getcwd()))

settings['datasets_dir'] = os.path.join(PROJECT_PATH, 'data\\datasets')
os.makedirs(os.path.join(PROJECT_PATH, 'experiments\\runs'), exist_ok=True)
os.makedirs(os.path.join(PROJECT_PATH, 'experiments\\weights'), exist_ok=True)
settings['runs_dir'] =  os.path.join(PROJECT_PATH, 'experiments\\runs')
settings['weights_dir'] =  os.path.join(PROJECT_PATH, 'experiments\\weights')
print(settings)
save_dir = os.path.join(PROJECT_PATH, 'logs')
os.makedirs(save_dir, exist_ok=True)




    
def train_yolo(model, datayaml_path, project):
    # Training.
    result = model.train(
        data = os.path.join(PROJECT_PATH, datayaml_path),
        task = 'detect',
        epochs = 3,
        verbose = True,
        batch = 32,
        imgsz = 640,
        patience = 50,
        save = True,
        #device = 0,
        project = project,
        cos_lr = True,
        lr0 = 0.0001,
        lrf = 0.00001,
        warmup_epochs = 3,
        warmup_bias_lr = 0.000001,
        optimizer = 'Adam',
        seed = 42,
        save_dir=save_dir
    )
    
    print(result)

    #results = model.val()
    #success = model.export(format="onnx")
    
    
    
    


if __name__ == '__main__':
    datayaml_path = 'data/datasets/aorta_dataset/data.yaml'
    # Load a model
    model = YOLO('models//yolov8n.pt')
    train_yolo(model, datayaml_path, 'logs//detection')
    