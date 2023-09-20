

import os
import yaml
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




    
def train_default(model, datayaml_path, project, epoch):
    """ Uses default hyperparameters from Ultralytics default.yaml
    see: https://docs.ultralytics.com/usage/cfg/#train
    """
    result = model.train(
        data = os.path.join(PROJECT_PATH, datayaml_path),
        task = 'detect',
        epochs = epoch,
        verbose = True,
        save = True,
        device = 0,
        project = project,
        seed = 42,
        save_dir=save_dir,
        exist_ok = True,
        plots = True,
        save = True
        )
    
    print(result)

    #results = model.val()
    #success = model.export(format="onnx")
    
def train_SAHI(model, model_path):
    """For small object detection, see: https://docs.ultralytics.com/guides/sahi-tiled-inference/"""
    # pip install -U ultralytics sahi
    from sahi.utils.yolov8 import download_yolov8s_model
    from sahi import AutoDetectionModel
    
    detection_model = AutoDetectionModel.from_pretrained(
    model_type=model,
    model_path=model_path,
    confidence_threshold=0.3,
    device='cuda:0'
)

    
    
def tune_model(model, data_yaml, epoch, itr=300):
    """ Hyperparameter tuning using Genetic Mutation Algorithm """
    model.tune(data=data_yaml, epochs=epoch, iterations=itr, optimizer='AdamW', plots=False, save=False, val=False)
    

def custom_train(model, args):
    """Train using tuned hyperparameters"""
    pass

    
    
    


if __name__ == '__main__':
    datayaml_path = 'data/datasets/data.yaml'
    # Load a model
    model = YOLO('models//yolov8n.pt')
    train_default(model, datayaml_path, 'logs//detection')
    