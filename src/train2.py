

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
from ultralytics import settings
from ultralytics.engine.model import Model
from utils import load_model
from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel



    
def train_default(model, datayaml_path, project, epoch):
    """ Uses default hyperparameters from Ultralytics default.yaml
    see: https://docs.ultralytics.com/usage/cfg/#train
    """
    result = model.train(
        data = datayaml_path,
        task = 'detect',
        epochs = epoch,
        verbose = True,
        save = True,
        device = 0,
        project = project,
        seed = 42,
        save_dir='logs',
        exist_ok = True,
        plots = True,
        )
    
    print(result)

    #results = model.val()
    #success = model.export(format="onnx")
    
    
def train_default_kfold_split(model, k_split, data_path, project_name, epoch, save_dir, resume=False):
    results = {}
    ds_yamls = []
    for split_folder in os.listdir(data_path):
        split_folder_path = os.path.join(data_path, split_folder)
        if os.path.isdir(split_folder_path):
            for yml_file in os.listdir(split_folder_path):
                if yml_file.endswith(".yaml"):
                    ds_yamls.append(os.path.join(split_folder_path, yml_file))
        
    for k in range(k_split):
        print("FOLD NUMBER: ", k)
        dataset_yaml = ds_yamls[k]
        model.train(resume=resume, data=dataset_yaml,epochs=epoch, imgsz=512, project=project_name, save_dir=save_dir)  
        results[k] = model.metrics
        print('###########################################################################################\n')
    print(results)

    
    
def train_SAHI(model, model_path):
    """For small object detection, 
    see: https://docs.ultralytics.com/guides/sahi-tiled-inference/"""
    # pip install -U ultralytics sahi
    
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
    
    project_path = "/home/sanatbyeka/calcium_scoring/"
    save_dir = os.path.join(project_path, 'logs')
    os.makedirs(save_dir, exist_ok=True)
    
    
    # Load a model
    #model_path = os.path.join(project_path, 'models', 'yolov8n.pt')
    #loaded_model = load_model(model_path)
    #assert type(loaded_model) != dict
    #data_path = os.path.join(project_path, 'data//datasets//aorta_bifurcation_mix_60data')
    #train_default_kfold_split(model=loaded_model, k_split=5, data_path=data_path, project_name='experiments//train_60data', epoch=10, save_dir=save_dir)
    
    
    #resume from last model
    #import torch
    best_model_path = "/home/sanatbyeka/calcium_scoring/src/experiments/mix/train5/weights/best.pt"
    best_model = YOLO(best_model_path)
    oversampled_data_path = os.path.join(project_path, 'data//datasets//oversample_split')
    #resumed_model = load_model(resumed_model_path)
    #assert type(resumed_model) != dict
    train_default_kfold_split(model=best_model, k_split=5, data_path=oversampled_data_path, project_name='experiments//oversampled', epoch=50, save_dir=save_dir)
    

