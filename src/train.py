

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
from ultralytics import settings
from ultralytics.engine.model import Model
from utils import load_model
from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel



    
    
def train_default_kfold_split(model, k_split, data_path, project_name, epoch):
    results = {}
    ds_yamls = []
    for split_folder in os.listdir(data_path):
        split_folder_path = os.path.join(data_path, split_folder)
        if os.path.isdir(split_folder_path):
            for yml_file in os.listdir(split_folder_path):
                if yml_file.endswith(".yaml"):
                    ds_yamls.append(os.path.join(split_folder_path, yml_file))
        
    for k in range(k_split):
        print("FOLD NUMBER: ", k+1)
        dataset_yaml = ds_yamls[k]
        model.train(
            data=dataset_yaml,
            epochs=epoch, 
            imgsz=512, 
            project=project_name, 
            close_mosaic=0)  
        results[k] = model.metrics
        print('###########################################################################################\n')
    print(results)

    
    
def inference_SAHI(model, model_path):
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
    


    
    


if __name__ == '__main__':
    
    project_path = r'E:\Aibota\calcium_scoring'
    #settings['datasets_dir'] = os.path.join(project_path, 'data\\datasets')
    #settings['runs_dir'] =  os.path.join(project_path, 'experiments\\runs')
    #settings['weights_dir'] =  os.path.join(project_path, 'experiments\\weights')


    data_path = os.path.join(project_path, 'data/datasets/bifurcation_dataset_split')
    for i in range(5):
        model = YOLO(f'model{i}/yolov8n.pt')
    #train_default_kfold_split(model=model_path, k_split=5, data_path=data_path, project_name='experiments/bifurcation', epoch=300)



    # Train using the best model
    #best_model_path = "/home/sanatbyeka/calcium_scoring/src/experiments/mix/train5/weights/best.pt"
    #best_model = YOLO(best_model_path)
    #oversampled_data_path = os.path.join(project_path, 'data//datasets//oversample_split')
    #train_default_kfold_split(model=best_model, k_split=5, data_path=oversampled_data_path, project_name='experiments/oversampled', epoch=50, save_dir=save_dir)
    

    
        