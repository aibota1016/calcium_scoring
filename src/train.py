

import os
from pathlib import Path
from ultralytics import YOLO
from ultralytics import settings
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
    #train_default_kfold_split(model=model_path, k_split=5, data_path=data_path, project_name='experiments/bifurcation', epoch=300)



    project_path = "/Users/aibotasanatbek/Documents/projects/calcium_scoring"
    data_path = os.path.join(project_path, 'data/datasets/train_val')
    dataset_yaml = os.path.join(data_path, 'split_5/split_5_dataset.yaml')

    model = YOLO('yolov8n.pt')

    
    """ 
    lr0: 0.01298
lrf: 0.0125
momentum: 0.98
weight_decay: 0.00062
warmup_epochs: 2.8244
warmup_momentum: 0.63642
box: 5.81596
cls: 0.69746
dfl: 1.44072
hsv_h: 0.01267
hsv_s: 0.55956
hsv_v: 0.3003
degrees: 0.0
translate: 0.07682
scale: 0.36183
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.20583
mosaic: 0.80822
mixup: 0.0
copy_paste: 0.0
    """
    model.train(data=dataset_yaml, epochs=5, single_cls=True, imgsz=512, project='experiments/final', mosaic=0, patience=0, save_json=True)

    
