
import os
import yaml
from pathlib import Path
from ultralytics import YOLO
from ultralytics import settings
from ultralytics.engine.model import Model




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


def train_default_kfold_split(k_split, data_path, project_name, epoch):
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
        model = YOLO(f'model_new{k}/yolov8s.pt')
        model.train(
                data=dataset_yaml,
                epochs=epoch,
                single_cls=True,
                imgsz=512,
                project=project_name,
                mosaic=0,
                patience=0)
        results[k] = model.metrics
        print('###########################################################################################\n')
    #print(results)
        


def train_tuned_kfold_split(k_split, data_path, project_name, epoch, bs=16):
    """ Changed hyperparameters, disabled early stopping, disabled mosaic augmentation"""
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
        model = YOLO(f'model{k}/yolov8n.pt')
        model.train(
                data=dataset_yaml,
                batch=bs,
                epochs=epoch,
                single_cls=True,
                imgsz=512,
                project=project_name,
                save_json=True,
                iou=0.4,
                max_det=3,
                mosaic=0,
                patience=0,
                lr0=0.005,
                lrf=0.005,
                momentum=0.8,
                weight_decay=0.0002,
                warmup_epochs=4,
                warmup_momentum=0.95,
                box=6.0,
                cls=0.2,
                dfl=1.8,
                hsv_h=0.02,
                hsv_s=0.5,
                hsv_v=0.3,
                translate=0.08,
                scale=0.3,
                fliplr=0.2,
                seed=k
                )
        results[k] = model.metrics
        print('###########################################################################################\n')
    return results


def tune_model(model, data_yaml, epoch, itr=300):
    """ Hyperparameter tuning using Genetic Mutation Algorithm """
    model.tune(data=data_yaml, epochs=epoch, iterations=itr, single_cls=True, imgsz=512, close_mosaic=0, plots=False, save=False, val=False)





if __name__ == '__main__':

    project_path = "/home/sanatbyeka/calcium_scoring"
    data_path = os.path.join(project_path, 'data/datasets/new')


    dataset_yaml = os.path.join(data_path, 'split_3/split_3_dataset.yaml')

    model = YOLO('yolov8n.pt')

    #tune_model(model=model, data_yaml=dataset_yaml, epoch=30)




