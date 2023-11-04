
import os
import random
import shutil
from pathlib import Path
from collections import Counter
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm
import time



def patient_split_k_folds(dataset_path, test_num, save_path, k=5):
    """ 
    train_num: number of patients data for test, 
    remaining patients data will be split using k-fold cross validation
    """
    print("Reading images and label files... \n")
    patient_files = os.listdir(dataset_path)
    image_files, label_files = [], []
    for patient in patient_files:
        for img in os.listdir(dataset_path/patient/'images'):
            image_files.append(img)
        for lbl in os.listdir(dataset_path/patient/'labels'):
            label_files.append(lbl)
    assert len(label_files) == len(image_files)
    
    print(f'Reserving {test_num} patient files for the test set...\n')
    save_path.mkdir(parents=True, exist_ok=True)
    test_patients, count = [], 0
    for i in range(test_num):
        idx = 0
        test_patient = patient_files[idx]
        while 'aug' in test_patient:
            idx += 1
            test_patient = patient_files[idx]
        shutil.copytree(dataset_path/test_patient, save_path.parent/'test_set'/test_patient, dirs_exist_ok=True)
        test_patients.append(test_patient)
        for test_im in os.listdir(dataset_path/test_patient/'images'):
            image_files.remove(test_im)
            count += 1
        for test_lbl in os.listdir(dataset_path/test_patient/'labels'):
            label_files.remove(test_lbl)
        patient_files.pop(idx)

    print("Patients data reserved for the test: ", test_patients)
    print('Number of 2D slices in the test patients: ', count)

    classes = {0 :'bifurcation'}
    cls_idx = list(range(len(classes)))
    labels_df = pd.DataFrame([], columns=cls_idx, index=patient_files)
    print("\nCounting labels...\n")
    for patient in patient_files:
        if patient not in test_patients:
            labels = os.listdir(dataset_path/patient/'labels')
            lbl_counter = Counter()
            for label in labels:
                lines = []
                with open(os.path.join(dataset_path/patient/'labels', label),'r') as lf:
                    lines.append(lf.readlines())
                for line in lines:
                    for l in line:
                        if l != '\n':
                            lbl_counter[int(l.split(' ')[0])] += 1
                            labels_df.loc[patient] = lbl_counter
    labels_df = labels_df.fillna(0.0)
    print("Total number of patients for the cross-val: ", len(labels_df))
    for i in labels_df.columns:
        print(f"Number of 2D slices in {classes[i]} class for the k-fold cross val set: ", labels_df[i].sum())
        print(f"Average number of {classes[i]} labels per patient: ", round(labels_df[i].mean()))

    print('\n...............................................\n')
    print(f"\nSplitting files into {k} folds... \n")
    kf = KFold(n_splits=k, shuffle=True, random_state=20)
    kfolds = list(kf.split(labels_df))
    folds = [f'split_{n}' for n in range(1, k + 1)]
    folds_df = pd.DataFrame(index=patient_files, columns=folds)
    for idx, (train, val) in enumerate(kfolds, start=1):
        folds_df[f'split_{idx}'].loc[labels_df.iloc[train].index] = 'train'
        folds_df[f'split_{idx}'].loc[labels_df.iloc[val].index] = 'val'
    
    fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)
    for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
        train_totals = labels_df.iloc[train_indices].sum()
        val_totals = labels_df.iloc[val_indices].sum()
        ratio = val_totals / (train_totals + 1E-7)
        fold_lbl_distrb.loc[f'split_{n}'] = ratio
    print("\nDistribution of class labels for each fold as a ratio of the classes present in val to those present in train: \n", fold_lbl_distrb)

    print("\nCreating split folders...")
    print("\nWriting yaml files...")
    ds_yamls = []
    for split in folds_df.columns:
        split_dir = save_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
        (split_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)
        dataset_yaml = split_dir / f'{split}_dataset.yaml'
        ds_yamls.append(dataset_yaml)
        with open(dataset_yaml, 'w') as ds_y:
            yaml.safe_dump({
                'path': str(save_path).split('/')[-1]+ f'/{split}',
                'train': 'train/images',
                'val': 'val/images',
                'names': classes
            }, ds_y)
    
    print("Copying files to each split...")


    for image, label in tqdm(zip(image_files, label_files), total=len(image_files)):
        for split, k_split in folds_df.loc[image.rsplit('_', 1)[0]].items():
            img_to_path = save_path / split / k_split / 'images'
            lbl_to_path = save_path / split / k_split / 'labels'
            shutil.copy(dataset_path/image.rsplit('_', 1)[0]/'images'/image, img_to_path / image)
            shutil.copy(dataset_path/image.rsplit('_', 1)[0]/'labels'/label, lbl_to_path / label)
    print("K-Fold split performed successfully")
    
    folds_df.to_csv(save_path / "kfold_datasplit.csv")
    fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")
    




    """
    temp = dataset_path.parent/'patients_grouped'
    temp.mkdir(parents=True, exist_ok=True)
    Path(temp/patient).mkdir(parents=True, exist_ok=True)
    print("Grouping slices from same patients...\n")
    for i in range(1, len(images)):
        current = str(images[i]).split('_')[0]
        if current != patient:
            patient = current
            patient_files.append(patient)
        Path(temp/patient/'images').mkdir(parents=True, exist_ok=True)
        Path(temp/patient/'labels').mkdir(parents=True, exist_ok=True)
        if not os.path.exists(temp/patient/'images'/images[0]):
            shutil.copy(os.path.join(dataset_path/'images', images[0]), temp/patient/'images'/images[0])
            shutil.copy(os.path.join(dataset_path/'labels', labels[0]), temp/patient/'labels'/labels[0])
        shutil.copy(os.path.join(dataset_path/'images', images[i]), temp/patient/'images'/images[i])
        shutil.copy(os.path.join(dataset_path/'labels', labels[i]), temp/patient/'labels'/labels[i])
    yaml_file = temp/'dataset.yaml'
    shutil.copy(os.path.join(dataset_path, 'dataset.yaml'), yaml_file)
    assert os.path.exists(yaml_file), f"The datasets folder {temp} do not contain the dataset.yaml file"
    with open(yaml_file, 'r', encoding="utf8") as y:
        classes = yaml.safe_load(y)['names']

    """
    





def k_fold_split(dataset_path, save_path, k=5):
    print("Reading label files... \n")
    labels = sorted(dataset_path.rglob("*labels/*.txt"))
    images = sorted((dataset_path / 'images').rglob("*.png"))  
    assert len(labels) == len(images)

    yaml_file = dataset_path / "dataset.yaml"
    assert os.path.exists(yaml_file), f"The datasets folder {dataset_path} do not contain the dataset.yaml file"
    with open(yaml_file, 'r', encoding="utf8") as y:
        classes = yaml.safe_load(y)['names']
 
    cls_idx = list(range(len(classes)))
    indx = [l.stem for l in labels]
    labels_df = pd.DataFrame([], columns=cls_idx, index=indx)
    
    for label in labels:
        lbl_counter = Counter()
        with open(label,'r') as lf:
            lines = lf.readlines()
        for l in lines:
            if l != '\n':
                lbl_counter[int(l.split(' ')[0])] += 1
                labels_df.loc[label.stem] = lbl_counter
    labels_df = labels_df.fillna(0.0)
    for i in labels_df.columns:
        print(f"Number of instances in {classes[i]} class: ", labels_df[i].sum())
    
    print(f"\nSplitting files into {k} folds... \n")
    kf = KFold(n_splits=k, shuffle=True, random_state=20)
    kfolds = list(kf.split(labels_df))
    folds = [f'split_{n}' for n in range(1, k + 1)]
    folds_df = pd.DataFrame(index=indx, columns=folds)
    print("Number of total data: ", len(folds_df))
    for idx, (train, val) in enumerate(kfolds, start=1):
        folds_df[f'split_{idx}'].loc[labels_df.iloc[train].index] = 'train'
        folds_df[f'split_{idx}'].loc[labels_df.iloc[val].index] = 'val'
    print('...............................................\n')

    fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)
    for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
        train_totals = labels_df.iloc[train_indices].sum()
        val_totals = labels_df.iloc[val_indices].sum()
        ratio = val_totals / (train_totals + 1E-7)
        fold_lbl_distrb.loc[f'split_{n}'] = ratio
    print("\nDistribution of class labels for each fold as a ratio of the classes present in val to those present in train: \n", fold_lbl_distrb)
    
    # Create the directories and dataset YAML files for each split
    print("Creating split folders...")
    save_path.mkdir(parents=True, exist_ok=True)
    print("Creating yaml files...")
    ds_yamls = []
    for split in folds_df.columns:
        split_dir = save_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
        (split_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)
        # Create dataset YAML files
        dataset_yaml = split_dir / f'{split}_dataset.yaml'
        ds_yamls.append(dataset_yaml)
        with open(dataset_yaml, 'w') as ds_y:
            yaml.safe_dump({
                'path': str(save_path).split('/')[-1]+ f'/{split}',
                'train': 'train/images',
                'val': 'val/images',
                'names': classes
            }, ds_y)

    print("Copying files to each split...")
    for image, label in tqdm(zip(images, labels), total=len(images)):
        for split, k_split in folds_df.loc[image.stem].items():
            img_to_path = save_path / split / k_split / 'images'
            lbl_to_path = save_path / split / k_split / 'labels'
            shutil.copy(image, img_to_path / image.name)
            shutil.copy(label, lbl_to_path / label.name)
    print("K-Fold split performed successfully")
    
    folds_df.to_csv(save_path / "kfold_datasplit.csv")
    fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")




    

def train_val_split(source_images_folder, source_labels_folder, destination_folder, train_ratio=0.8):
    # read files
    image_files = []
    for image_file in os.listdir(source_images_folder):
        image_files.append(image_file)
    image_files = sorted(image_files)
    # split into training, testing, validation sets
    random.seed(100)
    random.shuffle(image_files)
    split = int(train_ratio * len(image_files))
    #split_2 = int((train_ratio + (1-train_ratio)/2) * len(image_files))
    train_images = image_files[:split]
    #test_images = image_files[split_1:split_2]
    val_images = image_files[split:]
    print(f"Number of train images: {len(train_images)}\n", 
          f"validation images: {len(val_images)}")
    # write to a destination folder
    destination_folder.mkdir(parents=True, exist_ok=True)
    splits = [train_images, val_images]
    for i in range(2):
        if i == 0:
            split_folder = 'train'
            #elif i == 1:
            #    split_folder = 'test'
        else:
            split_folder = 'val'
        for image in splits[i]:
            image_path = os.path.join(source_images_folder, image) 
            destination_image_path = destination_folder / split_folder / "images" / image
            destination_image_path.parent.mkdir(parents=True, exist_ok=True)
            label_file = image.split('.')[0] + '.txt'
            label_path = os.path.join(source_labels_folder, label_file)
            if os.path.exists(label_path):
                destination_label_path = destination_folder / split_folder / "labels" / label_file
                destination_label_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(label_path, destination_label_path)
            shutil.copy2(image_path, destination_image_path)
            print("Image copied to ", destination_image_path)

 
    
    
if __name__ == '__main__':
        
    dataset_path = Path('../calcium_scoring/data/processed')
    save_path = Path(dataset_path.parent / 'datasets')
    
    #k_fold_split(dataset_path/'bifurcation', save_path/'bifurcation_dataset_split')
    
    #train_val_split(dataset_path/'bifurcation/images', dataset_path/'bifurcation/labels', save_path/'bifurcation_train_val_split')
    
    
    patient_split_k_folds(dataset_path/'bifurcation_grp_oversampled', test_num=2, save_path=save_path/'test_oversampled', k=5)
    

    dataset_path = Path('/home/sanatbyeka/calcium_scoring/data/processed')
    save_path = Path(dataset_path.parent / 'datasets')

    #k_fold_split(dataset_path/'bifurcation', save_path/'bifurcation_split2')