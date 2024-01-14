import numpy as np
import os
import sys
from pathlib import Path
import SimpleITK as sitk
import transform
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_dir)
import utils
from visualizations import viz
import nibabel as nib
import transform
import pandas as pd
import random
import shutil


def preprocess_data(root_data_folder, train_dest_folder, test_dest_folder, test_num):
    """ Accepts a folder containing patients data folders, each of which contain CT image and segmentation mask 
    Specify oversampling_factor only if the oversample is set to True
    """
    if os.path.exists(root_data_folder):
        os.makedirs(train_dest_folder, exist_ok=True)
        train_dest_folder = Path(train_dest_folder)
        test_patients = []
        for patient_folder in os.listdir(root_data_folder):
            patient_folder_path = os.path.join(root_data_folder, patient_folder)
            if os.path.isdir(patient_folder_path):
                LM_calcium_label = [file for file in os.listdir(patient_folder_path) if file.endswith('.xls')]
                if len(LM_calcium_label) > 0 and len(test_patients) < test_num:
                        test_patients.append(patient_folder)
                        preprocess_test_data(root_data_folder, patient_folder, test_dest_folder)
                else:
                    preprocess_bifurcation_data(root_data_folder, patient_folder, train_dest_folder, include_background=True)                        
        test_line = ''
        for patient in test_patients:
            test_line += patient
            test_line += " "
        with open(train_dest_folder.parent/"test_patients.txt", 'w') as f:
            f.write(f"Test patients: \n {str(test_line)} ")
        print("Name of test patients saved to: ", train_dest_folder.parent/"test_patients.txt")
    else:
        print(f'Root data folder path doesnt exist: {root_data_folder}')




def preprocess_bifurcation_data(root_folder, patient_name, destination_folder, include_background=False):
    ct_path = os.path.join(root_folder + "/" + patient_name, 'og_ct.nii')
    bifurcation_path = os.path.join(root_folder + "/" + patient_name, 'bifurcation.json')
    if os.path.exists(ct_path):
        ct_im  = utils.fix_direction(ct_path)
        label_bifurcation, bifurcation_idxs = utils.get_yololabel_from_3Dmarkup(bifurcation_path, ct_path)
        for idx in range(ct_im.shape[0]):
            # Slice 3d images into 2d
            #slice_data = transform.clip_values_3d(ct_im[idx, :, :])
            slice_data = ct_im[idx, :, :]
            img = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255
            image_path = destination_folder / patient_name / 'images' / f'{patient_name}_{str(idx)}.png'
            if idx in bifurcation_idxs:
                label_line_bifurcation = '0 ' + ' '.join(map(str, label_bifurcation))
                # Save
                label_path = destination_folder / patient_name / 'labels' / f'{patient_name}_{str(idx)}.txt'
                label_path.parent.mkdir(parents=True, exist_ok=True)
                with label_path.open('w') as f:
                    f.write(label_line_bifurcation)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                utils.save_array_as_png(img, image_path)

            if include_background is True and idx not in bifurcation_idxs:
                label_path = destination_folder / patient_name / 'labels' / f'{patient_name}_{str(idx)}.txt'
                label_path.parent.mkdir(parents=True, exist_ok=True)
                with label_path.open('w') as f:
                    f.write('') #empty label as just a background
                image_path.parent.mkdir(parents=True, exist_ok=True)
                utils.save_array_as_png(img, image_path)  
    else:
        print(f'CT image path doesnt exist')

       


def preprocess_test_data(root_folder, patient_name, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)
    destination_folder = Path(destination_folder)
    ct_path = os.path.join(root_folder + "/" + patient_name, 'og_ct.nii')
    bifurcation_path = os.path.join(root_folder + "/" + patient_name, 'bifurcation.json')
    if os.path.exists(ct_path):
        ct_im  = utils.fix_direction(ct_path)
        label_bifurcation, bifurcation_idxs = utils.get_yololabel_from_3Dmarkup(bifurcation_path, ct_path)
        for idx in range(ct_im.shape[0]):
            #slice_data = transform.clip_values_3d(ct_im[idx, :, :])
            slice_data = ct_im[idx, :, :]
            img = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255
            image_path = destination_folder / 'images' / f'{patient_name}_{str(idx)}.png'
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path = destination_folder / 'labels' / f'{patient_name}_{str(idx)}.txt'
            label_path.parent.mkdir(parents=True, exist_ok=True)
            if idx in bifurcation_idxs:
                label_line_bifurcation = '0 ' + ' '.join(map(str, label_bifurcation))
            else:
                label_line_bifurcation = '' #empty label as just a background
            with label_path.open('w') as f:
                f.write(label_line_bifurcation)
            utils.save_array_as_png(img, image_path)
    else:
        print(f'CT image path doesnt exist')




def oversampler(train_folder, oversampling_factor):
    """Oversamples bifurcation images"""
    train_folder = Path(train_folder)
    if os.path.exists(train_folder):
        image_folder = train_folder / 'images'
        labels_folder = train_folder / 'labels'
        for image_file in os.listdir(image_folder):
            image_path = image_folder/ image_file
            label_path = labels_folder / f'{image_file.split(".")[0]}.txt'
            label_line = utils.read_label_txt(label_path)
            print("len(label_line): ", len(label_line))
            for i in range(oversampling_factor):
                if len(label_line) != 0:
                    aug_im, aug_label = transform.apply_random_augmentation(utils.image_to_numpy(image_path), label_line[1:])
                    label_bifurcation_aug = '0 ' + ' '.join(map(str, aug_label))
                    aug_label_path = labels_folder / f'{image_file.split(".")[0]}_aug{i}.txt'
                    aug_label_path.parent.mkdir(parents=True, exist_ok=True)
                    with aug_label_path.open('w') as f:
                        f.write(label_bifurcation_aug)
                    aug_im_path = image_folder / f'{image_file.split(".")[0]}_aug{i}.png'
                    aug_im_path.parent.mkdir(parents=True, exist_ok=True)
                    utils.save_array_as_png(aug_im, aug_im_path)
    else: 
        print(f'Data folder path doesnt exist')
 


def create_df(labels_folder):
    train_images, train_labels = [], []
    for lbl in labels_folder.glob('*.txt'):
        coord = utils.read_label_txt(labels_folder/lbl)
        if len(coord) != 0:
            train_labels.append('bifurcation')
        else:
            train_labels.append('background')
        train_images.append('_'.join(lbl.parts[-1].split('.')[0:-1]))
    train_df = pd.DataFrame(data={'image': train_images, 'label': train_labels})
    return train_df


def undersampler(train_folder, undersampling_factor=0.3):
    train_folder = Path(train_folder)
    image_folder = train_folder / 'images'
    labels_folder = train_folder / 'labels'

    imbalanced_train_df = create_df(labels_folder)
    background_rows = imbalanced_train_df[imbalanced_train_df['label'] == 'background']
    # Determine the number of rows to keep for random undersampling
    undersampled_background = background_rows.sample(frac=undersampling_factor, random_state=42)
    bifurcation_rows = imbalanced_train_df[imbalanced_train_df['label'] != 'background']
    # Concatenate the sampled background rows with non-background rows
    undersampled_df = pd.concat([undersampled_background, bifurcation_rows])
    # Shuffle the DataFrame to mix the rows
    undersampled_df = undersampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
    for image_file in os.listdir(image_folder):
        if image_file.split(".")[0] not in undersampled_df['image'].values:
            os.remove(image_folder / image_file)
            os.remove(labels_folder / f'{image_file.split(".")[0]}.txt')
    return imbalanced_train_df, undersampled_df




def delete_empty_labels(labels_folder):
    labels_folder = Path(labels_folder)
    for label_file in os.listdir(labels_folder):
        label_path = labels_folder / label_file
        label_line = utils.read_label_txt(label_path)
        if len(label_line) == 0:
            os.remove(label_path)
            print(f'{label_file} deleted')










if __name__ == '__main__':

    project_path = os.path.dirname(src_dir)
    data_folder_path = os.path.join(project_path, "data/raw/annotated_data_bii")
    process_folder = os.path.join(project_path, "data/processed")
    test_folder = os.path.join(project_path, "data/datasets/test_set")



    preprocess_data(data_folder_path, os.path.join(process_folder, "bifurcation"), test_folder, test_num=3)


    "" 
    train_folder = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/datasets/train_val/split_3/train'

    before_df = create_df(Path(train_folder)/'labels')
    print("Number of bifurcation images before oversampling: ", len(before_df[before_df['label'] == 'bifurcation']))
    print("Number of background images before undersampling: ", len(before_df[before_df['label'] == 'background']))

    oversampler(train_folder, oversampling_factor=4)
    undersampler(train_folder, undersampling_factor=0.3)


    after_df = create_df(Path(train_folder)/'labels')
    print("Number of bifurcation images after oversampling: ", len(after_df[after_df['label'] == 'bifurcation']))
    print("Number of background images after undersampling: ", len(after_df[after_df['label'] == 'background']))
    

    labels_folder = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/datasets/train_val/split_1/train/labels'
    delete_empty_labels(labels_folder)
    


