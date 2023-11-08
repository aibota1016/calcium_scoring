import numpy as np
import os
import sys
from pathlib import Path
import SimpleITK as sitk
import transform as aug
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_dir)
import utils
from visualizations import viz
import nibabel as nib
import transform
import yaml



def preprocess_data(root_data_folder, destination_folder, mode, oversampling_factor=3, test_patients=None):
    """ Accepts a folder containing patients data folders, each of which contain CT image and segmentation mask 
    mode = default, aorta, bifurcation, oversample
    Specify oversampling_factor only if the mode is set to oversample
    """
    if os.path.exists(root_data_folder):
        os.makedirs(destination_folder, exist_ok=True)
        destination_folder = Path(destination_folder)
        i = 0
        for patient_folder in os.listdir(root_data_folder):
            patient_folder_path = os.path.join(root_data_folder, patient_folder)
            if os.path.isdir(patient_folder_path):
                if mode == 'bifurcation':
                    preprocess_bifurcation_data(root_data_folder, patient_folder, destination_folder, include_background=False)
                elif mode == 'oversample':
                    background = False
                    if i % 3 == 0:
                        background = True
                    preprocess_data_oversample(root_data_folder, patient_folder, destination_folder, oversampling_factor, include_background=background)
                    i = i + 1
                elif mode == 'test':
                    if patient_folder in test_patients:
                        preprocess_test_data(root_data_folder, patient_folder, destination_folder)
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


def preprocess_data_oversample(root_folder, patient_name, destination_folder, oversampling_factor=3, include_background=True):
    """Oversamples bifurcation class"""
    ct_path = os.path.join(root_folder + "/" + patient_name, 'og_ct.nii')
    bifurcation_path = os.path.join(root_folder + "/" + patient_name, 'bifurcation.json')
    if os.path.exists(ct_path):
        ct_im = utils.fix_direction(ct_path)
        label_bifurcation, bifurcation_idxs = utils.get_yololabel_from_3Dmarkup(bifurcation_path, ct_path)
        for idx in range(ct_im.shape[0]):
            # Slice 3d images into 2d
            slice_data = ct_im[idx, :, :]
            img = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255
            image_path = destination_folder / patient_name / 'images' / f'{patient_name}_{str(idx)}.png'
            label_path = destination_folder / patient_name / 'labels' / f'{patient_name}_{str(idx)}.txt'
            if idx in bifurcation_idxs:
                label_line_bifurcation = '0 ' + ' '.join(map(str, label_bifurcation))
                for i in range(1, oversampling_factor):
                    aug_im, aug_label = transform.apply_random_augmentation(img, label_bifurcation)
                    label_bifurcation_aug = '0 ' + ' '.join(map(str, aug_label))
                    aug_label_path = destination_folder / f'{patient_name}_aug{i}' / 'labels' / f'{patient_name}_aug{i}_{str(idx)}.txt'
                    aug_label_path.parent.mkdir(parents=True, exist_ok=True)
                    with aug_label_path.open('w') as f:
                        f.write(label_bifurcation_aug)
                    aug_im_path = destination_folder / f'{patient_name}_aug{i}' / 'images' / f'{patient_name}_aug{i}_{str(idx)}.png'
                    aug_im_path.parent.mkdir(parents=True, exist_ok=True)
                    utils.save_array_as_png(aug_im, aug_im_path)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                with label_path.open('w') as f:
                    f.write(label_line_bifurcation)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                utils.save_array_as_png(img, image_path)
                    
            if include_background is True and idx not in bifurcation_idxs:
                label_path.parent.mkdir(parents=True, exist_ok=True)
                with label_path.open('w') as f:
                    f.write('') #empty label as just a background
                image_path.parent.mkdir(parents=True, exist_ok=True)
                utils.save_array_as_png(img, image_path)
    else: 
        print(f'CT image path doesnt exist')

       



def preprocess_test_data(root_folder, patient_name, destination_folder):
    ct_path = os.path.join(root_folder + "/" + patient_name, 'og_ct.nii')
    bifurcation_path = os.path.join(root_folder + "/" + patient_name, 'bifurcation.json')
    if os.path.exists(ct_path):
        ct_im  = utils.fix_direction(ct_path)
        label_bifurcation, bifurcation_idxs = utils.get_yololabel_from_3Dmarkup(bifurcation_path, ct_path)
        for idx in range(ct_im.shape[0]):
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


 








if __name__ == '__main__':

    project_path = os.path.dirname(src_dir)
    data_folder_path = os.path.join(project_path, "data/raw/annotated_data_bii")
    destination_folder = os.path.join(project_path, "data/processed")

    with open(os.path.join(project_path, 'data/raw/test_patients.txt')) as f:
        test_patients = f.readlines()
    test_patients = test_patients[1].strip().split(" ")
    #test_patients = ['PD059', 'PD154', 'PD024', 'PD053', 'PD129', 'PD220', 'PD188', 'PD085', 'PD111', 'PD166', 'PD061', 'PD016', 'PD182', 'PD218', 'PD265', 'PD212', 'PD037', 'PD040', 'PD147', 'PD130', 'PD233', 'PD244', 'PD239', 'PD108', 'PD072', 'PD005', 'PD096', 'PD102', 'PD175', 'PD078']
    preprocess_data(data_folder_path, os.path.join(project_path, "data/datasets/test_set"), mode='test', test_patients=test_patients)

    #preprocess_data(data_folder_path, os.path.join(destination_folder, "overampled_grp"), mode='oversample')


