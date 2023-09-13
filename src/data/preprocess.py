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


""" 
    Notes as of 07 Sept:
    - combine aorta and bifurcation point labels, train only one detection model for both ~ done
    - implement pytorch Dataset class & online augmentation
    - don't resize to 640, use original 512 ~ done
    - update EDA notebook after labelling all clinical data and combining
    - write function to extract aorta_mask.nii from seg.nrrd files ~ done
    - modify function to plot both aorta and bifurcation bboxes on same image ~ done
    - Implement 5-fold for train and validation split
"""






def preprocess_data(root_data_folder, destination_folder, resize_to=512):
    """ Accepts a folder containing patients data folders, each of which contain CT image and segmentation mask """
    if os.path.exists(root_data_folder):
        os.makedirs(destination_folder, exist_ok=True)
        for patient_folder in os.listdir(root_data_folder):
            preprocess_patient_data(root_data_folder, patient_folder, destination_folder)
    else:
        print(f'Path doesnt exist: {root_data_folder}')
      
        
       
def preprocess_patient_data(root_folder, patient_name, destination_folder):
    destination_folder = Path(destination_folder)
    ct_path = os.path.join(root_folder + "\\" + patient_name, 'og_ct.nii')
    mask_path = os.path.join(root_folder + "\\" + patient_name, 'aorta_mask.nii')
    bifurcation_path = os.path.join(root_folder + "\\" + patient_name, 'bifurcation.json')
    if os.path.exists(mask_path) and os.path.exists(ct_path) and os.path.exists(bifurcation_path):
        ct_im, mask  = utils.fix_direction(ct_path), utils.fix_direction(mask_path)
        aorta_idxs = utils.get_idxs_segment(mask)
        label_bifurcation, bifurcation_idxs = utils.get_yololabel_from_3Dmarkup(bifurcation_path, ct_path)
        for idx in range(ct_im.shape[0]):
            # Slice 3d images into 2d
            slice_data = ct_im[idx, :, :]
            img = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255
            image_path = destination_folder / 'images' / f'{patient_name}_{str(idx)}.png'
            label_line_aorta, label_line_bifurcation = '', ''
            if idx in aorta_idxs:
                label_aorta = utils.calculate_bbox_from_slice(mask[idx])
                label_line_aorta = '0 ' + ' '.join(map(str, label_aorta))
            if idx in bifurcation_idxs:
                label_line_bifurcation = '1 ' + ' '.join(map(str, label_bifurcation))
            # Save    
            label_path = destination_folder / 'labels' / f'{patient_name}_{str(idx)}.txt'
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_lines = label_line_aorta + '\n' + label_line_bifurcation
            with label_path.open('w') as f:
                f.write(label_lines)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            utils.save_array_as_png(img, image_path)
    else: 
        print(f'Path doesnt exist')
        









if __name__ == '__main__':

    project_path = os.path.dirname(src_dir)
    data_folder_path = os.path.join(project_path, "data\\raw\\annotated_data_bii")
    destination_folder = os.path.join(project_path, "data\\processed")
    preprocess_data(root_data_folder=r'E:\Aibota\annotated_data_bii', destination_folder=destination_folder)
