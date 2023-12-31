import os


def check_raw_data(data_root_folder):
    """
    Given a root folder containing subdirectories of each patient, ensures that each patient directory contains 
    - og_ct.nii: CT image in nifti format
    - aorta_mask.nii: aorta segmentation file on nifti format
    - bifurcation.json: bifurcation point markup of 3 bounding box 
    """
    assert os.path.exists(data_root_folder), "Given data folder doesnt exist"
    for PD in os.listdir(data_root_folder):
        data_folder_path = os.path.join(data_root_folder, PD)
        if os.path.isdir(data_folder_path):
            assert 'og_ct.nii' in os.listdir(data_folder_path), f"CT image og_ct.nii doesnt exist in the folder: {PD}"
            assert 'aorta_mask.nii' in os.listdir(data_folder_path), f"Aorta segmentation mask aorta_mask.nii doesnt exist in the folder: {PD}"
            assert 'bifurcation.json' in os.listdir(data_folder_path), f"3D bifurcation point markup bifurcation.json doesnt exist in the folder: {PD}"
    
    
    
    
    
if __name__ == '__main__':
    PROJECT_PATH = os.getcwd()
    assert PROJECT_PATH.split('\\')[-1] == 'calcium_scoring'
    raw_data_folder = os.path.join(PROJECT_PATH, 'data\\raw\\annotated_data_bii')
    check_raw_data(raw_data_folder)