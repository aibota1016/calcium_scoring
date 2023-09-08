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


""" 
    Notes as of 07 Sept:
    - combine aorta and bifurcation point labels, train only one detection model for both
    - implement pytorch Dataset class & online augmentation
    - don't resize to 640, use original 512
    - update EDA notebook after labelling all clinical data and combining
    - write function to extract aorta_mask.nii from seg.nrrd files ~done
    - write function to plot both aorta and bifurcation bboxes on same image 
    - Use 5:1 fold for train and validation split
"""






def preprocess_data(root_data_folder, destination_folder, resize_to=416, apply_augment=False):
    """ Accepts a folder containing patients data folders, each of which contain CT image and segmentation mask """
    if os.path.exists(root_data_folder):
        # Create the destination folder if it doesn't exist
        os.makedirs(destination_folder, exist_ok=True)
        for patient_folder in os.listdir(root_data_folder):
            mask3d_to_bbox_yolo(root_data_folder, patient_folder, destination_folder, resize_to=resize_to, augment=apply_augment)
    else:
        print(f'Path doesnt exist: {root_data_folder}')
      

def bbox3d_to_yolo(root_folder, patient, destination_folder):
    """Converts 3D slices and 3D box coordinates to each 2D slices with 2D boxes and saves as png & txt"""
    destination_folder = Path(destination_folder)
    ct_path, bbox_path = "", ""
    for filename in os.listdir(root_folder + "\\" + patient):
        current_file_path = os.path.join(root_folder + "\\" + patient, filename)
        if filename.endswith(".nii"):
            ct_path = current_file_path
        elif filename.endswith(".json"): 
            bbox_path = current_file_path

    if os.path.exists(ct_path) and os.path.exists(bbox_path):
        ct_im, spacing = utils.read_nifti_image(ct_path, spacing=True)
        # ct_im = clip_values(ct_im)
        [x,y,z], [w,h,l] = get_3Dbox_coordinates(bbox_path, spacing)
        idxes = get_slice_idxs_with_bbox(z, l, spacing[0])
        for idx in range(ct_im.shape[0]):
            #write 3d images into 2d slice
            slice_data = ct_im[idx, :, :]
            slice_data = aug.clip_values_2d(slice_data)
            slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255
            # Convert the 2D slice to a PIL Image
            image_path = destination_folder / 'images' / f'{patient}_{str(idx)}.png'
            image_path.parent.mkdir(parents=True, exist_ok=True)
            utils.save_array_as_png(slice_data, image_path)
            #write label files if there is a bounding box in the slice
            if idx in idxes:
                # labels = normaliza_relative_to_image([x,y], [w,h], slice_data.shape)
                labels = utils.normalize_bbox([x,y,w,h], slice_data.shape)
                label_line = '1 ' + ' '.join(map(str, labels))
                label_path = destination_folder / 'labels' / f'{patient}_{str(idx)}.txt'
                label_path.parent.mkdir(parents=True, exist_ok=True)
                with label_path.open('w') as f:
                    f.write(label_line)
    else: 
        print(f'Path doesnt exist: {ct_path} or {bbox_path}')




def mask3d_to_bbox_yolo(root_folder, patient, destination_folder, resize_to, augment=False):
    """Function to convert 3D segmentation mask to 2D bounding box for each slice then save
    patient_data_path containing CT scan and segmented aorta mask (both NIFTI files)
    """
    destination_folder = Path(destination_folder)
    mask_path = os.path.join(root_folder + "\\" + patient, 'aorta_mask.nii')
    ct_path = os.path.join(root_folder + "\\" + patient, 'og_ct.nii')
    if os.path.exists(mask_path) and os.path.exists(ct_path):
        ct_im, mask = aug.clip_values_3d(fix_direction(ct_path)), fix_direction(mask_path)
        slices_idx_with_segment = utils.get_idxs_segment(mask)
        for idx in range(ct_im.shape[0]):
            # Slice 3d images into 2d
            slice_data = ct_im[idx, :, :]
            img = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255
            image_path = destination_folder / 'images' / f'{patient}_{str(idx)}.png'
            # write label files if there is a segmentation in the slice
            if idx in slices_idx_with_segment:
                label = calculate_bbox_from_slice(mask[idx])
                label_path = destination_folder / 'labels' / f'{patient}_{str(idx)}.txt'
                if augment:
                    img, label = aug.apply_random_augmentation(img, label)
                    image_path = destination_folder / 'images' / f'{patient}_{str(idx)}_aug.png'
                    label_path = destination_folder / 'labels' / f'{patient}_{str(idx)}_aug.txt'
                #save label
                label_line = '0 ' + ' '.join(map(str, label))
                label_path.parent.mkdir(parents=True, exist_ok=True)
                with label_path.open('w') as f:
                    f.write(label_line)
            # Save slice as png image
            image_path.parent.mkdir(parents=True, exist_ok=True)
            utils.save_array_as_png(aug.resize(img, new_size=resize_to), image_path)
    else: 
        print(f'Path doesnt exist: {mask_path}')
        
       


def get_3Dbox_coordinates(json_path, spacing):
    center, dimension = utils.read_json(json_path)
    x = center[0] / spacing[1]
    y = center[1] / spacing[2]
    z = center[2] / spacing[0]
    center = [x, y, z]
    w = dimension[0] / spacing[1]
    h = dimension[1] / spacing[2]
    l = dimension[2] / spacing[0]
    dimension = [w,h,l]
    return center, dimension


def get_slice_idxs_with_bbox(z, l, spacing_z):
    """ returns the index of slices with bounding box"""
    l = l / spacing_z
    idxs = [int(idx) for idx in range(int(z - l), int(z + l) + 1)]
    return idxs
    
    
def fix_direction(image_path):
    """This function will flip the image axis back to [1,1,1] """
    itk_image = sitk.ReadImage(image_path)
    direction_matrix = np.array(itk_image.GetDirection()).reshape((3, 3))
    flip_axes = [i for i, val in enumerate(np.diag(direction_matrix)) if val < 0]
    
    if flip_axes:
        direction_matrix[flip_axes, :] = -direction_matrix[flip_axes, :]
        itk_image.SetDirection(direction_matrix.flatten())
        
        data_array = sitk.GetArrayFromImage(itk_image)
        data_array = np.flip(data_array, axis=flip_axes)
        itk_image = sitk.GetImageFromArray(data_array)
        itk_image.SetDirection(direction_matrix.flatten())
    
    arr_im = sitk.GetArrayFromImage(itk_image)
    return arr_im



def calculate_bbox_from_slice(mask_slice):
    """Accepts single 2D slice of a binary mask 
    returns normalized center_x, center_y, w, h relative to the image (in yolo label format)
    """
    non_zero_pixels = np.argwhere(mask_slice == 1)
    if len(non_zero_pixels) == 0:
        return None 
    min_row, min_col = np.min(non_zero_pixels, axis=0)
    max_row, max_col = np.max(non_zero_pixels, axis=0)
    im_w, im_h = mask_slice.shape[0], mask_slice.shape[1]
    center_x = (min_col + (max_col - min_col) / 2) / im_w
    center_y = (min_row + (max_row - min_row) / 2) / im_h
    width = (max_col - min_col) / im_w
    height = (max_row - min_row) / im_h
    label = [center_x, center_y, width, height]
    return label





if __name__ == '__main__':
    pass

    # plot
    #images_folder = r"C:\Users\sanatbyeka\Desktop\calcium_scoring\data\processed\bifurcation_point\images"
    #labels_folder = r"C:\Users\sanatbyeka\Desktop\calcium_scoring\data\processed\bifurcation_point\labels"
    #viz.plot_imgs_bboxes(images_folder, labels_folder, title="Sample slices from PD002", rows=2, columns=3, save_path='bifurcation_point_bbox.png')

    #project_path = os.path.dirname(src_dir)
    #aorta_folder_path = os.path.join(project_path, "data\\raw\\aorta")
    #aorta_destination_folder = os.path.join(project_path, "data\\processed\\aorta")
    #preprocess_data(aorta_folder_path, aorta_destination_folder, resize_to=640)
    # preprocess the dataset again with augmentations appled
    #preprocess_data(aorta_folder_path, aorta_destination_folder, resize_to=640, apply_augment=True) 
    
    
    #nrrd_file = r'C:\Users\sanatbyeka\Desktop\bifurcation_markup\PD127\aorta_mask.nrrd'
    #og_ct_path = r'C:\Users\sanatbyeka\Desktop\bifurcation_markup\PD127\og_ct.nii'
    #save_path = r'C:\Users\sanatbyeka\Desktop\bifurcation_markup\PD127\aorta_mask.nii'

    #utils.extract_aorta_mask(nrrd_file, og_ct_path, save_path)
    #viz.plot_masks(fix_direction(og_ct_path), fix_direction(save_path), row=4)
    
    