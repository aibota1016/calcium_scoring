
import os
import json
import SimpleITK as sitk
from PIL import Image
import numpy as np
import nrrd
import nibabel as nib
import torch


def read_nifti_image(ct_path, only_img=True):
    """Reads NIFTI image and returns it as a 3D numpy array"""
    if os.path.exists(ct_path):
        itkimage = sitk.ReadImage(ct_path)
        ct_scan = sitk.GetArrayFromImage(itkimage)
        if not only_img:
            spacing = np.array(list(itkimage.GetSpacing()))
            origin = np.array(list(itkimage.GetOrigin()))
            direction = np.array(list(itkimage.GetDirection())).reshape((3,3))
            return ct_scan, spacing, origin, direction
        return ct_scan
    else:
        print("The file path doesn't exist")
        
def read_nifti_nibabel(ct_path):
    ct_nib = nib.load(ct_path)
    #print('original orientation:', nib.orientations.ornt2axcodes(nib.orientations.io_orientation(ct_nib.affine)))
    orig_ornt = nib.orientations.io_orientation(ct_nib.affine)
    targ_ornt = nib.orientations.axcodes2ornt("LPS")
    transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
    img_orient = ct_nib.as_reoriented(transform)
    print('final orientation:', nib.orientations.ornt2axcodes(nib.orientations.io_orientation(img_orient.affine)))
    #viz.plot_scan_slices(ct_nib.get_fdata())
    print("affine: ", img_orient.affine)
    
        
def extract_aorta_mask(segmentation_path, corresponding_ct_path, save_path):
    """Takes path to the seg.nrrd file and path to its corresponding ct image, extracts segmented aorta_mask and saves as nifti file"""
    if os.path.exists(segmentation_path): 
        data, header = nrrd.read(segmentation_path)
        landmark_number = -1
        if header['Segment0_Name'] == 'aorta_mask':
            landmark_number = header['Segment0_LabelValue']
        elif header['Segment1_Name'] == 'aorta_mask':
            landmark_number = header['Segment1_LabelValue']
        if landmark_number != -1:
            landmark_mask = np.where(data == int(landmark_number), data, 0)
            im = nib.load(corresponding_ct_path)
            nifti_img = nib.Nifti1Image(landmark_mask, affine=im.affine)
            try:
                nib.save(nifti_img, save_path)
                print(f"Image saved to {save_path}")
            except Exception as e:
                print(f"An error occurred while saving the image: {e}")
        else:
            print(f"Couldn't find aorta segmentation in {segmentation_path}")
    else:
        print("Segmentation file path doesn't exist")
        
        
        
def save_extracted_aorta_masks(root_data_folder):
    for patient_folder in os.listdir(root_data_folder):
        PD = os.path.join(root_data_folder, patient_folder)
        if os.path.isdir(PD):
            if 'aorta_mask.nii' not in os.listdir(PD) and 'segmentation.seg.nrrd' in os.listdir(PD):
                extract_aorta_mask(os.path.join(PD,'segmentation.seg.nrrd'), os.path.join(PD,'og_ct.nii'), os.path.join(PD,'aorta_mask.nii'))
            
            
            
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


   

def read_json(json_path, orientation=False):
    """ reads 3D bounding box coordinates from json file """    
    if os.path.exists(json_path):
        with open(json_path) as f:
            label_json = json.load(f)
            x, y, z = label_json['markups'][0]['center']
            width, height, length = label_json['markups'][0]['size']
            center, dimension = [x,y,z], [width, height, length]
            if orientation:
                orientation = np.array(label_json['markups'][0]['orientation'])
                return center, dimension, orientation
            return center, dimension
    else:
        print(f"The file '{json_path}' does not exist.")
        
        
        
def get_idxs_segment(mask):
    """Get index of the slices that contain segmentation"""
    segment_idx = []
    for i in range(mask.shape[0]):
        if sum(mask[i].ravel() > 0): #if there is a segmentation mask in slice i
            segment_idx.append(i)
    if not segment_idx:
        raise ValueError("There is no segmentation mask")
    else:
        return segment_idx   
        

   
def fix_direction(image_path):
    """This function will flip the image axis to right directioin [1,1,1] """
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




def get_yololabel_from_3Dmarkup(json_path, ct_path):
    ct_im, spacing, origin, direction = read_nifti_image(ct_path, only_img=False)
    flip_axes = [i for i, val in enumerate(np.diag(direction)) if val < 0]
    center, dimension = read_json(json_path)
    x = (center[0] - origin[0]) / spacing[0]
    y = (origin[1] - center[1]) / spacing[1]
    z = (center[2] - origin[2]) / spacing[2]
    im_h, im_w = ct_im.shape[1:]
    if 0 in flip_axes:
        x = im_w/2 - x + im_w/2
    if 1 in flip_axes:
        y = im_h/2 - y + im_h/2
    w = dimension[0] / spacing[0]
    h = dimension[1] / spacing[1]
    l = dimension[2] / spacing[2]
    idxs = get_slice_idxs_with_bbox(z, l, spacing[2])
    label = normalize_bbox([x,y,w,h], [im_w, im_h])
    return label, idxs


def get_slice_idxs_with_bbox(z, l, spacing_z):
    """ returns the index of slices with bounding box"""
    l = l / spacing_z
    idxs = [int(idx) for idx in range(int(z - l), int(z + l) + 1)]
    return idxs
    
     
         
        
def image_to_numpy(im_path):
    """ Reads 2d image file and returns 2d numpy array"""
    try:
        image = Image.open(im_path)
        image_array = np.array(image)
        return image_array
    except Exception as e:
        print("Error reading the image:", e)
        return None


def save_array_as_png(arr, save_path):
    """ Converts 2d numpy array to png image and saves to the given path """
    try: 
        image = Image.fromarray(arr.astype(np.uint8))
        image.save(save_path)
        print("Image saved as:", save_path)
    except Exception as e:
        print("Error saving the image:", e)
        
        
def read_label_txt(label_path):
    try:
        with open(label_path, 'r') as f:
            labels_data = f.readlines()
        
        labels = []  # Initialize an empty list to store multiple labels
        
        for line in labels_data:
            label = line.strip().split()
            if len(label) == 5:
                class_id, x_center, y_center, w, h = map(float, label)
                labels.append([class_id, x_center, y_center, w, h])
            else:
                raise ValueError("Invalid label format in the file.")
        if labels:
            return labels
        else:
            raise ValueError("Label file does not contain valid labels.")
    except FileNotFoundError:
        print("Label file not found:", label_path)
    except ValueError as ve:
        print("Error reading label:", ve)
        
        
def normalize_bbox(label, im_shape):
    """ Normalizes coordinates of a bounding box. Divide x-coordinates by image width and y-coordinates by image height.
    Args:
        label: List of enormalized bounding box `[x_center, y_center, width, height]`.
        im_size: [image width, image height]
    """
    im_h, im_w = im_shape
    center_x, center_y, w, h = label
    x_norm = center_x / im_w
    y_norm = center_y / im_h
    w_norm = w / im_w
    h_norm = h / im_h
    normalized_label = [x_norm, y_norm, w_norm, h_norm]
    return normalized_label

def denormalize_bbox(normalized_label, im_shape):
    [x, y, w, h] = normalized_label
    im_w, im_h = im_shape
    new_w = w * im_w
    new_h = h * im_h
    new_x = x * im_w
    new_y = y * im_h
    return [new_x, new_y, new_w, new_h]

    
    
def random_item(items):
    return items[np.random.randint(len(items))]


def load_model(model_path):
    with open(model_path, 'rb') as f:
        loaded_model = torch.load(f)
    return loaded_model

    
    



if __name__ == '__main__':
    
    
    root_data_folder = r'E:\Aibota\annotated_data_bii'
    save_extracted_aorta_masks(root_data_folder)
    
    #import shutil
    #source = r'E:\Aibota\aorta_seg_inference'
    #dest = r'E:\Aibota\data_part2'
    #for folder in os.listdir(source):
    #    file_path = os.path.join(source, folder, "og_ct", "og_ct_seg.nii.gz")
    #    shutil.copy(file_path, os.path.join(dest, folder))
    #    print(f"File copied from {file_path} to {os.path.join(dest, folder)}")