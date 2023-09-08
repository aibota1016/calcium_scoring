
import os
import json
import SimpleITK as sitk
from PIL import Image
import numpy as np
import nrrd
import nibabel as nib


def read_nifti_image(ct_path, spacing=False):
    """Reads NIFTI image and returns it as a 3D numpy array"""
    if os.path.exists(ct_path):
        itkimage = sitk.ReadImage(ct_path)
        ct_scan = sitk.GetArrayFromImage(itkimage)
        if spacing:
            spacing = np.array(list(reversed(itkimage.GetSpacing())))
            return ct_scan, spacing
        #origin = np.array(list(reversed(itkimage.GetOrigin())))
        #direction = np.array(list(reversed(itkimage.GetDirection())))
        return ct_scan
    else:
        print("The file path doesn't exist")
        
        
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
        if 'aorta_mask.nii' not in os.listdir(PD) and 'segmentation.seg.nrrd' in os.listdir(PD):
            extract_aorta_mask(os.path.join(PD,'segmentation.seg.nrrd'), os.path.join(PD,'og_ct.nii'), os.path.join(PD,'aorta_mask.nii'))
   

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
        
        
def image_to_numpy(im_path):
    """ Reads 2d image and returns 2d numpy array"""
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
        if labels_data:
            label = labels_data[0].strip().split()
            if len(label) == 5:
                class_id, x_center, y_center, w, h = map(float, label)
                return [x_center, y_center, w, h]
            else:
                raise ValueError("Invalid label format in the file.")
        else:
            raise ValueError("Label file is empty.")
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
    
    
def random_item(items):
    return items[np.random.randint(len(items))]
    
    



if __name__ == '__main__':
    
    
    root_data_folder = r'C:\Users\sanatbyeka\Desktop\calcium_scoring\data\raw\annotated_data_bii'
    save_extracted_aorta_masks(root_data_folder)