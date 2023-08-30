
import os
import SimpleITK as sitk
from PIL import Image
import numpy as np


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
        

if __name__ == "__main__":
    test_path = r'C:\Users\sanatbyeka\Desktop\calcium_scoring\data\processed\bifurcation_point\images\1_50.png'
    test_load = image_to_numpy(test_path)
    print(test_load.shape)
