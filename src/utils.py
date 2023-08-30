
import os
import SimpleITK as sitk
import numpy as np


def read_image(ct_path, only_img=False):
    """Reads NIFTI image and returns it as a 3D numpy array"""
    if os.path.exists(ct_path):
        itkimage = sitk.ReadImage(ct_path)
        ct_scan = sitk.GetArrayFromImage(itkimage)
        if only_img:
            return ct_scan
        spacing = np.array(list(reversed(itkimage.GetSpacing())))
        return ct_scan, spacing
    else:
        print("The file path doesn't exist")



if __name__ == "__main__":
    test_path = r'C:\Users\sanatbyeka\Desktop\calcium_scoring\data\raw\aorta\PD002\og_ct.nii'
    test_load = read_image(test_path, only_img=True)
    print(test_load.shape)
