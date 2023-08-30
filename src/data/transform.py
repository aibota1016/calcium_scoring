""" This file is used to store different functions for data augmentation """

import numpy as np
import sys
import os
import cv2

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import utils
from visualizations import viz



def clip_values_3d(ct_scan, min_bound=-1000, max_bound=1000):
    """
    Clips outliers, if not specified min_bound=-1000 max_bound=1000
    ct_scan is a 3D numpy array
    """
    img = ct_scan
    img = np.clip(img, min_bound, max_bound).astype(np.float32)
    return img


def clip_values_2d(im, min_bound=-1000, max_bound=1000):
    clipped_im = im
    clipped_im = np.clip(clipped_im, min_bound, max_bound).astype(np.float32)
    return clipped_im


def flip(im, label, flip_type='horizontal'):
    """
    Flip the given image horizontally and adjust its bounding box labels.
    Args:
        image_array (numpy.ndarray): 2D NumPy array representing the image.
        bbox_labels (list): List of bounding box labels [center_x, center_y, width, height].
        flip_type (str): Type of flip: 'horizontal' or 'vertical'.
    """
    if flip_type == 'horizontal':
        flipped_image = np.flip(im, axis=1)
        adjust_index = 0
    elif flip_type == 'vertical':
        flipped_image = np.flip(im, axis=0)
        adjust_index = 1 
    else:
        raise ValueError("Invalid flip_type. Use 'horizontal' or 'vertical'.")

    # Adjust the bounding box labels
    flipped_coord = 1.0 - label[adjust_index] 
    flipped_bbox = label.copy()
    flipped_bbox[adjust_index] = flipped_coord

    return flipped_image, flipped_bbox

def rotate(im, label, factor=0):
    """ 
    Rotates the image and its label by 90degrees * factor
        factor: Number of CCW rotations. Must be in set {0, 1, 2, 3} See np.rot90.
    """
    if factor not in {0, 1, 2, 3}:
        raise ValueError("Parameter n must be in set {0, 1, 2, 3}")
    x_center, y_center, w, h = label
    if factor == 1: # 90 degrees rotation
        label = y_center, 1 - x_center, h, w
    elif factor == 2: # 180 degrees rotation
        label = 1-x_center, 1-y_center, w, h
    elif factor == 3: # 270 degrees rotation
        label = 1 - y_center, x_center, h, w
    return np.ascontiguousarray(np.rot90(im, factor)), label


# def transpose
# def histogram_matching(im)
# def normalize(img, mean, std, max_pixel_value=255.0):
# def _shift_hsv_uint8(img, hue_shift, sat_shift, val_shift):
# def equalize_hist(img):


def random_crop(im, label):
    pass


def shift(im, label, shift_type='horizontal', shiftamount=15):
    """
    Shift the given image either horizontally or vertically by a given shift pixelamount and adjust the bounding box label.
    Args:
        shift_amount (int): number of pixels to shift.
        shift_direction (str): 'horizontal' for horizontal shift, 'vertical' for vertical shift.
    """
    x_center, y_center, w, h = label
    if shift_type == 'horizontal': 
        axis = 1
        shifted_center_x = x_center + shiftamount / im.shape[1]
        shifted_center_y = y_center
    elif shift_type == 'vertical':
        axis = 0
        shifted_center_y = y_center + shiftamount / im.shape[0]
        shifted_center_x = x_center
    shifted_image = np.roll(im, shiftamount, axis=axis)
    shifted_label = [shifted_center_x, shifted_center_y, w, h]
    return shifted_image, shifted_label
    

def resize(im, label, new_height=416, new_width=416):
    """Resizes the image and its label to a given size"""
    x_center, y_center, w, h = label
    scale_w = new_width / im.shape[1]
    scale_h = new_height / im.shape[0]
    new_x = x_center * scale_w
    new_y = y_center * scale_h
    new_w = w * scale_w
    new_h = h * scale_h
    resized_label = [new_x, new_y, new_w, new_h]
    resized_im = cv2.resize(im, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_im, resized_label



if __name__ == '__main__':
    test_im_path = r'C:\Users\sanatbyeka\Desktop\calcium_scoring\data\processed\bifurcation_point\images\1_50.png'
    test_label_path = r'C:\Users\sanatbyeka\Desktop\calcium_scoring\data\processed\bifurcation_point\labels\1_50.txt'
    test_im = utils.read_nifti_image(test_im_path)
    test_label = utils.read_label_txt(test_label_path)
    print(test_im.shape)
    print(test_label)
    viz.plot_image_bbox(test_im, test_label, title="Before resize")
    # flipped_image, flipped_bbox_label = flip(test_im, test_label, flip_type='vertical')
    rot_im, rot_box = resize(test_im, test_label)
    viz.plot_image_bbox( rot_im, rot_box, title="After resize")
    #with open(test_label_path, 'r') as f:
    #    test_label = f.read(test_label_path)
    # function to test    
    #test_out = horizontal_flip(test_im, test_label)
    # visualize
    #visualize.plot_yolo(test_out)