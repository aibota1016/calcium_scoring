""" This file is used to store different functions for data augmentation """

import numpy as np
import sys
import os
import cv2
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_dir)
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


def random_flip(im, label):
    """
    Flip the given image horizontally and adjust its bounding box labels.
    Args:
        image_array (numpy.ndarray): 2D NumPy array representing the image.
        bbox_labels (list): List of bounding box labels [center_x, center_y, width, height].
        flip_type (str): Type of flip: 'horizontal' or 'vertical'.
    """
    flip_type = utils.random_item(['horizontal', 'vertical'])
    if flip_type == 'horizontal':
        flipped_image = np.flip(im, axis=1)
        adjust_index = 0
    else:
        flipped_image = np.flip(im, axis=0)
        adjust_index = 1 
    # Adjust the bounding box labels
    flipped_coord = 1.0 - label[adjust_index] 
    flipped_bbox = label.copy()
    flipped_bbox[adjust_index] = flipped_coord
    return flipped_image, flipped_bbox

def random_rotate(im, label):
    """ 
    Rotates the image and its label by 90degrees * factor
        factor: Number of CCW rotations. Must be in set {0, 1, 2, 3}.
    """
    factors = [0, 1, 2, 3]
    random_k = utils.random_item(factors)
    x_center, y_center, w, h = label
    if random_k == 1: # 90 degrees rotation
        label = y_center, 1 - x_center, h, w
    elif random_k == 2: # 180 degrees rotation
        label = 1-x_center, 1-y_center, w, h
    elif random_k == 3: # 270 degrees rotation
        label = 1 - y_center, x_center, h, w
    return np.ascontiguousarray(np.rot90(im, k=random_k)), label


def random_crop_around_bbox(im, label):
    im_h, im_w = im.shape
    max_size_crop =  min(im_h, im_w)
    min_size = int(max_size_crop * 0.6) 
    crop_size = utils.random_item([s for s in range(min_size, max_size_crop)])
    center_x, center_y, w, h = label
    x = int(center_x * im_w - crop_size / 2)
    y = int(center_y * im_h - crop_size / 2)
    cropped_image = cv2.getRectSubPix(im, (crop_size, crop_size), (x + crop_size / 2, y + crop_size / 2))
    cropped_label = [0.5, 0.5, w*(im_w/crop_size), h*(im_h/crop_size)]
    return cropped_image, cropped_label


def random_shift(im, label):
    """
    Shift the given image either horizontally or vertically by a given shift pixelamount and adjust the bounding box label.
    Args:
        shift_amount (int): number of pixels to shift.
        shift_direction (str): 'horizontal' for horizontal shift, 'vertical' for vertical shift.
    """
    x_center, y_center, w, h = label
    im_h, im_w = im.shape
    shift_type = utils.random_item(['horizontal', 'vertical'])
    shift_range = int(min(im_w, im_h)*0.3)
    shift_amount = utils.random_item([x for x in range(shift_range)])
    if shift_type == 'horizontal': 
        shift_percentage = shift_amount / im_w
        axis = 1
        shifted_center_x = x_center + shift_percentage
        shifted_center_y = y_center
    elif shift_type == 'vertical':
        shift_percentage = shift_amount / im_h
        axis = 0
        shifted_center_y = y_center + shift_percentage
        shifted_center_x = x_center
    error_message = "Requested shift amount is more than 30% of the image size"
    assert shift_percentage  <= 0.3, error_message    
    shifted_image = np.roll(im, shift_amount, axis=axis)
    shifted_label = [shifted_center_x, shifted_center_y, w, h]
    return shifted_image, shifted_label
    

def resize(im, new_size):
    """Resizes the image and its label to a given size"""
    resized_im = cv2.resize(im, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
    return np.array(resized_im)

def apply_random_augmentation(im, label):
    augment_functions = [random_flip, random_rotate, random_crop_around_bbox, random_shift]
    random_func = utils.random_item(augment_functions)
    return random_func(im, label)

if __name__ == '__main__':
    test_im_path = r'C:\Users\sanatbyeka\Desktop\calcium_scoring\data\processed\bifurcation_point\images\1_52.png'
    test_label_path = r'C:\Users\sanatbyeka\Desktop\calcium_scoring\data\processed\bifurcation_point\labels\1_52.txt'
    test_im = utils.image_to_numpy(test_im_path)
    test_label = utils.read_label_txt(test_label_path)

    new_im, new_label = apply_random_augmentation(test_im, test_label)
    viz.plot_augment_results(test_im, test_label, new_im, new_label)