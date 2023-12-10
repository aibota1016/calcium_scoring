import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import numpy as np
import os
import sys
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_dir)
import utils


def plot_scan_slices(ct_scan, n=5, window=500, level=100):
    """ plots the first n*n slices of the given CT image """
    fig, ax = plt.subplots(n, n, figsize=(10, 10))
    assert n*n < ct_scan.shape[0], "n out of range. Please enter smaller number"
    for i in range(n*n):
        row = i // n
        col = i % n
        vmin = level - window/2
        vmax = level + window/2
        ax[row,col].imshow(ct_scan[i, :, :], cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
        ax[row,col].axis('off')
    fig.suptitle(f"First {n*n} slices of a given CT scan")
    plt.show()


def plot_single_slice(slice):
  """ plots mid slice of a CT scan """
  plt.imshow(slice, cmap='gray')
  plt.colorbar(label='Signal intensity')
  plt.show() 
  

  

def plot_augment_results(im, label, augmented_im, augmented_label, title="", save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle(title, fontsize=20)
    axes[0].imshow(im, cmap=plt.cm.gray)
    axes[0].set_title('Before', fontsize=20)
    [x, y, w, h] = utils.denormalize_bbox(label, im.shape)
    box1 = plt.Rectangle((x-w/2, y-h/2), w, h, fill=False, edgecolor='red', linewidth=1.5)
    axes[0].add_patch(box1)
    axes[1].imshow(augmented_im, cmap=plt.cm.gray)
    axes[1].set_title('After', fontsize=20)
    [x2, y2, w2, h2] = utils.denormalize_bbox(augmented_label, augmented_im.shape)
    box2 = plt.Rectangle((x2-w2/2, y2-h2/2), w2, h2, fill=False, edgecolor='red', linewidth=1.5)
    axes[1].add_patch(box2)
    if save_path:
      save_dir = "src/visualizations/augmentations"
      if not os.path.exists(save_dir):
        os.makedirs(save_dir)
      plt.savefig(os.path.join(save_dir, save_path), bbox_inches='tight') 
    plt.tight_layout()
    plt.show()
    
    
def plot_masks(ct_scan, mask, row=3, col=6, save_path=None):
    """Function to plot overlaying masks on top of the slices of the given CT scan image"""
    fig, axes = plt.subplots(row, col, figsize=(col*3, row*3))
    fig.suptitle("CT scan slices with aorta segmentation mask", fontsize=16)
    idxs = utils.get_idxs_segment(mask)
    for i in range(len(idxs)):
        if i < col * row:
            ax = axes.flat[i]
            ax.imshow(ct_scan[idxs[i]], cmap='gray')
            seg = np.ma.masked_where(mask[idxs[i]] == False, mask[idxs[i]])
            reduced_transparency_cmap = ListedColormap([(r, g, b, 0.3) for r, g, b in plt.get_cmap('Set1').colors])
            ax.imshow(seg, cmap=reduced_transparency_cmap)
            ax.set_title("Slice " + str(idxs[i]))
            ax.axis('off')
    if save_path:
        save_dir = "src/visualizations"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, save_path), bbox_inches='tight')         
    plt.show()
    

  
def plot_bboxes_single_image(im_path, label_path, title=""):
  """
  Plots a single 2d array image with its bounding box labels
  Args:
    label is a list containing x_center, y_center, width, heiht in yolo format
  """
  im_array = utils.image_to_numpy(im_path)
  label = utils.read_label_txt(label_path)
  print(label)
  plt.imshow(im_array, cmap=plt.cm.gray)
  if title != "":
    plt.title(title)
  [x, y, w, h] = utils.denormalize_bbox(label[1:], im_array.shape)
  x_min = x - w / 2
  y_min = y - h / 2
  box = plt.Rectangle((x_min, y_min), w, h, fill=False, edgecolor='red', linewidth=1)
  plt.gca().add_patch(box)
  plt.colorbar(label='Signal intensity')
  plt.show()


def plot_imgs_bboxes(images_folder, labels_folder, title="", columns=5, rows=3, save_path=None):
  # filter images with labels for plotting
  image_files, label_files = [], []
  n = columns*rows
  for label_file in os.listdir(labels_folder):
    if len(label_files) >= n*n:
        break 
    label_files.append(os.path.join(labels_folder, label_file))
    for image_file in os.listdir(images_folder):
      if image_file.split('.')[0] == label_file.split('.')[0]:
        image_files.append(os.path.join(images_folder, image_file))
  image_files, label_files = sorted(image_files), sorted(label_files)
  #plot
  fig, axes = plt.subplots(rows, columns, figsize=(columns*3, rows*3))
  fig.suptitle(title, fontsize=16)
  j = 0
  for i, (image_file, label_file) in enumerate(zip(image_files, label_files)):
    if j < columns * rows:
      label = utils.read_label_txt(label_file)
      if len(label) > 1:
        ax = axes.flat[j]
        # Read and display the image
        img = utils.image_to_numpy(image_file)
        ax.imshow(img, cmap='gray')
        # Draw bounding box
        [x, y, w, h] = utils.denormalize_bbox(label[1:], img.shape)
        x_min = x - w / 2
        y_min = y - h / 2
        box = plt.Rectangle((x_min, y_min), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(box)
        slice_idx = (image_file.split('/')[-1]).split('.')[0]
        ax.set_title(slice_idx)
        ax.axis('off')
        j = j + 1
  # save plot as file
  if save_path:
    save_dir = "src/visualizations"
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, save_path), bbox_inches='tight') 
  plt.show()


def display_ct_planes(ct_data):
    # Display CT images in axial, coronal, and sagittal planes side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Axial plane
    axes[0].imshow(ct_data[ct_data.shape[0] // 2, :, :], cmap='gray', origin='lower')
    axes[0].set_title("Axial Plane")
    axes[0].axis('off')
    # Coronal plane
    axes[1].imshow(ct_data[:, ct_data.shape[1] // 2, :], cmap='gray', origin='lower')
    axes[1].set_title("Coronal Plane")
    axes[1].axis('off')
    # Sagittal plane
    axes[2].imshow(ct_data[:, :, ct_data.shape[2] // 2].T, cmap='gray', origin='lower')
    axes[2].set_title("Sagittal Plane")
    axes[2].axis('off')
    plt.show()


if __name__ == '__main__':
  im_path = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/datasets/train_val/split_3/train/images/PD001_31_aug0.png'
  label = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/datasets/train_val/split_3/train/labels/PD001_31_aug0_0.txt'
  #plot_bboxes_single_image(im_path, label)
  #plot_imgs_bboxes(r'E:\Aibota\calcium_scoring\data\processed\images', r'E:\Aibota\calcium_scoring\data\processed\labels', columns=7, rows=4, save_path=r'E:\Aibota\calcium_scoring\src\visualizations\aorta_bbox.png')

  images_folder = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/datasets/train_val/split_2/train/images'
  labels_folder = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/datasets/train_val/split_2/train/labels'
  plot_imgs_bboxes(images_folder, labels_folder, title="", columns=8, rows=4, save_path=None)