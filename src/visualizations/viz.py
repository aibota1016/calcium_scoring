import matplotlib.pyplot as plt
import numpy as np

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


def plot_mid_slice(ct_scan):
  """ plots mid slice of a CT scan """
  mid = int(ct_scan.shape[0] / 2)
  plt.imshow(ct_scan[mid, :, :], cmap='gray')
  plt.colorbar(label='Signal intensity')
  plt.title(f'Slice {mid}')
  plt.show() 
  
  
def plot_image_bbox(im_array, label, title=""):
  """
  Plots a single 2d array image with its bounding box labels
  Args:
    label is a list containing x_center, y_center, width, heiht in yolo format
  """
  [x, y, w, h] = label
  im_w, im_h = im_array.shape
  w = w * im_w
  h = h * im_h
  x_min = x * im_w - w / 2
  y_min = y * im_h- h / 2

  plt.imshow(slice, cmap=plt.cm.gray)
  if title != "":
    plt.title(title)
  box = plt.Rectangle((x_min, y_min), w, h, fill=False, edgecolor='red', linewidth=2)
  plt.gca().add_patch(box)
  plt.colorbar(label='Signal intensity')
  plt.show()
  




