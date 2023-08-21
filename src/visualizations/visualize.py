import matplotlib.pyplot as plt
import numpy as np

def plot_scan(ct_scan, n, window=500, level=100):
  #visualizing the first n*n images of the given patient data
  fig, ax = plt.subplots(n, n, figsize=(20, 20))
  for i in range(n*n):
    row = i // n
    col = i % n
    vmin = level - window/2
    vmax = level + window/2
    ax[row,col].imshow(ct_scan[i], cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
    ax[row,col].axis('off')
  plt.show()

def plot_mid_slice(ct_scan, n=32):
  # plot n-th slice of a scan
  plt.imshow(np.rot90(ct_scan[n], k=2), cmap='gray', origin='lower')
  plt.colorbar(label='Signal intensity')
  plt.show() 

