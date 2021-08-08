import os

import matplotlib.pyplot as plt
import numpy as np

from radiomics_pg.utilities.geometry import bounding_box, centroid
from radiomics_pg.utilities.dicom import read as read_dicom
from radiomics_pg.utilities.nii import read as read_nii

out_folders = ['tests/input/signal', 'tests/input/mask']
for folder in out_folders:
    if not os.path.isdir(folder):
        os.makedirs(folder)
    
#Read the mask
filename = '../../../../Pac_F_Bianco/ProgettiRicerca/Radiomics/Datasets/Sassari/Studies/SPN-01/115GA/nifti-scans-and-rois/CT/mask.nii'
mask = read_nii(filename)
_, mask_slice, slice_idxs = bounding_box(mask)
            
for s in range(mask_slice.shape[2]):
    plt.matshow(mask_slice[:,:,s], cmap="gray")
    plt.axis('off')
    plt.savefig(f'{out_folders[1]}/{s:03d}.jpg', dpi=300, 
    bbox_inches='tight', pad_inches = 0)
    
#Read the signal
folder_name = '../../../../Pac_F_Bianco/ProgettiRicerca/Radiomics/Datasets/Sassari/Studies/SPN-01/115GA/dicom-scans/CT'
signal, x, y, z, x_length, y_length, z_length = read_dicom(folder_name)
signal_slice = signal[slice_idxs]
x_slice = x[slice_idxs]
y_slice = y[slice_idxs]
z_slice = z[slice_idxs]
                
for s in range(signal_slice.shape[2]):
    plt.matshow(signal_slice[:,:,s], cmap="gray")
    plt.axis('off')
    plt.savefig(f'{out_folders[0]}/{s:03d}.jpg', dpi=300, 
    bbox_inches='tight', pad_inches = 0)
    
#Clar the figure
plt.clf()
    
#Plot the points
num_slots = np.ceil(np.sqrt(signal_slice.shape[2])).astype(np.int)
fig, axs = plt.subplots(num_slots, num_slots)

point_id = 0
for k in range(signal_slice.shape[2]):
    row, col = np.unravel_index(indices = k, 
                                shape = (num_slots, num_slots), 
                                order='C')    
    
    #Plot the signal
    axs[row, col].scatter(x_slice[:,:,k].flatten(), 
                          y_slice[:,:,k].flatten(),
                          c = signal_slice[:,:,k].flatten(),
                          vmin = np.min(signal_slice[:,:,k]),
                          vmax = np.max(signal_slice[:,:,k]),
                          cmap = plt.get_cmap('coolwarm'))
    
    #Compute and plot the centroid
    x_centr = centroid(x_slice[:,:,k], signal_slice[:,:,k])
    y_centr = centroid(y_slice[:,:,k], signal_slice[:,:,k])
    axs[row, col].plot(x_centr, y_centr, 'P', color='white',)
    
    for i in range(signal_slice.shape[0]):
        for j in range(signal_slice.shape[1]):
            #axs[row, col].annotate(point_id, (x_slice[i,j,k], y_slice[i,j,k]))
            #axs[row, col].plot(x_slice[i,j,k], y_slice[i,j,k])
            point_id += 1
plt.show()
