import os

from utilities.geometry import bounding_box, centroid, inertia_tensor
from utilities.dicom import read as read_dicom
from utilities.nii import read as read_nii

patient_id = '115GA'
    
#Read the mask
filename = f'../../../../Pac_F_Bianco/ProgettiRicerca/Radiomics/Datasets/Sassari/Studies/SPN-01/{patient_id}/nifti-scans-and-rois/CT/mask.nii'
mask = read_nii(filename)
_, mask_slice, slice_idxs = bounding_box(mask)

#Read the signal
folder_name = f'../../../../Pac_F_Bianco/ProgettiRicerca/Radiomics/Datasets/Sassari/Studies/SPN-01/{patient_id}/dicom-scans/CT'
signal, x, y, z = read_dicom(folder_name)
signal_slice = signal[slice_idxs]
x = x[slice_idxs]
y = y[slice_idxs]
z = z[slice_idxs]

#Compute the inertia tensor
it, princomps = inertia_tensor(x, y, z, signal_slice, normalise_mass = True)