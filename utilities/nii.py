"""Operations with nii (NIfTY) files"""
import nibabel as nib
import numpy as np

def read(filename):
    """Read a NIfTY file
    
    Parameters
    ----------
    filename : str
        The path to the input file.
        
    Returns
    -------
    data : nparray
        The data contained in the input file
    """
    img = nib.load(filename)
    memmap = img.get_fdata()
    
    #Generate an array from the memory map
    data = np.array(memmap)
    
    #Read the axial location of each slice
    slice_locations = list()
    for s in range(data.shape[2]):
        ras_coords = _get_ras_coordinates(voxel_coords = [0,0,s], img = img)
        slice_locations.append(ras_coords[2])
        
    #Sort the data matrix by slice location
    index_array = np.argsort(np.array(slice_locations))
    data = data[:,:,index_array]    
            
    #Take the transpose to make the orientation coherent with DICOM reference
    #system
    data = np.transpose(data, axes = [1,0,2])
    
    return data

def _get_ras_coordinates(voxel_coords, img):
    """Coordinates in the RAS (Right-Anetrior-Superior) reference system
    
    Parameters
    ----------
    voxel_coords : int (3)
        The coordinates (indices) of a voxel in the voxel reference system
    img : numpy memmap
        The nii image as returned by nibabel.load
    
    Returns
    -------
    ras_coords : float (3)
        The coordinates in the RAS reference system
        
    Reference
    ---------
    [1] https://nipy.org/nibabel/coordinate_systems.html
    """
    
    M = img.affine[:3, :3] 
    abc = img.affine[:3, 3]   
    res_coords = M.dot(voxel_coords) + abc
    return res_coords