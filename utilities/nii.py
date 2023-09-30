"""Operations with nii (NIfTY) files"""
import nibabel as nib
import numpy as np

def save(data, filename):
    """Save ndarray into a NIfTY file
    
    Parameters
    ----------
    data: ndarray
        The data to save
    filename : str
        The path to the output file.   
    """
    affine = np.array(object = [[-1,0,0,0],
                                [0,-1,0,0],
                                [0,0,1,0],
                                [0,0,0,1]])    
    
    data = np.transpose(data, axes = [1,0,2])
    img = nib.Nifti1Image(dataobj=data, affine=affine)
    img.to_filename(filename=filename)

def read(filename):
    """Read a NIfTY file into a ndarray
    
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