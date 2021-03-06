"""Utility functions for extracting data from DICOM files"""
import os

import matplotlib.pyplot as plt
import numpy as np

import dateutil.parser
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

_dicom_lut = {
    'Age' : (0x10, 0x1010),
    'Gender' : (0x10, 0x40),
    'ImagePositionPatient' : (0x20, 0x32),
    'InstanceNumber' : (0x20, 0x13),
    'PixelSpacing' : (0x28, 0x30), 
    'PatientName' : (0x10, 0x10),
    'ScanOptions' : (0x18, 0x22),
    'SliceLocation' : (0x20, 0x1041),
    'SliceThickness' : (0x18, 0x50),
    'StudyDate' : (0x08, 0x20),
    'TubeVoltage' : (0x18, 0x60)
}

def read_metadata(folder_name, names = set(['Age', 'Gender']), 
                  accepted_extensions = ['.dcm']):
    """Read metadata from a DICOM folder. 
    
    Parameters
    ----------
    folder_name : str
        The folder where the DICOM files are stored.
    names : set of str
        The names of the metadata to retrieve. See _dicom_lut for possible
        values.
    accepted_extensions : list of str
        Discard the files with extension not in the list.
        
    Returns
    -------
    metadata : dict
        A dictionary containing the requested metadata. The keys of the
        dictionary are the elmenets of names
    """
    metadata = dict()
    
    #Walk through the files in the folder
    dicoms = list()
    for root, dirs, files in os.walk(folder_name):
        for file in files: 
            _, ext = os.path.splitext(file)
            if ext in accepted_extensions:            
                dicoms.append(file) 
    dicoms.sort()
    
    #Read the metadata from the first file
    ds = pydicom.read_file(f'{folder_name}/{dicoms[0]}', force=True)  
    for name in names:
        value = _get_value_of_generic_attribute(ds, name)
        metadata.update({name : value})
            
    return metadata
 
def read(folder_name, accepted_extensions = ['.dcm']):
    """Read data from a DICOM folder
    
    Parameters
    ----------
    folder_name : str
        The folder where the DICOM files are stored.
    accepted_extensions : list of str
        Discard the files with extension not in the list.
    
    Returns
    -------
    data : nparray
        The voxel values (signal)
    x : nparray 
        The coronal coordinate of the centroid of each voxel. The x-axis is 
        increasing to the left hand side of the patient.
    y : nparray
        The sagittal coordinate of the centroid of each voxel. The y-axis is
        increasing to the posterior side of the patient.
    z : nparray
        The axial coordinate of the centroid of each voxel. The z-axis is
        increasing toward the head of the patient.
    x_length : The length of the side of each voxel along the coronal coordinate.
    y_length : The length of the side of each voxel along the sagittal coordinate.
    z_length : The length of the side of each voxel along the axial coordinate
               (slice thickness).
    
    Notes
    -----
    The returned data, x, y, z, x_length, y_length and z_length have all the
    same size. It is assumed that the original dicom data are in the default 
    human standard anatomical position.
    The data array is organised as follows:
        0th axis -> y
        1st axis -> x
        3rd axis -> z
    """
    
    unsorted_list = []
    
    #Walk through the files in the folder
    for root, dirs, files in os.walk(folder_name):
        for file in files: 
            _, ext = os.path.splitext(file)
            if ext in accepted_extensions:
                unsorted_list.append(os.path.join(root, file)) 
    
    #***********************************************        
    #******* Create the empty signal matrix ********
    #***********************************************
    
    #Get the size
    ds = pydicom.read_file(unsorted_list[0], force=True)
    width, height = ds.pixel_array.shape
    depth = len(unsorted_list)
    
    #Get the type
    converted_data = apply_modality_lut(ds.pixel_array, ds)
    dtype = converted_data.flatten()[0].dtype
    
    #Create the empty matrix
    data = np.zeros((height,width,depth), dtype = dtype)
    #***********************************************
    #***********************************************
    #***********************************************
    
    #Create an empty box for the return values besides data
    retval = np.zeros((height,width,depth,6), dtype = np.float)
    
    #Indices matrix
    idxs = np.meshgrid(range(height), range(width), indexing = 'ij')
                   
    #Fill the data matrix and read the axial locations of each slice
    slice_locations = list()
    for i, dicom_loc in enumerate(unsorted_list):
        
        #Get the slice-specific data
        ds = pydicom.read_file(dicom_loc, force=True)  
        row_spacing, column_spacing =\
            _get_value_of_generic_attribute(ds, 'PixelSpacing')
        slice_thickness = _get_value_of_generic_attribute(ds, 'SliceThickness')
        
        #Get the image offset (x, y, and z coordinates of the upper left hand 
        #corner of the image)
        offset = get_attribute_value(dicom_loc, 'ImagePositionPatient')        
        slice_location = offset[2]
        slice_locations.append(slice_location)
        
        #Get the raw data (generally uint16)
        raw_data = ds.pixel_array 
        
        #Convert them to physical units
        data[:,:,i] = apply_modality_lut(raw_data, ds)
                  
        retval[idxs[0],idxs[1],i,0] = column_spacing * (idxs[1] + 0.5) + offset[0]
        retval[idxs[0],idxs[1],i,1] = row_spacing * (idxs[0] + 0.5) + offset[1]
        retval[:,:,i,2] = slice_location
        retval[idxs[0],idxs[1],i,3] = column_spacing
        retval[idxs[0],idxs[1],i,4] = row_spacing 
        retval[:,:,i,5] = slice_thickness
          
    #Sort the matrices by slice location
    index_array = np.argsort(np.array(slice_locations))
    data = data[:,:,index_array]
    retval = retval[:,:,index_array,:]
        
    return data, *[retval[:,:,:,i] for i in range(retval.shape[3])]

def get_attribute_value(source, attribute):
    """Retrieve the attribute value from a DICOM file
    
    Parameters
    ----------
    source : str
        Path to the DICOM source 
    attribute : str
        String indicating the attribute of which the value is to be retrieved.
        Possible values are:
            **** Patient's data ****
            'Age'       -> Patient's age
            'Gender'    -> Patient's gender
            'Name'      -> Patient's name
            'Surname'   -> Patient's surname
            
            **** Scan data ****
            'InstanceNumber'  -> The number that identifies the image
            'PixelSpacing'    -> The in-plane pixel (voxel) spacing
            'ScanOptions'     -> The scan modality (e.g. helicoidal)
            'SliceLocation'   -> The relative position of the image plane in mm
            'SliceThickness'  -> The slice thickness
            'StudyDate'       -> The scan acquisition date
            
            **** CT specific data ****
            'TubeVoltage'     -> The X-Ray tube voltage
        
    value : 
        The attribute value
    """
    
    retval = None 
    dicom_dict = pydicom.read_file(source)
    
    if attribute == 'Age':
        retval = _get_patient_age(dicom_dict)  
    elif attribute == 'Name':
        retval, _ = _get_name_and_surname(dicom_dict)
    elif attribute == 'Surname':
        _, retval = _get_name_and_surname(dicom_dict)   
    elif attribute == 'ImagePositionPatient':
        retval = _get_image_position_patient(dicom_dict)
    else:
        retval = _get_value_of_generic_attribute(dicom_dict, attribute)
    
    return retval

def _get_value_of_generic_attribute(dicom_dict, attribute):
    """Get the value of a generic attribute
    
    Parameters
    ----------
    dicom_dict : dict
        The DICOM dictionary returned by pydicom.read_file()
    attribute : str
        The attribute of which the value is to be retrieved. For possible 
        values see _dicom_lut
        
    Returns
    -------
    value
        The attribute value
    """
    
    retval = None
    
    if attribute not in _dicom_lut.keys():
        raise Exception('Attribute not supported') 
    
    retval = dicom_dict[_dicom_lut[attribute]].value
    return retval
    
def _get_study_date(dicom_dict):
    """Get the study date
    
    Parameters
    ----------
    dicom_dict : dict
        The DICOM dictionary returned by pydicom.read_file()
        
    Returns
    -------
    date : str
        The study date in [day month year] format
    """
    iso_date = _get_value_of_generic_attribute(dicom_dict, 'StudyDate')
    return iso_date
    
    
def _get_patient_age(dicom_dict):
    """Get patients' age
    
    Parameters
    ----------
    dicom_dict : dict
        The DICOM dictionary returned by pydicom.read_file()
        
    Returns
    -------
    age : int
        The patient's age
    """
    retval = _get_value_of_generic_attribute(dicom_dict, 'Age')
    retval = int(retval.strip('Y'))
    return retval
    
def _get_name_and_surname(dicom_dict):
    """Get patient's name and surname
    
    Parameters
    ----------
    dicom_dict : dict
        The DICOM dictionary returned by pydicom.read_file()
        
    Returns
    -------
    name : str
        The patient's given name
    surname : str
        The patient's surname
    """
    
    value = _get_value_of_generic_attribute(dicom_dict, 'PatientName')
    name_string = value.__repr__()
    name_string = name_string.strip("'")
    retval = name_string.split("^", 2)
    retval.reverse()
    return retval

def _get_image_position_patient(dicom_dict):
    """Returns the x, y, and z coordinates of the upper left hand corner 
    (center of the first voxel transmitted) of the image, in mm
    
    Parameters
    ----------
    dicom_dict : dict
        The DICOM dictionary returned by pydicom.read_file()
        
    Returns
    -------
    image_pos : list of float (3)
        The x, y, and z coordinates of the upper-left corner
    """
    image_pos = _get_value_of_generic_attribute(dicom_dict, 
                                                'ImagePositionPatient')
    return image_pos

def _get_slice_location(dicom_dict):
    """Axial slice location
    
    Parameters
    ----------
    dicom_dict : dict
        The DICOM dictionary returned by pydicom.read_file()
        
    Returns
    -------
    slice_location : float
        The axial location of the slice
    """
    slice_location = _get_value_of_generic_attribute(dicom_dict, 'SliceLocation')
    return slice_location
