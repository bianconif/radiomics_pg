import pickle

import numpy as np

from utilities.nii import read as read_nii
from utilities.dicom import read as read_dicom
from utilities.geometry import bounding_box, zingg_shape, TriangularMesh

class Roi():
    """Read and and save regions (volumes) of interest"""
    
    @staticmethod
    def from_dcm_and_nii(mask_file, scan_folder, **kwargs):
        """Generate ROI from scan (dicom) and mask (nii) data
        
        Parameters
        ----------
        mask_file : str
            Path to the folder containing the mask (.nii file).  
        scan_folder : str
            Path to the folder containing the dicom data. It is assumed each
            dicom file in the folder represents one slice. 
        diagnosis (optional) : str
            A string indicating the diagnosis for the Roi.
        """
        
        #Create an empty Roi and import data from scan and mask 
        roi = Roi()
        roi._import(mask_file, scan_folder)
        
        #Load the diagnosis if present
        if 'diagnosis' in kwargs.keys():
            roi.diagnosis = kwargs['diagnosis']
        else:
            roi.diagnosis = None
        
        return roi
    
    @staticmethod
    def from_pickle(source):
        """Read the roi from a previously generated pickle
        
        Parameters
        ----------
        source : str
            Path to the pickle source.
        """
        
        roi = Roi()
        roi._load(source)
        return roi        
           
    def __init__(self):
        self.empty = True
        self.diagnosis = None
    
    def _import(self, mask_file, scan_folder):
        """Import scan (dicom) and mask (nii) data
        
        Parameters
        ----------
        mask_file : str
            Path to the folder containing the mask (.nii file).  
        scan_folder : str
            Path to the folder containing the dicom data. It is assumed each
            dicom file in the folder represents one slice.        
        """
        
        #Read the mask
        mask = read_nii(mask_file)
        _, mask_slice, slice_idxs = bounding_box(mask)
        self.mask = mask[slice_idxs]   
        
        #Read the signal
        signal, x, y, z = read_dicom(scan_folder)
        self.signal = signal[slice_idxs]
        self.x = x[slice_idxs]
        self.y = y[slice_idxs]
        self.z = z[slice_idxs]
        
        self.empty = False
    
    def get_signal(self):
        """Return the signal volume"""
        return self.signal
    
    def get_mask(self):
        """Return the mask volume"""
        return self.mask    
    
    def get_average_spacing(self):
        """The average inter-voxel spacing in the x, y and z directions
        
        Returns
        -------
        avg_spacing : float (3)
            The average spacing along x, y and z
        """
        x_spacing = np.abs(np.diff(self.x, n = 1, axis = 1))
        y_spacing = np.abs(np.diff(self.y, n = 1, axis = 0))
        z_spacing = np.abs(np.diff(self.z, n = 1, axis = 2))
        avg_spacing = (np.mean(x_spacing.flatten()), 
                       np.mean(y_spacing.flatten()),
                       np.mean(z_spacing.flatten()))
        return avg_spacing
        
    def save(self, destination):
        """Save the ROI into a pickle file
        
        Parameters
        ----------
        destination : str
            The destination file (.pkl)
        """
        
        self._not_empty_check()
        
        with open(destination, "wb") as f:
            pickle.dump((self.mask, 
                         self.signal, 
                         self.x,
                         self.y,
                         self.z,
                         self.diagnosis), f)           
            
    def _load(self, source):
        """Load the ROI from a pickle file
        
        Parameters
        ----------
        source : str
            The source file (.pkl)
        """
        
        with open(source, 'rb') as f:
            self.mask, self.signal, self.x, self.y, self.z, self.diagnosis =\
                pickle.load(f)
             
        self.empty = False
            
    def _not_empty_check(self):
        """Returns an exception if the Roi is empty"""
        if self.empty:
            raise Exception(f'The Roi is empty')
        
    def get_mask_mesh(self):
        """Returns the triangular mesh generated from the mask. Does not
        recompute the mesh if it is cached.
        
        Returns
        -------
        tmesh : TriangularMesh
            The triangular mesh generated from the mask. 
        """
        try:
            self.mask_tmesh
        except AttributeError:
            self.mask_tmesh = TriangularMesh.by_marching_cubes(self)
        
        tmesh = self.mask_tmesh    
        return tmesh
        
    def get_surface_area(self):
        """Returns the surface area
        
        Returns
        -------
        area : float
            The surface area.
        """
        area = self.get_mask_mesh().get_surface_area()
        return area
    
    def get_voxel_volume(self):
        """Returns the total volume as the sum of the volume of each voxel
        
        Returns
        -------
        voxel_volume : float
            The voxel volume.
        """
        voxel_side_length_x = np.zeros(self.signal.shape)
        voxel_side_length_y = np.zeros(self.signal.shape)
        voxel_side_length_z = np.zeros(self.signal.shape)
        
        #Compute the voxel side lengths as the distance between the centroids
        #of adjacent voxels
        voxel_side_length_x[:,1::,:] = np.abs(np.diff(self.x, axis = 1))
        voxel_side_length_y[1::,:,:] = np.abs(np.diff(self.y, axis = 0))
        voxel_side_length_z[:,:,1::] = np.abs(np.diff(self.z, axis = 2))
        
        #Pad the borders
        voxel_side_length_x[:,0,:] = voxel_side_length_x[:,1,:]
        voxel_side_length_y[0,:,:] = voxel_side_length_y[1,:,:]
        voxel_side_length_z[:,:,0] = voxel_side_length_z[:,:,1]
        
        #Compute the volumes of all the voxels
        volumes = voxel_side_length_x * voxel_side_length_y * voxel_side_length_z
        
        #Only retain the volumes within the mask
        valid_volumes = self.get_mask() * volumes
        
        return np.sum(valid_volumes.flatten())
        
    def zingg_shape(self, on_signal = True):
        """Zingg shape parameters and classification
        
        Parameters
        ----------
        on_signal : bool
            A flag indicating whether the shape parameters are computed on the
            signal (True) or on the mask (False).
                    
        Returns
        -------
        r1, r2 : float
            Respectively b/a and c/b, where a, b and c indicate the principal 
            inertia moments of the mass distribution sorted in descending order 
            (a > b > c).
        zingg_class : int (possible values = 0, 1, 2 and 3)
            The Zingg class:
                0 -> rod (prolate)
                1 -> blade (oblate)
                2 -> sphere (equant)
                3 -> disc
             
        References
        ----------
        [1] Domokos, G., Sipos, A., Szabo, T., Varkonyi, P.
            Pebbles, Shapes, and Equilibria
            (2010) Mathematical Geosciences, 42 (1), pp. 29-47. 
        """   
        r1 = r2 = zingg_class = None
        if on_signal:
            r1, r2, zingg_class = zingg_shape(self.x, self.y, self.z, self.signal)
        else:
            r1, r2, zingg_class = zingg_shape(self.x, self.y, self.z, self.mask)
            
        return r1, r2, zingg_class
            
