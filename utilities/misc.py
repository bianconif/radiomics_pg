import pickle

import numpy as np

from radiomics_pg.utilities.nii import read as read_nii
from radiomics_pg.utilities.dicom import read as read_dicom
from radiomics_pg.utilities.geometry import bounding_box, zingg_shape, TriangularMesh,\
     inertia_tensor

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
        
        #Read and store the signal
        signal, x, y, z, x_length, y_length, z_length = read_dicom(scan_folder)
        self.signal = signal[slice_idxs]
        
        #Read and store the coordinates of the centroids and the side length of
        #each voxel. The 'geometry' attribute is arranged as follows:
        #self.geometry[:,:,:,0-2] -> x, y and z coordinates of the centroid of
        #each voxel
        #self.geometry[:,:,:,3-5] -> side length of each voxel along x, y and z
        self.geometry = np.zeros((*self.signal.shape, 6))
        self.geometry[:,:,:,0] = x[slice_idxs]
        self.geometry[:,:,:,1] = y[slice_idxs]
        self.geometry[:,:,:,2] = z[slice_idxs]
        self.geometry[:,:,:,3] = x_length[slice_idxs]
        self.geometry[:,:,:,4] = y_length[slice_idxs]
        self.geometry[:,:,:,5] = z_length[slice_idxs]        
        
        self.empty = False
    
    def get_voxel_centroid_coordinates(self):
        """Coordinates of the centroid of each voxel
        
        Returns
        -------
        x, y, z : nparray of float
            The coordinates of the centroid of each voxel
        """
        return [self.geometry[:,:,:,i] for i in (0,1,2)]
    
    def get_voxel_dims(self):
        """Dimensions of each voxel along x, y and z
        
        Returns
        -------
        x_length, y_length, z_length : nparray of float
            The dimension (side length) of each voxel along x, y and z
        """
        return [self.geometry[:,:,:,i] for i in (3,4,5)]
    
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
        x, y, z = self.get_voxel_centroid_coordinates()
        x_spacing = np.abs(np.diff(x, n = 1, axis = 1))
        y_spacing = np.abs(np.diff(y, n = 1, axis = 0))
        z_spacing = np.abs(np.diff(z, n = 1, axis = 2))
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
                         self.geometry,
                         self.diagnosis), f)           
            
    def _load(self, source):
        """Load the ROI from a pickle file
        
        Parameters
        ----------
        source : str
            The source file (.pkl)
        """
        
        with open(source, 'rb') as f:
            self.mask, self.signal, self.geometry, self.diagnosis =\
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
    
    def get_mesh_volume(self):
        """Returns the roi volume computed on the triangulara mesh
        
        Returns
        -------
        mesh_volume : float
            The volume.
        """
        mesh_volume = self.get_mask_mesh().get_volume()
        return mesh_volume    
    
    def get_principal_moments_mask(self):
        """Principal moments of inertia computed on the mask (non signal-weighted)
        
        Returns
        -------
        principal_moments : nparray of float (3)
            The principal moments of inertia sorted in descending order of 
            magnitude: [I1, I2, I3]
        """
        _, principal_moments = inertia_tensor(self.x, self.y, self.z, 
                                              self.get_mask())
        return principal_moments
    
    def get_principal_moments_signal(self):
        """Principal moments of inertia computed on the signal-weighted mask
        
        Returns
        -------
        principal_moments : nparray of float (3)
            The principal moments of inertia sorted in descending order of 
            magnitude: [I1, I2, I3]
        """
        _, principal_moments = inertia_tensor(
            self.x, self.y, self.z, 
            np.multiply(self.get_signal(), self.get_mask()))   
        return principal_moments
    
    def get_roi_dimensions(self):
        """Returns the dimensions of the ROI along the axial, sagittal and coronal
        axis
        
        Returns
        -------
        roi_dimensions : float (3)
            The dimensions of the ROI respectively along the coronal, sagittal
            and axial directions.
        """
        roi_dimensions = (np.abs(self.x[-1,-1,-1] - self.x[0,0,0]),
                          np.abs(self.y[-1,-1,-1] - self.y[0,0,0]),
                          np.abs(self.z[-1,-1,-1] - self.z[0,0,0]))
        return roi_dimensions
    
    def get_roi_volume(self):
        """Returns the volume of the roi, that is of the axis-aligned bounding box
        
        Returns
        -------
        volumes : nparray (S,C,A)
            The volume of each voxel in the roi. The axes respectively represent
            the sagittal, coronal and axial direction.
        overall_volume : float
            The overall volume (sum of the volumes of each voxel).
        """
        
        #Compute the volumes of all the voxels
        x_length, y_length, z_length = self.get_voxel_dims()
        volumes = x_length * y_length * z_length
        overall_volume = np.sum(volumes.flatten())
        
        return volumes, overall_volume        
    
    def get_voxel_volume(self):
        """Returns the total volume as the sum of the volume of each voxel for
        which the corresponding mask value is True.
        
        Returns
        -------
        voxel_volume : float
            The voxel volume.
        """
        
        volumes, _ = self.get_roi_volume()
        valid_volumes = self.get_mask() * volumes
        voxel_volume = np.sum(valid_volumes.flatten())
        
        return voxel_volume
        
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

            
