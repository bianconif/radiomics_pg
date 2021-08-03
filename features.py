"""Radiomics features computed on a ROI"""
import numpy as np

from utilities.misc import Roi

class RadiomicsFeatures():
    """Base class for radiomics features"""
    
    def __init__(self, roi):
        """
        Parameters
        ----------
        roi : Roi
            The Roi on which the features are computed.
        """ 
        self.roi = roi

class Shape(RadiomicsFeatures):
    """Morphological (shape features)
    
    References
    ----------
    [1] Zwanenburg, A. et al. 
        The image biomarker standardisation initiative (rev. v11)
        https://arxiv.org/abs/1612.07003
    """
    
    def surface_to_volume_ratio(self):
        """The surface to volume ratio as defined in paragraph 3.1.4 of [1]).
        Surface area and volume are both computed from the triangular mesh.
        
        Returns
        -------
        sv_ratio : float
            The surface to volume ratio. The dimension is length**(-1).
        """
        return self.roi.get_surface_area()/self.roi.get_mesh_volume()
    
    def compactness_1(self):
        """Deviation of the ROI volume from a representative spheroid
        as defined in paragraph 3.1.5 of [1]. Surface area and volume are both 
        computed from the triangular mesh.
        
        Returns
        -------
        compactness : float
            The compactness value- The dimension is 1/sqrt(length)
        """
        A = self.roi.get_surface_area()
        V = self.roi.get_mesh_volume()
        return V/(np.pi**(1/2) * A**(3/2))