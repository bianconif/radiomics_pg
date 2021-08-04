"""Morphological (shape) features based on the axis-aligned bounding box around
the ROI mask.

References
----------
[1] Zwanenburg, A. et al. 
    The image biomarker standardisation initiative (rev. v11)
    https://arxiv.org/abs/1612.07003
[2] Blott, S.J., Pye, K.
    Particle shape: A review and new methods of characterization and classification
    (2008) Sedimentology, 55 (1), pp. 31-63
[3] Benn, D.I., Ballantyne, C.K.
    The description and representation of particle shape
    (1993) Earth Surface Processes and Landforms, 18 (7), pp. 665-672. 
"""

import numpy as np

from features.shape.functions import shape_index
from utilities.misc import Roi

def volume_density(roi):
    """Ratio between the volume of the mask and the volume of the axis-aligned
    bounding box (paragraph 3.1.17 of [1]). Also referred to as 'Rectangular fit'.
    
    Parameters
    ----------
    roi : Roi
        The input roi.
    
    Returns
    -------
    vdensity : float
        The volume density (dimensionless units).
    """
    _, roi_volume = roi.get_roi_volume()
    return roi.get_voxel_volume()/roi_volume

def length_breadth_thickness(roi):
    """The length, breadth and thickness of the ROI, that is, the three
    dimensions of the ROI sorted in descending order
    
    Parameters
    ----------
    roi : Roi
        The input roi.
    
    Returns
    -------
    lbt : float(3)
        The length, breadth and thickness of the ROI
    """
    dimensions = roi.get_roi_dimensions()
    lbt = np.sort(np.array(dimensions))
    return tuple(reversed(lbt))

def zingg_ratios(roi):
    """The breadth/length and thickness/breadth ratios as defined by
    Zingg [2]
    
    Parameters
    ----------
    roi : Roi
        The input roi.
    zratios : tuple(2)
        The breadth/length and thickness/breadth ratios.
    """
    lbt = length_breadth_thickness(roi)
    zratios = (lbt[1]/lbt[0], lbt[2]/lbt[1])
    return zratios

def disc_rod_index(roi):
    """The disc-rod index as defined in [3, Tab. 1]
    
    Parameters
    ----------
    roi : Roi
        The input roi.
        
    Returns
    -------
    dr_index : float
        The disc-rod index (dimensionless units).
    """   
    lbt = length_breadth_thickness(roi)
    return shape_index(*lbt, 'disc-rod')

def oblate_prolated_index(roi):
    """The oblate-prolate index as defined in [3, Tab. 1]
    
    Parameters
    ----------
    roi : Roi
        The input roi.
        
    Returns
    -------
    op_index : float
        The oblate-prolate index (dimensionless units).
    """   
    lbt = length_breadth_thickness(roi)
    return shape_index(*lbt, 'oblate-prolate')
    