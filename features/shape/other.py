"""Other shape features

References
----------
[1] Zwanenburg, A. et al. 
    The image biomarker standardisation initiative (rev. v11)
    https://arxiv.org/abs/1612.07003
"""

import numpy as np

from radiomics_pg.features.shape.mask import max_3d_diameter

def centre_of_mass_shift(roi):
    """The distance between the signal (intensity-weighted) centroid and the 
    mask centroid (paragraph 3.1.10 of [1]).
    
    Parameters
    ----------
    roi : Roi
        The input roi.
    
    Returns
    -------
    cm_shift : float
        The centre of mass shift (length units).
    """
    w_centroid = np.array(
        roi.get_mask_centroid(mode = 'intensity-weighted'))
    nw_centroid = np.array(
        roi.get_mask_centroid(mode = 'non-intensity-weighted'))    
    cm_shift = np.linalg.norm(w_centroid - nw_centroid, ord = 2)
    return cm_shift

def normalised_centre_of_mass_shift(roi):
    """Ratio between the centre of mass shift and the maximum 3D diameter
    
    Parameters
    ----------
    roi : Roi
        The input roi.
    
    Returns
    -------
    normalised_cm_shift : float
        The normalised centre of mass shift (dimensionless units).
    """
    return centre_of_mass_shift(roi)/max_3d_diameter(roi)