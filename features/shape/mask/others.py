"""Morphological (shape) features based on the ROI mask (signal intensity
is not considered)

References
----------
[1] Zwanenburg, A. et al. 
    The image biomarker standardisation initiative (rev. v11)
    https://arxiv.org/abs/1612.07003
"""

import numpy as np

from utilities.misc import Roi


def surface_to_volume_ratio(roi):
    """The surface to volume ratio as defined in paragraph 3.1.4 of [1]).
    Surface area and volume are both computed from the triangular mesh.

    Parameters
    ----------
    roi : Roi
        The input roi.

    Returns
    -------
    sv_ratio : float
        The surface to volume ratio. The dimension is length**(-1).
    """
    return roi.get_surface_area()/roi.get_mesh_volume()

def compactness_1(roi):
    """Deviation of the ROI volume from a representative spheroid
    as defined in paragraph 3.1.5 of [1]. Surface area and volume are both
    computed from the triangular mesh.
    
    Parameters
    ----------
    roi : Roi
        The input roi.

    Returns
    -------
    compactness : float
        The compactness (dimensionless units).
    """
    
    A = roi.get_surface_area()
    V = roi.get_mesh_volume()
    compactness = V / (np.pi**2 * A**(3/2))
    return compactness

    
        

