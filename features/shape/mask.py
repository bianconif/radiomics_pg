"""Shape parameters computed on the mask data

References
----------
[1] Zwanenburg, A. et al. 
    The image biomarker standardisation initiative (rev. v11)
    https://arxiv.org/abs/1612.07003
"""

import numpy as np

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

