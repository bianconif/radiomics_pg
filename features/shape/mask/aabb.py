"""Morphological (shape) features based on the axis-aligned bounding box around
the ROI mask.

References
----------
[1] Zwanenburg, A. et al. 
    The image biomarker standardisation initiative (rev. v11)
    https://arxiv.org/abs/1612.07003
"""

from utilities.misc import Roi

def volume_density(roi):
    """Ratio between the volume of the mask and the volume of the axis-aligned
    bounding box. Also referred to as 'Rectangular fit'.
    
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