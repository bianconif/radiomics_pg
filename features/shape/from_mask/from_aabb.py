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
    """Ratio between the volume of mask and the volume of the axis-aligned
    bounding box.
    
    Parameters
    ----------
    roi : Roi
        The input roi.
    
    Returns
    -------
    vdensity : float
        The volume density (dimensionless units).
    """