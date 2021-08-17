"""Other shape features

References
----------
[1] Zwanenburg, A. et al. 
    The image biomarker standardisation initiative (rev. v11)
    https://arxiv.org/abs/1612.07003
"""

import numpy as np

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