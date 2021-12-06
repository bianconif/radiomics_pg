"""Shape features obtained by combining the length measured along three principal 
 (orthogonal) dimensions: length (l), breadth (b) and thickness (t). The l, b 
 and t values can be:
 1) The side lengths of the axis-aligned bounding box (aabb);
 2) The axis of the ellipsoid of inertia computed on the mask; that is, on a
    uniform-density body;
 3) The axis of the ellipsoid of inertia computed on the image signal (limited
    to the mask); that is, on a non-uniform-density body.

References
----------
[1] Benn, D.I., Ballantyne, C.K.
    The description and representation of particle shape
    (1993) Earth Surface Processes and Landforms, 18 (7), pp. 665-672.
[2] Blott, S.J., Pye, K. Particle shape: A review and new methods of 
    characterization and classification (2008) Sedimentology, 55 (1), pp. 31-63.
[3] Angelidakis, V., Nadimi, S., Utili, S. Elongation, flatness and compactness 
    indices to characterise particle form (2022) Powder Technology, 396, 
    pp. 689-695.
[4] Kong, D., Fonseca, J. Quantification of the morphology of shelly carbonate 
    sands using 3D images (2018) Geotechnique, 68 (3), pp. 249-261
"""
import numpy as np

from radiomics_pg.utilities.geometry import axes_inertia_equivalent_ellipsoid

def lbt_index(roi, index, mode):
    """Shape indices based on length, breadth and thickness.
    
    Parameters
    ----------
    roi : Roi
        The roi on which the index is computed.
    index : str
        A string indicating the index to compute. See _lbt_index() for possible
        values.
    mode : str
        The modality used to compute the length, breadth and thickness of the 
        ROI. Possible values are:
            'aabb'  (axis-aligned bounding box). Length, breadth and thickness 
                    are those of the axis-aligned bounding box.
            'ae'    (non signal-weighted approximating ellipsoid). Length,
                    breadth and thickness are the axes of an ellipsoid with the 
                    same inertia as the mask, where each voxel is given
                    weight '1'.
            'swae'  (signal-weighted approximating ellipsoid) Length,
                    breadth and thickness are the axes of an ellipsoid with the 
                    same inertia as the signal-weighted mask.
            
    Returns
    -------
    value : float
        The value of the requested index.
    """
    
    def _inertia_to_lbt(A, B, C, M):
        a, b, c = axes_inertia_equivalent_ellipsoid(A, B, C, M)
        l, b, t = np.sort(np.array([a, b, c]))[::-1]
        return l, b, t
        
    
    #Compute length, breadth and thickness
    l, b, t = None, None, None
    if mode == 'aabb':
        dims = list(roi.get_dimensions())
        dims.sort(reverse = True)
        l, b, t = dims
    elif mode == 'ae':
        A, B, C = roi.get_principal_moments_mask()
        M = np.sum(roi.get_mask().flatten())
        l, b, t = _inertia_to_lbt(A, B, C, M)
    elif  mode == 'swae':
        A, B, C = roi.get_principal_moments_signal()
        M = np.sum(np.multiply(roi.get_mask(), roi.get_signal(zero_min = True))\
                   .flatten())
        l, b, t = _inertia_to_lbt(A, B, C, M)

    else:
        raise Exception(f'Mode *{mode}* not supported')
    
    #Compute the requested index
    value = _lbt_index(l, b, t, index)
    
    return value
    

def _lbt_index(a, b, c, index):
    """One among some of the shape indices as defined in [1, Tab. 1]
    
    Parameters
    ----------
    a, b, c : float
        The lengths along three orthogonal axes. These may represent the 
        dimensions of the ROI as well as the axes of the inertia ellipsoid.
    index : str
        A string indicating the index requested. Possible values are:
            - 'angelidakis-elongation' ([3, eq. 5])
            - 'angelidakis-flatness' ([3, eq. 5])
            - 'angelidakis-compactness' ([3, eq. 5])
            - 'breadth-to-length' (breadth/length ratio, represents a measure
               of elongation [2])
            - 'cailleux-flatness' (Cailleux flatness)
            - 'csi' (Corey shape index)
            - 'disc-rod' (Disc-rod index)
            - 'krumbein-sphericity' (Krumbein sphericity)
            - 'mps' (Maximum projection sphericity)
            - 'oblate-prolate' (Oblate-prolate index)
            - 'lt-percent-diff' (Percentage difference between length and
                                 thickness)
            - 'thickness-to-breadth' (thickness/breadth ratio, represents a
               measure of flatness [2])
            - 'thickness-to-length' (thickness/length ratio, represents a
               measure of equancy [2])
            - 'wentworth-flatness' (Wentworth flatness)
            
        
    Returns
    -------
    value : float
        The value of the shape index.
    """
    value = None
    if index == 'angelidakis-elongation':
        value = (a * c)/(a * c + b ** 2) - c/(a + c)
    elif index == 'angelidakis-flatness':
        value = (b ** 2)/(a * c + b ** 2) - c/(a + c)
    elif index == 'angelidakis-compactness':
        value = (2 * c)/(a + c)     
    elif index == 'breadth-to-length':
        value = b/a
    elif index == 'cailleux-flatness':
        value = 1000 * (a + b) / (2 * c)
    elif index == 'csi':
        value = c / ((a * b) ** (1/2))
    elif index == 'disc-rod':
        threshold = 1.0
        lt_perc_diff = _lbt_index(a, b, c, 'lt-percent-diff')
        if lt_perc_diff <= threshold:
            value = 1.0
        else:
            value = (a - b)/(a - c)
    elif index == 'kong-elongation':
        value = 1 - b/a
    elif index == 'kong-flatness':
        value = 1 - c/b
    elif index == 'krumbein-sphericity':
        value = ((c * b) / (a ** 2)) ** (1/3)   
    elif index == 'mps':
        value = (c**2 / (a*b)) ** (1/3)
    elif index == 'lt-percent-diff':
        value = 100 * np.abs(a - c)/np.mean([a,c])
    elif index == 'oblate-prolate':
        threshold = 1.0
        lt_perc_diff = _lbt_index(a, b, c, 'lt-percent-diff')
        if lt_perc_diff <= threshold:
            value = 1.0
        else:        
            value = 10 * ((a - b)/(a - c) - 1/2) / (c/a) 
    elif index == 'thickness-to-breadth':
        value = c/b 
    elif index == 'thickness-to-length':
        value = c/a         
    elif index == 'wentworth-flatness':
        value = (a + b)/c
    else:
        raise Exception('Shape index not supported')

    return value