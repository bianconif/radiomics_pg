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
"""

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
            'aabb' -> length, breadth and thickness are those of the axis-aligned
            bounding box;
            'mask' -> length, breadth and thickness are respectively the length 
            (sorted in descending order) of the three axes of the ellipsoid of 
            inertia computed on the mask (non signal-wighted);
            'signal' -> length, breadth and thickness are respectively the length 
            (sorted in descending order) of the three axes of the ellipsoid of 
            inertia computed on the signal-weighted mask.
            
    Returns
    -------
    value : float
        The value of the requested index.
    """
    
    #Compute length, breadth and thickness
    l, b, t = None, None, None
    if mode == 'aabb':
        dims = list(roi.get_roi_dimensions())
        dims.sort(reverse = True)
        l, b, t = dims
    elif mode == 'mask':
        l, b, t = roi.get_principal_moments_mask()
    elif  mode == 'signal':
        l, b, t = roi.get_principal_moments_signal()
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
            - 'cailleux-flatness' (Cailleux flatness)
            - 'csi' (Corey shape index)
            - 'disc-rod' (Disc-rod index)
            - 'krumbein-spericity' (Krumbein sphericity)
            - 'mps' (Maximum projection sphericity)
            - 'oblate-prolate' (Oblate-prolate index)
            - 'wentworth-flatness' (Wentworth flatness)
            
        
    Returns
    -------
    value : float
        The value of the shape index.
    """
    value = None
    if index == 'cailleux-flatness':
        value = 1000 * (a + b) / (2 * c)
    elif index == 'csi':
        value = c / ((a * b) ** (1/2))
    elif index == 'disc-rod':
        value = (a - b)/(a - c)
    elif index == 'krumbein-sphericity':
        value = ((c * b) / (a ** 2)) ** (1/3)   
    elif index == 'mps':
        value = (c**2 / (a*b)) ** (1/3)
    elif index == 'oblate-prolate':
        value = 10 * ((a - b)/(a - c) - 1/2) / (c/a) 
    elif index == 'wentworth-flatness':
        value = (a + b)/c
    else:
        raise Exception('Shape index not supported')

    return value