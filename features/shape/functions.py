"""General functions for computing shape features

References
----------
[1] Benn, D.I., Ballantyne, C.K.
    The description and representation of particle shape
    (1993) Earth Surface Processes and Landforms, 18 (7), pp. 665-672. 
"""

def shape_index(a, b, c, index):
    """One among some of the shape indices as deifned in [1, Tab. 1]
    
    Parameters
    ----------
    a, b, c : float
        The lengths along three orthogonal axes. These may represent the 
        dimensions of the ROI as well as the axes of the inertia ellipsoid.
    index : str
        A string indicating the index requested. Possible values are:
            
        
    Returns
    -------
    value : float
        The value of the shape index.
    """
    value = None
    if index == 'disc-rod':
        value = (a - b)/(a - c)
    elif index == 'oblate-prolate':
        value = 10 * ((a - b)/(a - c) - 1/2) / (c/a) 
    else:
        raise Exception('Shape index not supported')

    return value