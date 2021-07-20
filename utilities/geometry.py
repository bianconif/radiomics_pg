import numpy as np

def cross_moment(coordinates_a, coordinates_b, mass_distro, order_a = 2, 
                 order_b = 2, central = False):
    """Central cross-moment of a discrete mass distribution
    
    Parameters
    ----------
    coordinates_a : nparray
        The coordinates of the mass distribution along the first axis.
    coordinates_b : nparray
        The coordinates of the mass distribution along the second axis.
    mass_distro : nparray
        The mass distribution.
    order_a : int
        Order of the moment along the first axis.
    order_b : int
        Order of the moment along the second axis.
    central : bool
        Whether the moment should be computed around the centroid.
    
    Returns
    -------
    moment : float
        The value of the moment.
    
    Notes
    -----
    Parameters coordinates_a, coordinates_b and mass_distro should all have the 
    same size.
    """    
    mass_distro = mass_distro.flatten()
    coordinates_a = coordinates_a.flatten()
    coordinates_b = coordinates_b.flatten()
    
    if central:
        centroid_a = centroid(coordinates_a, mass_distro)
        centroid_b = centroid(coordinates_b, mass_distro)
        coordinates_a = coordinates_a - centroid_a
        coordinates_b = coordinates_b - centroid_b
    
    a_x_b = np.multiply(np.power(coordinates_a, order_a), 
                        np.power(coordinates_b, order_b))    
    moment = np.sum(np.dot(a_x_b, mass_distro))
    return moment

def moment(coordinates, mass_distro, order = 2, central = False):
    """Central n-th moment of a discrete mass distribution
    
    Parameters
    ----------
    coordinates : nparray
        The coordinates of the mass distrbution along one given axis.
    mass_distro : nparray
        The mass distribution.
    order : int
        Order of the moment.
    central : bool
        Whether the moment should be computed around the centroid.
    
    Returns
    -------
    moment : float
        The value of the moment.
    
    Notes
    -----
    Parameters coordinates and mass_distro should have the same size.
    """            
    moment = cross_moment(coordinates, coordinates, mass_distro, order, central)
    return moment
        
def centroid(coordinates, mass_distro):
    """Centroid of a discrete mass distribution
    
    Parameters
    ----------
    coordinates : nparray
        The coordinates of the mass distrbution along one given axis.
    mass_distro : nparray
        The mass distribution.
    
    Returns
    -------
    coordinate : float
        The coordinate of the centroid along the given axis.
    
    Notes
    -----
    Parameters coordinates and mass_distro should have the same size.
    """

    coordinate = moment(coordinates, mass_distro, order = 1)/np.sum(mass_distro)
    return coordinate

def bounding_box(data):
    """Bounding box for non-zero values in an n-dimensional array
    
    Parameters
    ----------
    data : numpy array
        The input data
    
    Returns
    -------
    bbox : nparray (d,2)
        The bounding box, where d is the dimension of the input data.
    data_slice : nparray
        The slice of the input data corresponding to the bounding box.
    idxs : nparray
        Indices corresponding to the slice (data_slice = data[idxs]). 
    """
    
    where_non_zero = np.argwhere(data)
    lower_bounds = np.min(where_non_zero, axis = 0)
    upper_bounds = np.max(where_non_zero, axis = 0)
    
    bbox = np.zeros((len(lower_bounds), 2), dtype = np.int)
    bbox[:,0] = lower_bounds
    bbox[:,1] = upper_bounds
    
    slice_indices = list()
    for i in range(bbox.shape[0]):
        slice_indices.append(bbox[i,:].tolist())
    idxs = tuple(slice(s[0],s[1], 1) for s in slice_indices)
    data_slice = data[idxs] 
    
    return bbox, data_slice, idxs

def inertia_tensor(x, y, z, mass_distro, normalise_mass = False):
    """Inertia tensor of a discrete mass ditribution
    
    Parameters
    ----------
    x, y, z : nparray of numeric
        The coordinates of each point of the mass distribution.
    mass_distro : nparray of numeric
        The mass of each point.
    normalise_mass : bool
        If True normalise the mass distribution to sum 1.
        
    Returns
    -------
    itensor : nparray of float (3,3)
        The tensor of inertia: [Ix, -Ixy, Ixz; Ixy, Iy, Iyz; Ixz, Iyz, Iz]
    principal_moments : nparray of float (3)
        The principal moments of inertia sorted in descending order of 
        magnitude: [I1, I2, I3]
    
    Notes
    -----
    All the input parameters need to have the same shape.
    """
    
    if not (x.shape == y.shape == z.shape == mass_distro.shape):
        raise Exception('x, y, z and mass_distro must all have the same shape')
    
    if normalise_mass:
        mass_distro = mass_distro/np.sum(mass_distro[:])
    
    #Compute the inertia tensor
    i_x = np.sum((y**2 + z**2)*mass_distro)
    i_y = np.sum((x**2 + z**2)*mass_distro)
    i_z = np.sum((x**2 + y**2)*mass_distro)
    i_xy = np.sum((x*y)*mass_distro)
    i_yz = np.sum((y*z)*mass_distro)
    i_xz = np.sum((x*z)*mass_distro)
    itensor = np.array([[i_x, -i_xy, -i_xz],
                        [-i_xy, i_y, -i_yz],
                        [-i_xz, -i_yz, i_z]])
    
    #Compute the principal moments
    principal_moments, _ = np.linalg.eig(itensor)
    principal_moments = np.sort(principal_moments)
    a = 0
    