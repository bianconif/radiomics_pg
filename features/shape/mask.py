"""Shape parameters computed on the mask data

References
----------
[1] Zwanenburg, A. et al. 
    The image biomarker standardisation initiative (rev. v11)
    https://arxiv.org/abs/1612.07003
"""

import numpy as np
from itertools import combinations
from scipy.spatial.distance import pdist

def voxel_volume(roi):
    """The volume computed by summing up the volume of each voxel in the mask
    (paragraph 3.1.2 of [1]).
    
    Parameters
    ----------
    roi : Roi
        The input roi.
    
    Returns
    -------
    voxel_volume : float
        The volume density (units: length^3).
    """    
    return roi.get_voxel_volume()

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
    _, roi_volume = roi.get_volume()
    return roi.get_voxel_volume()/roi_volume

def area_density(roi):
    """Ratio between the surface area of the mask and the area of the axis-aligned
    bounding box [1, 3.1.18].
    
    Parameters
    ----------
    roi : Roi
        The input roi.
    
    Returns
    -------
    adensity : float
        The area density (dimensionless units).
    """   
    dims = length_breadth_thickness(roi)
    bounding_box_area = 0
    combs = combinations(dims, 2)
    for comb in combs:
        bounding_box_area = bounding_box_area + 2*comb[0]*comb[1]
    
    return roi.get_surface_area()/bounding_box_area

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
    dimensions = roi.get_dimensions()
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

def sphericity(roi):
    """Ratio between the surface area of a sphere with the same volume as the given 
    ROI and the surface area of the ROI (paragraph 3.1.8 of [1]).
    
    Parameters
    ----------
    roi : Roi
        The input roi.

    Returns
    -------
    sphericity : float
        The sphericity (dimensionless units).
    """
    
    A = roi.get_surface_area()
    V = roi.get_mesh_volume()
    sphericity = ((36 * np.pi * V ** 2) ** (1/3))/A
    return sphericity

def max_3d_diameter(roi):
    """The distance between the centroid of the most apart voxels. Note that
    this definition is slightly different from that proposed in [1, par. 3.1.11]
    where the distance is computed on the triangular mesh.
    
    Parameters
    ----------
    roi : Roi
        The input roi.

    Returns
    -------
    max_3d_diameter : float
        The maximum pairwise distance between the centroids of the two
        most apart voxels (dimension: length).
    """
    centroid_coords = roi.get_voxel_centroid_coordinates()
    mask = roi.get_mask()
    ndims = len(centroid_coords)
    
    #Make up the observation matrix
    M = np.zeros((np.sum(mask.flatten().astype(np.int)), ndims))
    for d in range(ndims):
        M[:,d] = centroid_coords[d][mask == True].flatten()
        
    #Compute the pairwise distances
    pairwise_dists = pdist(M, metric='euclidean')
    max_3d_diameter = np.max(pairwise_dists)
    
    return max_3d_diameter

