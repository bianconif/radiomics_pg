import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage.measure import marching_cubes, mesh_surface_area



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
    
    #Add 1 to the upper bounds for correct slicing
    upper_bounds += 1
    
    bbox = np.zeros((len(lower_bounds), 2), dtype = np.int)
    bbox[:,0] = lower_bounds
    bbox[:,1] = upper_bounds
        
    slice_indices = list()
    for i in range(bbox.shape[0]):
        slice_indices.append(bbox[i,:].tolist())
    idxs = tuple(slice(s[0],s[1], 1) for s in slice_indices)
    data_slice = data[idxs] 
    
    return bbox, data_slice, idxs

def inertia_tensor(x, y, z, mass_distro, normalise_mass = False, central = True):
    """Inertia tensor of a discrete mass ditribution
    
    Parameters
    ----------
    x, y, z : nparray of numeric
        The coordinates of each point of the mass distribution.
    mass_distro : nparray of numeric
        The mass of each point.
    normalise_mass : bool
        If True normalise the mass distribution to sum 1.
    central : bool
        Whether the tensor should be computed around the centroid.
        
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
        
    if central:
        x = x - centroid(x, mass_distro)
        y = y - centroid(y, mass_distro)
        z = z - centroid(z, mass_distro)
    
    #Compute the inertia tensor
    i_xx = np.sum((y**2 + z**2)*mass_distro)
    i_yy = np.sum((x**2 + z**2)*mass_distro)
    i_zz = np.sum((x**2 + y**2)*mass_distro)
    i_xy = -np.sum((x*y)*mass_distro)
    i_yz = -np.sum((y*z)*mass_distro)
    i_xz = -np.sum((x*z)*mass_distro)
    itensor = np.array([[i_xx, i_xy, i_xz],
                        [i_xy, i_yy, i_yz],
                        [i_xz, i_yz, i_zz]])
    
    #Compute the principal moments
    principal_moments, _ = np.linalg.eig(itensor)
    principal_moments = np.sort(principal_moments)
    principal_moments = np.flip(principal_moments)
    return itensor, principal_moments
    
def zingg_shape(x, y, z, mass_distro):
    """Zingg shape parameters and classification
    
    Parameters
    ----------
    x, y, z : nparray of numeric
        The coordinates of each point of the mass distribution.
    mass_distro : nparray of numeric
        The mass of each point.
        
    Returns
    -------
    r1, r2 : float
        Respectively b/a and c/b, where a, b and c indicate the principal 
        inertia moments of the mass distribution sorted in descending order 
        (a > b > c).
    zingg_class : int (possible values = 0, 1, 2 and 3)
        The Zingg class:
            0 -> rod (prolate)
            1 -> blade (oblate)
            2 -> sphere (equant)
            3 -> disc
         
    References
    ----------
    [1] Domokos, G., Sipos, A., Szabo, T., Varkonyi, P.
        Pebbles, Shapes, and Equilibria
        (2010) Mathematical Geosciences, 42 (1), pp. 29-47. 
    """
    
    _, principal_moments = inertia_tensor(x, y, z, mass_distro, 
                                          normalise_mass = True)
    
    r1 = principal_moments[1]/principal_moments[0]
    r2 = principal_moments[2]/principal_moments[1]
    
    condition_1 = (r1 >= 2/3)
    condition_2 = (r2 >= 2/3)
    zingg_class = 2*condition_1 + condition_2
    
    return r1, r2, zingg_class

class TriangularMesh():
    """Wrapper for triangular mesh"""
    
    @staticmethod
    def by_marching_cubes(roi):
        """Generate a triangular mesh from a given ROI
        
        Parameters
        ----------
        roi : Roi
            The ROI object
            
        Returns
        -------
        tmesh : TriangularMesh
            The triangular mesh
        """
        mask = roi.get_mask()
        spacing_x, spacing_y, spacing_z = roi.get_average_spacing()
        spacing = (spacing_y, spacing_x, spacing_z)
        
        #Pad zeros around the mask to avoid border effects
        padded_mask = np.pad(array = mask, pad_width = 1, mode = 'constant')
        
        verts, faces, normals, _ = marching_cubes(volume = padded_mask, 
                                                  level = 0.5,
                                                  spacing = spacing, 
                                                  allow_degenerate = False)
        
        return TriangularMesh(verts, faces, normals)
        
    def __init__(self, verts, faces, normals):
        """
        Parameters
        ----------
        verts : (V, 3) nparray
            Spatial coordinates of the vertices unique mesh vertices. 
        faces : (F, 3) nparray
            Triangular faces via referencing vertex indices from verts. 
        normals : (V, 3) array
            The normal direction at each vertex, as calculated from the data.
        """
        self.verts = verts
        self.faces = faces
        self.normals = normals
        
    def show(self):
        """Display the mesh
        
        Sourced from 
        """
        # Display the triangular mesh using Matplotlib. This can also be done
        # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh = Poly3DCollection(self.verts[self.faces])
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)
        
        ax.set_xlabel("Coronal")
        ax.set_ylabel("Sagittal")
        ax.set_zlabel("Axial")
        
        #Grant extra space around the mesh for plotting
        extra_margin_frac = 0.30
        min_ = [np.min(self.verts[:,i]) for i in range(3)]
        max_ = [np.max(self.verts[:,i]) for i in range(3)]
        spread = np.abs(np.subtract(max_, min_))
        upper_limits = [M + spread[i]*extra_margin_frac for i, M in enumerate(max_)]
        lower_limits = [m - spread[i]*extra_margin_frac for i, m in enumerate(min_)]
        
        xlims = (lower_limits[0], upper_limits[0])
        ylims = (lower_limits[1], upper_limits[1])
        zlims = (lower_limits[2], upper_limits[2])
        
        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
        ax.set_zlim(*zlims)
        
        plt.tight_layout()
        plt.show()  
        
    def _compute_surface_area(self):
        """Computes the area of the triangular mesh and stores is as an attribute
        """
        self.surface_area = mesh_surface_area(self.verts, self.faces)
        
    def _compute_volume(self):
        """Computes the volume of the triangular mesh and stores it as an 
        attribute"""
        
        #Computes the volume of each tetrahedron whose vertices are the vertices 
        #of each triangular face of the mesh and the origin of the reference system. 
        #Denote with O, A, B and C respectively the origin and vertices of each 
        #tetrahedron. Let also n be the normal of ABC and G the centroid. 
        #The volume of the tetrahedron is considered positive if dot(OG, n) > 0; 
        #negative otherwise.  
        
        #First vertex of the triangular face
        xA = self.verts[self.faces[:,0],0]
        yA = self.verts[self.faces[:,0],1]
        zA = self.verts[self.faces[:,0],2]
        
        #Second vertex of the triangular face
        xB = self.verts[self.faces[:,1],0]
        yB = self.verts[self.faces[:,1],1]
        zB = self.verts[self.faces[:,1],2]
        
        #Third vertex of the triangular face
        xC = self.verts[self.faces[:,2],0]
        yC = self.verts[self.faces[:,2],1]
        zC = self.verts[self.faces[:,2],2]
        
        #Triangles' centroids
        G = np.zeros((self.faces.shape[0],3))
        origin = np.zeros((self.faces.shape[0],3))
        G[:,0] = (xA + xB + xC)/3 
        G[:,1] = (yA + yB + yC)/3 
        G[:,2] = (zA + zB + zC)/3        
                        
        #Compute the volume of each tetrahedron
        AB = np.zeros((self.faces.shape[0],3))
        AO = np.zeros((self.faces.shape[0],3))
        AC = np.zeros((self.faces.shape[0],3))
        AB[:,0] = xB - xA
        AB[:,1] = yB - yA
        AB[:,2] = zB - zA
        AC[:,0] = xC - xA
        AC[:,1] = yC - yA
        AC[:,2] = zC - zA
        AO[:,0] = 0 - xA
        AO[:,1] = 0 - yA
        AO[:,2] = 0 - zA
        
        #Volume of each tetrahedron = |AO . (AB x AC)|
        AB_outer_AC = np.zeros((self.faces.shape[0],3))
        AB_outer_AC[:,0] =   (AB[:,1]*AC[:,2] - AB[:,2]*AC[:,1])
        AB_outer_AC[:,1] = - (AB[:,0]*AC[:,2] - AB[:,2]*AC[:,0])
        AB_outer_AC[:,2] =   (AB[:,0]*AC[:,1] - AB[:,1]*AC[:,0])
        
        #Compute the dot product between the origin-centroid vector and the
        #face normal
        dot_prods = np.einsum('ij,ij->i', AB_outer_AC, AO)
        
        #From the dot products get the sign of the volume of each tetrahedron
        signs = np.where(dot_prods > 0, 
                         np.ones(dot_prods.shape[0]),
                         -1 * np.ones(dot_prods.shape[0]))        
        
        unsigned_vols = np.abs(np.einsum('ij,ij->i', AO, AB_outer_AC)/6)
        signed_vols = signs * unsigned_vols
        
        #Total volume
        self.volume = np.sum(signed_vols)
        #***********************************************
        #***********************************************
        #*********************************************** 
        
    def get_surface_area(self):
        """Returns the surface area. Does not recompute it if cached.
        
        Returns
        -------
        area : float
            The surface area.
        """
        area = None
        try:
            self.surface_area
        except AttributeError:
            self._compute_surface_area()
        
        area = self.surface_area            
        return area
    
    def get_volume(self):
        """Returns the volume. Does not recompute it if cached
        
        Returns
        -------
        volume : float
            The volume.
        """     
        volume = None
        try:
            self.volume
        except AttributeError:
            self._compute_volume()
        
        volume = self.volume            
        return volume            
             
        
            
            

    