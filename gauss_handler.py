import numpy as np
import torch
import time

from math import floor

"""
These functions convert a gaussian scale and rotation into a covariance matrix.
Originally provided: https://github.com/graphdeco-inria/gaussian-splatting
"""

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(q):
    #norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    #q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)

    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = torch.exp(s[:,0])
    L[:,1,1] = torch.exp(s[:,1])
    L[:,2,2] = torch.exp(s[:,2])

    L = R @ L
    return L

def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

class Gaussians():
    """
    Manages all loaded gaussians in the renderer
    """

    def __init__(self, xyz, scales, rots, colours, opacities):
        self.xyz = xyz
        self.scales = scales
        self.rots = rots
        self.opacities = opacities
        self.colours = colours
        self.normals = None

        self.scaling_modifier = 1.0

        # Calculates 3D covariance matrices
        covariances = build_covariance_from_scaling_rotation(scales, self.scaling_modifier, rots)

        # Validate and add covariances matrices
        self.validate_and_add_covariances(covariances)

    def calculate_normals(self):
        # Choose the smallest side of the Gaussian for the normal 
        min_values = torch.min(self.scales, 1)

        # Create normal matrix
        normal_matrices = torch.zeros(self.xyz.shape, dtype=torch.float, device=self.xyz.get_device())
        normal_matrices[torch.arange(self.xyz.shape[0]), min_values[1]] = 1
        
        # Rotate normal by the rotation matrix
        R = build_rotation(self.rots)
        normal_matrices = normal_matrices.unsqueeze(1)
        normals = torch.bmm(R,  normal_matrices.permute(0, 2, 1))

        self.normals = normals.permute(0, 2, 1).squeeze(1)

    def non_posdef_covariances(self, covariances):
        """
        Returns a boolean mask of which covariances are definitepositive
        """
        return torch.any(torch.linalg.eigvals(covariances).real <= 0, 1)

    def clamp_covariances(self, covariances, mask=None, epsilon=1e-6):
        """
        Clips Eigenvalues to positive to enforce covariances to be positive-definite
        Credit: MultiTrickFox
        """
        
        if mask is None:
            mask = torch.ones(covariances.shape[0], dtype=torch.bool)
            
        eigvals, eigvecs = torch.linalg.eigh(covariances[mask])
        eigvals = torch.clamp(eigvals, min=epsilon)
        covariances[mask] = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)

        return covariances

    def regularise_covariances(self, covariances, mask=None, epsilon=1e-6):
        """
        Increases the value of the diagonal of the covariances matrices to ensure it is positive-definite
        """
        if mask is None:
            mask = torch.ones(covariances.shape[0], dtype=torch.bool)

        eye_matrix = epsilon * torch.eye(3, device=self.xyz.get_device()).expand(mask.sum(), 3, 3)
        covariances[mask] += eye_matrix

        return covariances

    def validate_and_add_covariances(self, covariances):
        """
        Regularises Gaussian covariances and ensures that all covariances are positive-definite
        since sometimes gaussian values are slightly wrong when loaded (most likely due to floating point errors)
        """

        # Regularises gaussians with a low factor, which ensures almost all covaricnes are positive-definite
        covariances = self.regularise_covariances(covariances)
        
        # Check if any non positive-definite covariances exist, if so, clamp gaussians to ensure
        # all covariances are positive-definite
        non_positive_covariances = self.non_posdef_covariances(covariances)
        if non_positive_covariances.sum() > 0:
            covariances = self.clamp_covariances(covariances, mask=non_positive_covariances, epsilon=0.001)

        self.covariances = covariances

        # If still not positive-definite then delete erroneous Gaussians
        non_positive_covariances = self.non_posdef_covariances(covariances)
        if non_positive_covariances.sum() > 0:
            self.filter_gaussians(~non_positive_covariances)

    def filter_gaussians(self, filter_indices):
        """
        Filters gaussians based on given indices
        """

        self.xyz = self.xyz[filter_indices]
        self.scales = self.scales[filter_indices]
        self.rots = self.rots[filter_indices]
        self.colours = self.colours[filter_indices]
        self.opacities = self.opacities[filter_indices]
        self.covariances = self.covariances[filter_indices]
        
        if self.normals is not None:
            self.normals = self.normals[filter_indices]

    def apply_min_opacity(self, min_opacity):
        """
        Removes gaussians with opacity lower than the min_opacity
        """

        if min_opacity > 0.0:
            valid_gaussians_indices = self.opacities > (min_opacity)

            self.filter_gaussians(valid_gaussians_indices)

    def apply_bounding_box(self, bounding_box_min, bounding_box_max):
        """
        Removes gaussians outside of the bounding box
        """

        valid_gaussians_indices = torch.logical_not(torch.zeros(self.xyz.shape[0], dtype=torch.bool, device=self.xyz.get_device()))

        if bounding_box_min is not None:
            bounding_box_min_indices = (self.xyz[:,0] > bounding_box_min[0]) & (self.xyz[:,1] > bounding_box_min[1]) \
                                    & (self.xyz[:,2] > bounding_box_min[2])

            valid_gaussians_indices = valid_gaussians_indices & bounding_box_min_indices

        if bounding_box_max is not None:
            bounding_box_max_indices = (self.xyz[:,0] < bounding_box_max[0]) & (self.xyz[:,1] < bounding_box_max[1]) \
                                    & (self.xyz[:,2] < bounding_box_max[2])

            valid_gaussians_indices = valid_gaussians_indices & bounding_box_max_indices

        self.filter_gaussians(valid_gaussians_indices)

    """def apply_k_nearest_neighbours(self, k=10):
        tree = KDTree(self.xyz.detach().cpu().numpy())

        distances, indices = tree.query(gaussian_positions, k=k)

        total_distances = (np.sum(distances, axis=1)/k)

        invalid_gaussians = torch.tensor(total_distances > max_dist, device="cuda:0")"""

    def cull_large_gaussians(self, cull_gauss_size_percent):
        """
        Orders the gaussians by size and removes gaussians with a size greater than the 'cull_gauss_size_percent'
        """

        gaussian_sizes = self.get_gaussian_sizes()

        cull_index = floor(gaussian_sizes.shape[0] *(1-cull_gauss_size_percent))

        sorted_sizes, sorted_indices = torch.sort(gaussian_sizes)

        culled_gaussians = sorted_indices[:cull_index]

        self.filter_gaussians(culled_gaussians)

    def get_gaussian_sizes(self):
        """
        Orders the gaussians by volume size after calculating the exponent (to prioritse larger Gaussians)
        """
        return torch.sqrt(torch.sum(torch.pow(torch.exp(self.scales * self.scaling_modifier), 2), axis=1))



            
        

