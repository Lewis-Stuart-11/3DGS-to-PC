#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    mask: torch.tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool

class GaussianRasterizer(nn.Module):
    def __init__(self,  means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, 
                cov3D_precomp = None, visible_gaussian_threshold=0.0, surface_distance_std=None, calculate_surface_distance=False):
        super().__init__()

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        self.means3D = means3D
        self.means2D = means2D
        self.opacities = opacities 
        self.shs = shs if shs is not None else torch.Tensor([])
        self.colors_precomp = colors_precomp if colors_precomp is not None else torch.Tensor([])
        self.scales = scales if scales is not None else torch.Tensor([])
        self.rotations = rotations if rotations is not None else torch.Tensor([])
        self.cov3D_precomp = cov3D_precomp if cov3D_precomp is not None else torch.Tensor([])

        self.device = torch.device(f"cuda:{means3D.get_device()}")

        # Tensor of the maximum contributions each Gaussian made 
        self.gaussian_max_contribution = torch.zeros(means3D.shape[0], device=self.device, dtype=torch.float)

        self.gaussian_min_surface_distance = torch.full((means3D.shape[0],), torch.finfo(torch.float).max, device=self.device, dtype=torch.float) 

        # Tensor of total contributions of each Gaussian
        self.gaussian_total_contribution = torch.zeros(means3D.shape[0], device=self.device, dtype=torch.float)

        # Tensor of new Gaussian colours calculated for point cloud generation
        self.gaussian_colours = torch.zeros((means3D.shape[0], 3), device=self.device, dtype=torch.float)

        # Value to filter low visbility Gaussians
        self.visible_gaussian_threshold = visible_gaussian_threshold

        # Whether or not to calculate the distance of Gaussians the predicted depth
        self.surface_distance_std = surface_distance_std

        # Set to calculate surface distance
        self.calculate_surface_distance = calculate_surface_distance

    """def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible"""

    def forward(self, raster_settings):
        """
        Render new image and calculate Gaussian contributions
        """

        if raster_settings.mask is None:
            mask = torch.full((raster_settings.image_height * raster_settings.image_width,), 1, device=self.device, dtype=torch.int)
        else:
            mask = raster_settings.mask

        args = (
            raster_settings.bg, 
            self.means3D,
            self.colors_precomp,
            self.opacities,
            self.scales,
            self.rotations,
            raster_settings.scale_modifier,
            self.cov3D_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            self.shs,
            raster_settings.sh_degree,
            raster_settings.campos,
            mask,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            self.calculate_surface_distance,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        num_rendered, colour, depths, radii, geomBuffer, binningBuffer, imgBuffer, invdepths, current_gauss_contributions, current_gauss_surface_distances, current_gauss_pixels = _C.rasterize_gaussians(*args)

        colour_flat = colour.permute(1,2,0).contiguous().view(-1, 3)

        current_gauss_pixels = current_gauss_pixels.to(torch.long)

        # Get new Gaussian colours from returned pixel values
        current_gauss_colours = colour_flat[current_gauss_pixels]
        
        # Update contributions
        self.update_max_contributions(current_gauss_contributions, current_gauss_colours)

        self.update_min_surface_distances(current_gauss_surface_distances)

        return colour, radii, invdepths, depths

    def update_max_contributions(self, new_gauss_contributions, new_gauss_colours):
        """ 
        Updates maximum contribiutions and gaussian colours based on the contributions of the rendered image 
        """

        gaussians_to_update = new_gauss_contributions > self.gaussian_max_contribution

        self.gaussian_max_contribution[gaussians_to_update] = new_gauss_contributions[gaussians_to_update]
        self.gaussian_colours[gaussians_to_update] = new_gauss_colours[gaussians_to_update]

        self.gaussian_total_contribution += new_gauss_contributions

    def update_min_surface_distances(self, new_gauss_surface_distances):

        gaussians_to_update = new_gauss_surface_distances < self.gaussian_min_surface_distance 

        self.gaussian_min_surface_distance[gaussians_to_update] = new_gauss_surface_distances[gaussians_to_update]

    def get_gaussian_colours(self):
        """ 
        Returns the new calculated Gaussian colours 
        """
        return self.gaussian_colours * 255

    def get_max_gaussian_contributions(self):
        """
        Returns the maximum contributions that each Gaussian made to a pixel
        """
        return self.gaussian_max_contribution

    def get_total_gaussian_contributions(self):
        """
        Returns the total contributions that each Gaussian made to every rendered image
        """
        return self.gaussian_total_contribution

    def get_gaussians_above_contribution_threshold(self, contribution_threshold):
        """ 
        Returns mask of Gaussians that have a contribution above a given threshold
        """
        return self.get_max_gaussian_contributions() > contribution_threshold

    def get_gaussians_above_total_contribution_threshold(self, contribution_threshold):
        """ 
        Returns mask of Gaussians that have a total contribution above a given threshold
        """
        return self.get_total_gaussian_contributions() > contribution_threshold

    def get_surface_gaussians_below_distance_threshold(self, surface_distance_threshold):
        """
        Returns mask of Gaussians with the lowest surface distance below a given threshold
        """
        if not self.calculate_surface_distance:
            raise Exception("Cannot determine Gaussian surface distance as this feature was not set at the start of rendering")

        surface_indices = (self.gaussian_min_surface_distance < torch.finfo(torch.float).max)

        mean_and_std = torch.std_mean(self.gaussian_min_surface_distance[surface_indices])

        return self.gaussian_min_surface_distance < mean_and_std[1] * surface_distance_threshold

    def get_visible_gaussians(self):
        """ 
        Returns mask of Gaussians that have a large visible contribution to rendered images (according to the visible gaussian threshold)
        """
        return self.get_gaussians_above_contribution_threshold(self.visible_gaussian_threshold)
    
    def get_gaussians_with_low_surface_distance(self):
        """
        Returns mask of Gaussians that are close to the surface (according to the surface distance standard deviation)
        """
        return self.get_surface_gaussians_below_distance_threshold(self.surface_distance_std)

    def get_predicted_surface_gaussians(self, predicted_surface_std=0.5):
        """
        Returns mask of Gaussians that are predicted to be on the surface of the scene
        """
        return self.get_surface_gaussians_below_distance_threshold(predicted_surface_std)
        #return self.get_gaussians_above_contribution_threshold(torch.mean(self.get_max_gaussian_contributions()))

