### Based on code from the Torch Splatting Repo     ###
### Credit: https://github.com/hbb1/torch-splatting ###

import torch
import copy 

from math import tan, floor, ceil

# Constant values for calculating spherical harmonics
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

homogeneous = lambda points: torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def eval_sh(deg, sh, dirs=None):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        assert dirs is not None
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def build_covariance_2d(mean3d: torch.tensor, cov3d: torch.tensor, viewmatrix: torch.tensor,
                        fov_x: float, fov_y: float, focal_x: float, focal_y: float):
    """
    Converts a 3D gaussian, from a mean and covariance matrix, into a 2D covariance matrix for rendering.

    Args:
        mean3d (torch.tensor): tensor of centres for gaussians
        cov3d (torch.tensor): tensor of 3D covariance matrices for gaussians
        viewmatrix (torch.tensor): 4x4 view matrix of the camera
        fov_x (float), fov_y (float), focal_x (float), focal_y (float): intrinsic camera parameters

    returns:
        conv2D: tensor of 2D covariance matrices

    """
    
    # The following models the steps outlined by equations 29
    # and 31 in "EWA Splatting" (Zwicker et al., 2002). 
    # Additionally considers aspect / scaling of viewport.
    # Transposes used to account for row-/column-major conventions.
    tan_fovx = tan(fov_x * 0.5)
    tan_fovy = tan(fov_y * 0.5)

    t = (mean3d @ viewmatrix[:3,:3]) + viewmatrix[-1:,:3]

    # truncate the influences of gaussians far outside the frustum.
    tx = (t[..., 0] / t[..., 2]).clip(min=-tan_fovx*1.3, max=tan_fovx*1.3) * t[..., 2]
    ty = (t[..., 1] / t[..., 2]).clip(min=-tan_fovy*1.3, max=tan_fovy*1.3) * t[..., 2]
    tz = t[..., 2]

    # Eq.29 locally affine transform 
    # perspective transform is not affine so we approximate with first-order taylor expansion
    # notice that we multiply by the intrinsic so that the variance is at the sceen space
    J = torch.zeros(mean3d.shape[0], 3, 3).to(mean3d)
    J[..., 0, 0] = 1 / tz * focal_x
    J[..., 0, 2] = -tx / (tz * tz) * focal_x
    J[..., 1, 1] = 1 / tz * focal_y
    J[..., 1, 2] = -ty / (tz * tz) * focal_y
    # J[..., 2, 0] = tx / t.norm(dim=-1) # discard
    # J[..., 2, 1] = ty / t.norm(dim=-1) # discard
    # J[..., 2, 2] = tz / t.norm(dim=-1) # discard
    W = viewmatrix[:3,:3].T # transpose to correct viewmatrix

    cov2d = J @ W @ cov3d @ W.T @ J.permute(0,2,1)
    
    # add low pass filter here according to E.q. 32
    filter = torch.eye(2,2).to(cov2d) * 0.3
    return cov2d[:, :2, :2] + filter[None]


def projection_ndc(points: torch.tensor, viewmatrix: torch.tensor, projmatrix: torch.tensor):
    """
    Converts points into ndc and filters points not in camera frame

    Args:
        points (torch.tensor): points to convert to ndc
        viewmatrix (torch.tensor): the view matrix of the current camera
        projmatrix (torch.tensor): the projection matrix of the current camera
    """
    points_o = homogeneous(points) # object space
    points_h = points_o @ viewmatrix @ projmatrix # screen space # RHS
    p_w = 1.0 / (points_h[..., -1:] + 0.000001)
    p_proj = points_h * p_w
    p_view = points_o @ viewmatrix

    # Points in the camera frame are in the mask
    in_mask = p_view[..., 2] <= -0.000001
    return p_proj, p_view, in_mask


@torch.no_grad()
def get_radius(cov2d):
    """
    Get the 2D radii of each gaussian 
    """
    det = cov2d[:, 0, 0] * cov2d[:,1,1] - cov2d[:, 0, 1] * cov2d[:,1,0]
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()

@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    """
    Calculate render tiles
    """
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max

class GaussRenderer():
    """
    A Gaussian splatting renderer
    """

    def __init__(self, means3D, opacity, colour, cov3d, white_bkgd=True):
        self.white_bkgd = white_bkgd

        self.device = means3D.get_device()

        # Tensor of the maximum contributions each Gaussian made 
        self.gaussian_max_contribution = torch.zeros(means3D.shape[0], device=self.device)

        # Tensor of new Gaussian colours calculated for point cloud generation
        self.gaussian_colours = torch.zeros((means3D.shape[0], 3), device=self.device, dtype=torch.double)

        """# Tensor of surface Gaussians
        self.surface_gaussian_idxs = torch.zeros(means3D.shape[0], device=self.device)"""

        self.means3D =  means3D
        self.opacity = opacity 
        self.cov3d = cov3d
        self.colour = colour

    def get_colours(self):
        """ 
        Returns the new calculated Gaussian colours 
        """
        return self.gaussian_colours * 255

    def get_seen_gaussians(self):
        """ 
        Returns indices of Gaussians that have been rendered 
        """
        return self.gaussian_max_contribution > 0

    def get_surface_gaussians(self):
        """
        Returns indices of Gaussians that are predicted to be on the surface of the scene
        """
        return self.gaussian_max_contribution > torch.mean(self.gaussian_max_contribution)

    def render(self, camera, means2D, cov2d, colour, opacity, depths, projection_mask, max_tile_size=60, max_gaussians_per_tile=60000):
        """
        Renders an image given a set of Gaussians and camera transform

        Args:
            camera: the camera parameters to render the image
            means2D: positions of the Gaussians in 2D spacse
            cov2D: 2D covariance matrices for Gaussians
            colour: colours of the Gaussians
            opacity: opacity of the Gaussians
            depths: depths of the Gaussians
            projection_mask: mask that filters Gaussians not included in camera frame
        Returns:
            render_image: the rendered RGB image
        """

        with torch.no_grad():
            radii = get_radius(cov2d)

            rect = get_rect(means2D, radii, width=camera.image_width, height=camera.image_height)
            
            render_colour = torch.ones(*self.pix_coord.shape[:2], 3).to(self.device)

            # Stores a list of tiles that need to be processed
            tile_queue = [[[0,0], [camera.image_width, camera.image_height]]]

            # Loop until the tile queue is empty
            while tile_queue != []:
                current_tile = tile_queue.pop(0)

                # Get the corner pixel of the current tile and the size
                start_px = current_tile[0]
                tile_size = current_tile[1]

                # If the tile size is less than 2 in any dimension then it is erroneous
                if tile_size[0] <= 1 or tile_size[1] <= 1:
                    continue 

                tile_size[0] = min(tile_size[0], camera.image_width-start_px[1])
                tile_size[1] = min(tile_size[1], camera.image_height-start_px[0])

                # Calculate if Gaussian is in tile
                over_tl = (rect[0][..., 0].clip(min=start_px[1]), rect[0][..., 1].clip(min=start_px[0]))
                over_br = (rect[1][..., 0].clip(max=start_px[1]+tile_size[0]-1), rect[1][..., 1].clip(max=start_px[0]+tile_size[1]-1))
                tile_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1]) 

                # If there are no gaussians in the tile then set as the background colour
                if tile_mask.sum() <= 0:
                    render_colour[start_px[0]: start_px[0]+tile_size[1], start_px[1]: start_px[1]+tile_size[0], :] = 0 if not self.white_bkgd else 1
                    continue

                # If the current tile is larger than the max tile size, or has more gaussians than the maximum per tile, 
                # then divide into four seperate tiles and add to queue
                if tile_mask.sum() > max_gaussians_per_tile or tile_size[0] > max_tile_size or tile_size[1] > max_tile_size:

                    tile_size[0] = ceil(tile_size[0]/2)
                    tile_size[1] = ceil(tile_size[1]/2)
                    tile_queue.append([copy.deepcopy(start_px), copy.deepcopy(tile_size)])

                    start_px[0] += floor(tile_size[1])
                    tile_queue.append([copy.deepcopy(start_px), copy.deepcopy(tile_size)])
                    
                    start_px[0] -= floor(tile_size[1])
                    start_px[1] += floor(tile_size[0])
                    tile_queue.append([copy.deepcopy(start_px), copy.deepcopy(tile_size)])
                    
                    start_px[0] += floor(tile_size[1])
                    tile_queue.append([copy.deepcopy(start_px), copy.deepcopy(tile_size)])

                    continue

                tile_coord = self.pix_coord[start_px[0]: start_px[0]+tile_size[1], start_px[1]: start_px[1]+tile_size[0]].flatten(0,-2)
                        
                # Order Gaussians based on the depth (descending away from cam)
                sorted_depths, index = torch.sort(depths[tile_mask]) 

                index = torch.flip(index, [0,])

                inverse_index = index.argsort(0)

                # Filter Gaussians to only those in mask and reorder
                sorted_means2D = means2D[tile_mask][index]
                sorted_cov2d = cov2d[tile_mask][index] 
                sorted_conic = sorted_cov2d.inverse() 
                sorted_opacity = opacity[tile_mask][index]
                sorted_colour = colour[tile_mask][index]

                dx = (tile_coord[:,None,:] - sorted_means2D[None,:]) 

                # Calculate contributions of each Gaussian
                gauss_weight = torch.exp(-0.5 * (
                    dx[:, :, 0]**2 * sorted_conic[:, 0, 0] 
                    + dx[:, :, 1]**2 * sorted_conic[:, 1, 1]
                    + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 0, 1]
                    + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 1, 0]))

                # Calculate alpha and transmittance of each Gaussian in pixel
                alpha = (gauss_weight[..., None] * sorted_opacity[None]).clip(max=0.99) 
                T = torch.cat([torch.ones_like(alpha[:,:1]), 1-alpha[:,:-1]], dim=1).cumprod(dim=1)
                acc_alpha = (alpha * T).sum(dim=1)

                tile_colour = (T * alpha * sorted_colour[None]).sum(dim=1) + (1-acc_alpha) * (1 if self.white_bkgd else 0)

                render_colour[start_px[0]: start_px[0]+tile_size[1], start_px[1]: start_px[1]+tile_size[0]] = tile_colour.reshape(tile_size[1], tile_size[0], -1)

                # Get the current max representations of each Gaussian by applying projection and tile mask
                combined_mask = torch.zeros_like(self.gaussian_max_contribution, dtype=torch.bool)
                combined_mask[projection_mask] = tile_mask
                current_gaussian_reps = self.gaussian_max_contribution[combined_mask]

                indices_in_mask = combined_mask.nonzero(as_tuple=True)[0]

                # Calculate the represntation of each Gaussian for the current pixels 
                # This is the amount it contributed to the pixel colour and is what is used to determine what colour the points of each Gaussian should have!
                contribution = ((T * alpha)).squeeze(2)[:, inverse_index]

                # Get what pixel the Gaussian contributed the most and what its biggest contribution value was
                biggest_contribution_in_tile = torch.max(contribution, 0)
                biggest_contribution_in_tile_vals = biggest_contribution_in_tile[0]
                biggest_contribution_in_tile_pixel = biggest_contribution_in_tile[1]
                # Filter Gaussians that have a new biggest contribution
                new_gaussians = biggest_contribution_in_tile_vals > current_gaussian_reps

                new_gaussian_mask_indices = new_gaussians.nonzero()

                gaussians_to_update = indices_in_mask[new_gaussian_mask_indices]

                # Update the colours and maximum contributions
                self.gaussian_max_contribution[gaussians_to_update] = biggest_contribution_in_tile_vals[new_gaussians].unsqueeze(1)
                self.gaussian_colours[gaussians_to_update] = tile_colour[biggest_contribution_in_tile_pixel[new_gaussians]].unsqueeze(1)

                """# Get the Gaussian that contributed most for each pixel in the tile, and set the surface Gaussian ids to 1 for these Gaussians
                biggest_contribution_per_pixel = torch.max(contribution, 1)[1]
                surface_gaussians_to_update = indices_in_mask[biggest_contribution_per_pixel]
                self.surface_gaussian_idxs[surface_gaussians_to_update] += 1"""

            return torch.flip(render_colour, [1,])

    def add_img(self, camera, **kwargs):
        """
        Renders an image from the given camera and updates the gaussian colours 
        """
        
        with torch.no_grad():
            # Reset the pixel coordinates to current camera parameters
            self.pix_coord = torch.stack(torch.meshgrid(torch.arange(camera.image_width), torch.arange(camera.image_height), indexing='xy'), dim=-1).to(self.device)

            # Calculate 2D coveriance matrix
            cov2d = build_covariance_2d(
                    mean3d=self.means3D, 
                    cov3d=self.cov3d, 
                    viewmatrix=camera.world_view_transform,
                    fov_x=camera.FoVx, 
                    fov_y=camera.FoVy, 
                    focal_x=camera.focal_x, 
                    focal_y=camera.focal_y)

            # Project Gaussians into 2D and filter Gaussians outside of view range
            mean_ndc, mean_view, in_mask = projection_ndc(self.means3D, 
                        viewmatrix=camera.world_view_transform, 
                        projmatrix=camera.projection_matrix)

            mean_ndc = mean_ndc[in_mask]
            mean_view = mean_view[in_mask]
            depths = mean_view[:,2]
            cov2d = cov2d[in_mask]
            opacity = self.opacity[in_mask]
            current_colour = self.colour[in_mask]

            mean_coord_x = ((mean_ndc[..., 0] + 1) * camera.image_width - 1.0) * 0.5
            mean_coord_y = ((mean_ndc[..., 1] + 1) * camera.image_height - 1.0) * 0.5
            means2D = torch.stack([mean_coord_x, mean_coord_y], dim=-1)

            # Estimate of the maximum Gaussians for each tile for the provided amount of memory
            estimated_mem_per_gaussian = 175000
            max_gaussians_per_tile = int((torch.cuda.mem_get_info()[1] - torch.cuda.memory_allocated(self.device))/estimated_mem_per_gaussian)

            # Estimate of the max tile size for the provided amount of memory
            max_tile_size = max_gaussians_per_tile//1000

            # Attempts to render image with set tile size (for speed). If this fails then the tile size is reduced
            while max_tile_size >= 5 and max_gaussians_per_tile > 5000:
                try:
                    return self.render(
                        camera=camera, 
                        means2D=means2D,
                        cov2d=cov2d,
                        colour=current_colour,
                        opacity=opacity, 
                        depths=depths,
                        projection_mask=in_mask,
                        max_tile_size=max_tile_size,
                        max_gaussians_per_tile=max_gaussians_per_tile
                    )

                except Exception:
                    max_gaussians_per_tile -= 5000
                    max_tile_size -= 5

            raise Exception("Failed to render image")

