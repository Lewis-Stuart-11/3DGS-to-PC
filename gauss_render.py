### Based on code from the Torch Splatting Repo     ###
### Credit: https://github.com/hbb1/torch-splatting ###

import torch

from math import tan, floor

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
    det = cov2d[:, 0, 0] * cov2d[:,1,1] - cov2d[:, 0, 1] * cov2d[:,1,0]
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()

@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max

class GaussRenderer():
    """
    A gaussian splatting renderer
    """

    def __init__(self, means3D, opacity, colour, cov3d, white_bkgd=True):
        self.white_bkgd = white_bkgd

        self.device = means3D.get_device()

        # Tensor of the maximum contributions each gaussian made 
        self.gaussian_max_rep = torch.zeros(means3D.shape[0], device=self.device)

        # Tensor of new gaussian colours calculated for point cloud generation
        self.gaussian_colours = torch.zeros((means3D.shape[0], 3), device=self.device, dtype=torch.double)

        self.means3D =  means3D
        self.opacity = opacity 
        self.cov3d = cov3d
        self.colour = colour

    def get_colours(self):
        """ returns the new calculated gaussian colours """
        return self.gaussian_colours * 255

    def get_seen_gaussians(self):
        """ returns indices of gaussians that have been rendered """
        return self.gaussian_max_rep > 0
    
    def render(self, camera, means2D, cov2d, color, opacity, depths, projection_mask, render_img=False):
        
        radii = get_radius(cov2d)

        rect = get_rect(means2D, radii, width=camera.image_width, height=camera.image_height)
        
        self.render_color = torch.ones(*self.pix_coord.shape[:2], 3).to(self.device)
        self.render_depth = torch.zeros(*self.pix_coord.shape[:2], 1).to(self.device)
        self.render_alpha = torch.zeros(*self.pix_coord.shape[:2], 1).to(self.device)

        avg_gaussians_per_pixel = means2D.shape[0]/(camera.image_width)

        recommended_gaussians_per_tile = 300**2

        TILE_SIZE =  max(min(int(recommended_gaussians_per_tile/avg_gaussians_per_pixel),40), 5) 

        for w in range(0, camera.image_width, TILE_SIZE):
            for h in range(0, camera.image_height, TILE_SIZE):
                height_tile_size = min(TILE_SIZE, camera.image_height-h)
                width_tile_size = min(TILE_SIZE, camera.image_width-w)
                
                # check if the rectangle penetrate the tile
                over_tl = (rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h))
                over_br = (rect[1][..., 0].clip(max=w+width_tile_size-1), rect[1][..., 1].clip(max=h+height_tile_size-1))
                in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1]) # 3D gaussian in the tile 

                if not in_mask.sum() > 0:
                    continue

                P = in_mask.sum()

                tile_coord = self.pix_coord[h:h+height_tile_size, w:w+width_tile_size].flatten(0,-2)

                sorted_depths, index = torch.sort(depths[in_mask]) 

                index = torch.flip(index, [0,])

                inverse_index = index.argsort(0)

                sorted_means2D = means2D[in_mask][index]
                sorted_cov2d = cov2d[in_mask][index] 
                sorted_conic = sorted_cov2d.inverse() 
                sorted_opacity = opacity[in_mask][index]
                sorted_color = color[in_mask][index]

                dx = (tile_coord[:,None,:] - sorted_means2D[None,:]) 

                gauss_weight = torch.exp(-0.5 * (
                    dx[:, :, 0]**2 * sorted_conic[:, 0, 0] 
                    + dx[:, :, 1]**2 * sorted_conic[:, 1, 1]
                    + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 0, 1]
                    + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 1, 0]))

                alpha = (gauss_weight[..., None] * sorted_opacity[None]).clip(max=0.99) 
                T = torch.cat([torch.ones_like(alpha[:,:1]), 1-alpha[:,:-1]], dim=1).cumprod(dim=1)

                acc_alpha = (alpha * T).sum(dim=1)

                tile_color = (T * alpha * sorted_color[None]).sum(dim=1) + (1-acc_alpha) * (1 if self.white_bkgd else 0)
                #tile_depth = ((T * alpha) * sorted_depths[None,:,None]).sum(dim=1)

                self.render_color[h:h+height_tile_size, w:w+width_tile_size] = tile_color.reshape(height_tile_size, width_tile_size, -1)
                #self.render_depth[h:h+height_tile_size, w:w+width_tile_size] = tile_depth.reshape(height_tile_size, width_tile_size, -1)
                #self.render_alpha[h:h+height_tile_size, w:w+width_tile_size] = acc_alpha.reshape(height_tile_size, width_tile_size, -1)

                # Calculate Representaion of each gaussian in tile and update colours if current rep is largest

                combined_mask = torch.zeros_like(self.gaussian_max_rep, dtype=torch.bool)
                combined_mask[projection_mask] = in_mask

                current_gaussian_reps = self.gaussian_max_rep[combined_mask]

                indices_in_mask = combined_mask.nonzero(as_tuple=True)[0]

                rep = ((T * alpha)).squeeze(2)[:, inverse_index]

                biggest_rep_in_tile = torch.max(rep, 0)

                biggest_rep_in_tile_vals = biggest_rep_in_tile[0]
                biggest_rep_in_tile_pixel = biggest_rep_in_tile[1]

                new_gaussians = biggest_rep_in_tile_vals > current_gaussian_reps

                new_gaussian_mask_indices = new_gaussians.nonzero()

                gaussians_to_update = indices_in_mask[new_gaussian_mask_indices]

                self.gaussian_max_rep[gaussians_to_update] = biggest_rep_in_tile_vals[new_gaussians].unsqueeze(1)
                self.gaussian_colours[gaussians_to_update] = tile_color[biggest_rep_in_tile_pixel[new_gaussians]].unsqueeze(1)

        self.render_color =  torch.flip(self.render_color, [1,])

        return self.render_color

       #""" return {
       #     "render": self.render_color,
       #     #"depth": self.render_depth,
       #     #"alpha": self.render_alpha,
       #     "visiility_filter": radii > 0,
       #     "radii": radii,
       #     "gaussian_colours": self.gaussian_colours
       # }"""

    def add_img(self, camera, **kwargs):

        self.pix_coord = torch.stack(torch.meshgrid(torch.arange(camera.image_width), torch.arange(camera.image_height), indexing='xy'), dim=-1).to(self.device)

        cov2d = build_covariance_2d(
                mean3d=self.means3D, 
                cov3d=self.cov3d, 
                viewmatrix=camera.world_view_transform,
                fov_x=camera.FoVx, 
                fov_y=camera.FoVy, 
                focal_x=camera.focal_x, 
                focal_y=camera.focal_y)

        mean_ndc, mean_view, in_mask = projection_ndc(self.means3D, 
                    viewmatrix=camera.world_view_transform, 
                    projmatrix=camera.projection_matrix)

        mean_ndc = mean_ndc[in_mask]
        mean_view = mean_view[in_mask]
        depths = mean_view[:,2]
        cov2d = cov2d[in_mask]
        opacity = self.opacity[in_mask]
        current_color = self.colour[in_mask]

        mean_coord_x = ((mean_ndc[..., 0] + 1) * camera.image_width - 1.0) * 0.5
        mean_coord_y = ((mean_ndc[..., 1] + 1) * camera.image_height - 1.0) * 0.5
        means2D = torch.stack([mean_coord_x, mean_coord_y], dim=-1)

        rets = self.render(
            camera=camera, 
            means2D=means2D,
            cov2d=cov2d,
            color=current_color,
            opacity=opacity, 
            depths=depths,
            projection_mask=in_mask
        )

        return rets
