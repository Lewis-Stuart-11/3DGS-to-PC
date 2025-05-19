### Based on code from the Torch Splatting Repo     ###
### Credit: https://github.com/hbb1/torch-splatting ###

import math
import numpy as np
import torch

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

class Camera():
    def __init__(self, width, height, focal_x, focal_y, c2w, znear=10, zfar=100):
        self.znear = znear
        self.zfar = zfar
        self.focal_x = focal_x 
        self.focal_y = focal_y
        self.FoVx = focal2fov(self.focal_x, width)
        self.FoVy = focal2fov(self.focal_y, height)
        self.image_width = int(width)
        self.image_height = int(height)
        self.world_view_transform = torch.linalg.inv(c2w).permute(1,0)
        self.c2w = c2w
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(c2w.device)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix


def get_camera(renderer_type, transform, cam_intrinsic, colour_resolution=1920, sh_degree=3):
    diff = colour_resolution / int(cam_intrinsic[0])

    img_width = int(int(cam_intrinsic[0]) * diff) 
    img_height = int(int(cam_intrinsic[1]) * diff) 

    focal_x = float(cam_intrinsic[2])*diff
    focal_y = float(cam_intrinsic[3])*diff

    if renderer_type == "python":
        return Camera(img_width, img_height, focal_x, focal_y, transform)

    elif renderer_type == "cuda":
        from gaussian_pointcloud_rasterization import GaussianRasterizationSettings

        transform[:, 1:3] = -transform[:, 1:3]

        fovX = focal2fov(focal_x, img_width)
        fovY = focal2fov(focal_y, img_height)

        tanfovx = math.tan(fovX * 0.5)
        tanfovy = math.tan(fovY * 0.5)

        scaling_modifier = 1.0
        white_bkgd = True
        
        znear = 10
        zfar = 100

        projmatrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovX, fovY=fovY).transpose(0,1).to(transform.device)

        viewmatrix = torch.linalg.inv(transform).permute(1,0)
        campos = viewmatrix.inverse()[3, :3]

        return GaussianRasterizationSettings(
            image_height=int(img_height),
            image_width=int(img_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.tensor([0.,0.,0.], device="cuda:0") if not white_bkgd else torch.tensor([1.0,1.0,1.0], device="cuda:0"),
            scale_modifier=scaling_modifier,
            campos=campos, 
            viewmatrix=viewmatrix,
            projmatrix=viewmatrix @ projmatrix,
            sh_degree=sh_degree,
            prefiltered=False,
            debug=True,
            antialiasing=False
        )


