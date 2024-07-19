### Based on code from the Torch Splatting Repo     ###
### Credit: https://github.com/hbb1/torch-splatting ###

import math
import numpy as np
import torch

"""def generate_cam_positions(radius, rings=5, sectors=5):
    ring_d = 1.0/(rings)
    sect_d = 1.0/(sectors)

    cut_off = 0.3
    ring_d *= (1-cut_off)

    cam_positions = []

    for sect in range(sectors):
        for ring in range(1, rings+1, 1):
            x = cos(2 * pi * sect * sect_d) 
            y = sin(2 * pi * sect * sect_d) 
                
            x *= sin(pi * ring * ring_d)
            y *= sin(pi * ring * ring_d)

            z = -sin(-(pi/2) + (pi * ring * ring_d))

            x *= radius
            y *= radius
            z *= radius

            cam_positions.append([x,y,z])

    return cam_positions

def generate_predicted_dataset_cams(radius, conex_pattern = True):
    pass"""

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

