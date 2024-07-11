import os
import numpy as np
import torch

from plyfile import PlyData, PlyElement

def load_splat_data():
    pass

def load_ply_data(path, max_sh_degree=3):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)

    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    active_sh_degree = max_sh_degree

    features_all = torch.cat((torch.tensor(features_dc, device="cuda:0"), torch.tensor(features_extra, device="cuda:0")), 2)

    opacities = (1 / (1 + torch.exp(torch.tensor(-opacities, device="cuda:0")))).type(torch.float).squeeze(1) # torch.full(opacities.shape, 0.05, device="cuda:0")-

    xyz_tensor = torch.tensor(xyz, device="cuda:0")
    scales_tensor = torch.tensor(scales, device="cuda:0")
    rots_tensor = torch.tensor(rots, device="cuda:0")

    return xyz_tensor, scales_tensor, rots_tensor, features_all, opacities
    
    
def save_xyz_to_ply(xyz_points, filename, rgb_colors=None):
    """
    Save a series of XYZ points to a PLY file.

    Parameters:
    xyz_points (numpy.ndarray): An array of shape (N, 3) containing XYZ coordinates.
    filename (str): The name of the output PLY file.
    """
    # Ensure the points are in the correct format
    assert xyz_points.shape[1] == 3, "Input points should be in the format (N, 3)"

    if rgb_colors is None:
        rgb_colors = np.repeat(np.array([[255, 255, 255]]), len(xyz_points), axis=0)

    xyz_and_colours = np.hstack((xyz_points, rgb_colors))

    # Create a structured array
    vertex = np.array([tuple(point) for point in xyz_and_colours], 
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    
    # Create a PlyElement
    ply_element = PlyElement.describe(vertex, 'vertex')
    
    # Write the ply file
    PlyData([ply_element]).write(filename)


def load_gaussians(input_path):
    file_extension = os.path.splitext(input_path)[1]
    if file_extension == ".splat":
        return load_splat_data(input_path)
    elif file_extension == ".ply":
        return load_ply_data(input_path)
    else:
        raise AttributeError("Unsupported input type {file_extension}")
