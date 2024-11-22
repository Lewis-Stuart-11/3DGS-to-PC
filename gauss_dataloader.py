import os
import numpy as np
import torch
from tqdm import tqdm

from plyfile import PlyData, PlyElement

def computeColorFromLowDegSH(sh):
    """
    Calculates colour from first degree spherical harminics
    """
    SH_C0 = 0.28209479177387814
    
    return ((SH_C0 * sh[:,:,0].to(torch.double)) + 0.5).clip(0, 1).type(torch.double) 

def load_ply_data(path, max_sh_degree=3):
    """
    Loads in Gaussians from .ply file
    """
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

    features_all = torch.cat((torch.tensor(features_dc, device="cuda:0"), torch.tensor(features_extra, device="cuda:0")), 2)

    colours = computeColorFromLowDegSH(features_all)

    opacities = (1 / (1 + torch.exp(torch.tensor(-opacities, device="cuda:0")))).type(torch.float).squeeze(1) # torch.full(opacities.shape, 0.05, device="cuda:0")-

    xyz_tensor = torch.tensor(xyz, device="cuda:0")
    scales_tensor = torch.tensor(scales, device="cuda:0")
    rots_tensor = torch.tensor(rots/np.expand_dims(np.linalg.norm(rots, axis=1), 1), device="cuda:0") 

    return xyz_tensor, scales_tensor, rots_tensor, colours, opacities

def load_splat_data(path):
    """
    Loads in Gaussians from .splat file
    """

    with open(path, "rb") as test_file:
        file_content = test_file.read()
    
    dtype = np.dtype([
        ('xyz', np.float32, 3), 
        ('scales', np.float32, 3),  
        ('colour', np.uint8, 4),  
        ('rots', np.uint8, 4)
    ])
    
    num_structures = len(file_content) // dtype.itemsize
    
    data_array = np.frombuffer(file_content, dtype=dtype, count=num_structures)

    xyz = data_array['xyz']
    scales = data_array['scales']
    colours = data_array['colour']
    rots = data_array['rots']
    
    xyz_tensor = torch.tensor(xyz, device="cuda:0")
    scales_tensor = torch.tensor(np.log(scales), device="cuda:0")
    colours_tensor = torch.tensor(colours[:, :3] / 255, device="cuda:0")
    opacities = torch.tensor(colours[:, 3] / 255, device="cuda:0")
    
    rots_tensor = torch.tensor((rots.astype(np.float32) - 128) / 128, device="cuda:0")
    
    return xyz_tensor, scales_tensor, rots_tensor, colours_tensor, opacities

    
def save_xyz_to_ply(xyz_points, filename, rgb_colors=None, normals_points=None, chunk_size=10**6):
    """
    Save a series of XYZ points to a PLY file.

    Args:
        xyz_points: An array of shape (N, 3) containing XYZ coordinates.
        filename: The name of the output PLY file.
        rgb_colors: An array of shape (N, 3) containing RGB colors. Defaults to white.
        chunk_size: Sizes of chunks that will be saved iteratively (reduces chances of out of memory errors)
    """

    # Ensure the points are in the correct format
    assert xyz_points.shape[1] == 3, "Input points should be in the format (N, 3)"

    if rgb_colors is None:
        rgb_colors = tensor.full((xyz_points.shape[0], 3), 255, dtype=torch.int)

    total_points = xyz_points.shape[0]

    num_chunks = (total_points + chunk_size - 1) // chunk_size

    with open(filename, 'wb') as ply_file:
    
        if normals_points is not None:
            header = f"""ply
format binary_little_endian 1.0
element vertex {total_points}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
end_header
"""
        else:
            # Write PLY header
            header = f"""ply
format binary_little_endian 1.0
element vertex {total_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
""" 

        ply_file.write(header.encode('utf-8'))

        for i in tqdm(range(num_chunks), position=0, leave=True):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total_points)

            points_chunk = xyz_points[start_idx:end_idx].cpu().detach().numpy()
            colors_chunk = rgb_colors[start_idx:end_idx].cpu().detach().numpy().astype(np.uint8)

            if normals_points is not None:
                normals_chunk = normals_points[start_idx:end_idx].cpu().detach().numpy()

                # Create a structured array directly
                vertex = np.zeros(points_chunk.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                                                        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                                                        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

                vertex['nx'] = normals_chunk[:, 0]
                vertex['ny'] = normals_chunk[:, 1]
                vertex['nz'] = normals_chunk[:, 2]
            else:
                # Create a structured array directly
                vertex = np.zeros(points_chunk.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                                                        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

            vertex['x'] = points_chunk[:, 0]
            vertex['y'] = points_chunk[:, 1]
            vertex['z'] = points_chunk[:, 2]
            vertex['red'] = colors_chunk[:, 0]
            vertex['green'] = colors_chunk[:, 1]
            vertex['blue'] = colors_chunk[:, 2]

            ply_file.write(vertex.tobytes())

def load_gaussians(input_path, max_sh_degree=3):
    file_extension = os.path.splitext(input_path)[1]
    if file_extension == ".splat":
        return load_splat_data(input_path)
    elif file_extension == ".ply":
        return load_ply_data(input_path, max_sh_degree=max_sh_degree)
    else:
        raise AttributeError("Unsupported input type {file_extension}")
