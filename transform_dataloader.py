import os
import numpy as np
import torch

def convert_sfm_pose_to_nerf(transform):
    c2w = np.linalg.inv(transform)

    flip_mat = np.array([
                                [1, 0, 0, 0],
                                        [0, -1, 0, 0],
                                        [0, 0, -1, 0],
                                        [0, 0, 0, 1]
                                ])

    return np.matmul(c2w, flip_mat)

def qvec2rotmat(qvec):
    return np.array([
		[
			1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
			2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
			2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
		], [
			2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
			1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
			2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
		], [
			2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
			2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
			1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
		]
	])

def get_colmap_intrinsics(file_path):

    camera_intrinsics = {}

    with open(file_path, "r") as colmap_file:
         for line in colmap_file:
            line = line.strip()
            if len(line) != 0 and line[0] == "#":
                continue

            elems=line.split(" ")

            camera_id = int(elems[0])
            
            if elems[1].lower().strip() != "pinhole":
                raise AttributeError("Colmap cameras txt must be Pinhole camera type")

            camera_intrinsics[camera_id] = elems[2:]

    return camera_intrinsics

def get_colmap_transforms(input_path):
    colmap_transforms = {}
    transform_cameras = {}

    i = 0

    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    
    transform_file_path = os.path.join(input_path, "images.txt")
    intrinsics_file_path = os.path.join(input_path, "cameras.txt")

    colmap_cameras = get_colmap_intrinsics(intrinsics_file_path)

    with open(transform_file_path, "r") as colmap_file:
        for line in colmap_file:
            line = line.strip()
            if len(line) != 0 and line[0] == "#":
                continue
            i = i + 1

            if len(line) == 0:
                continue

            if  i % 2 == 1:
                elems=line.split(" ")

                name = str(elems[9])

                image_id = str(elems[0])

                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))

                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3,1])

                c2w = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                    
                c2w_flipped = convert_sfm_pose_to_nerf(c2w)
                        
                colmap_transforms[name] = c2w_flipped.tolist()
                
                camera_id = int(elems[8])

                transform_cameras[name] = colmap_cameras[camera_id]

    return colmap_transforms, transform_cameras

def load_colmap_data(input_path):
    return get_colmap_transforms(input_path)

def load_transform_json_data(input_path):
    pass

def load_transform_data(input_path):
    if os.path.isdir(input_path):
        if os.path.exists(os.path.join(input_path, "images.txt")):
            return load_colmap_data(input_path)
        else:
            raise AttributeError("Unsupported transform data type")
    else:
        file_extension = os.path.splitext(input_path)[1]
        if file_extension == ".json":
            return load_transform_json_data(input_path)
        else:
            raise AttributeError("Unsupported transform data type")



