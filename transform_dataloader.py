import os
import numpy as np
import torch
import cv2
import struct
import json

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

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def get_colmap_bin_intrinsics(file_path):
    camera_intrinsics = {}
    with open(file_path, "rb") as colmap_file:
        num_cameras = read_next_bytes(colmap_file, 8, "Q")[0]
        for _ in range(num_cameras):
            elems = read_next_bytes(
                colmap_file, num_bytes=56, format_char_sequence="iiQQdddd"
            )
            camera_id = elems[0]
            
            if elems[1] != 1:
                raise AttributeError("Colmap cameras bin must be Pinhole camera type")

            camera_intrinsics[camera_id] = elems[2:]

    return camera_intrinsics

def get_colmap_txt_intrinsics(file_path):

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

def get_colmap_img_transform(elems):

    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])

    image_id = str(elems[0])

    qvec = np.array(tuple(map(float, elems[1:5])))
    tvec = np.array(tuple(map(float, elems[5:8])))

    R = qvec2rotmat(-qvec)
    t = tvec.reshape([3,1])

    c2w = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                    
    c2w_flipped = convert_sfm_pose_to_nerf(c2w)

    return c2w_flipped.tolist()

def load_colmap_bin_data(input_path):

    colmap_transforms = {}
    transform_cameras = {}

    transform_file_path = os.path.join(input_path, "images.bin")
    intrinsics_file_path = os.path.join(input_path, "cameras.bin")

    colmap_cameras = get_colmap_bin_intrinsics(intrinsics_file_path)

    with open(transform_file_path, "rb") as colmap_file:
        num_reg_images = read_next_bytes(colmap_file, 8, "Q")[0]
        for _ in range(num_reg_images):
            elems = read_next_bytes(
                colmap_file, num_bytes=64, format_char_sequence="idddddddi"
            )

            image_id = elems[0]

            transform = get_colmap_img_transform(elems)

            camera_id = elems[8]

            binary_image_name = b""
            current_char = read_next_bytes(colmap_file, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                binary_image_name += current_char
                current_char = read_next_bytes(colmap_file, 1, "c")[0]
            name = binary_image_name.decode("utf-8")

            num_points2D = read_next_bytes(
                colmap_file, num_bytes=8, format_char_sequence="Q"
            )[0]
            x_y_id_s = read_next_bytes(
                colmap_file,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )

            colmap_transforms[name] = transform
            transform_cameras[name] = colmap_cameras[camera_id]
    

    return colmap_transforms, transform_cameras

def load_colmap_txt_data(input_path):
    colmap_transforms = {}
    transform_cameras = {}

    i = 0
    
    transform_file_path = os.path.join(input_path, "images.txt")
    intrinsics_file_path = os.path.join(input_path, "cameras.txt")

    colmap_cameras = get_colmap_txt_intrinsics(intrinsics_file_path)

    with open(transform_file_path, "r") as colmap_file:
        for line in colmap_file:
            line = line.strip()
            if len(line) != 0 and line[0] == "#":
                continue
            i = i + 1

            if len(line) == 0:
                continue

            if  i % 2 == 1:
                elems = line.split(" ")

                camera_id = int(elems[8])
                name = str(elems[9])

                transform = get_colmap_img_transform(elems)

                colmap_transforms[name] = transform
                transform_cameras[name] = colmap_cameras[camera_id]

    return colmap_transforms, transform_cameras

def get_transform_intrinsics(transforms, fname):
    intrinsics = [0, 0, 0, 0]

    intrinsics[2] = transforms["fl_x"]

    if "fl_y" in transforms.keys():
        intrinsics[3] = transforms["fl_y"] 
    else:
        # Assuming that focal lengths are same in both dimensions
        intrinsics[3] = intrinsics[2]

    if "w" in transforms and "h" in transforms:
        intrinsics[0] = transforms["w"] 
        intrinsics[1] = transforms["h"] 
    else:
        img_pixels = cv2.imread(fname)

        intrinsics[0] = img_pixels.shape[1]
        intrinsics[1] = img_pixels.shape[0]

    return intrinsics

def load_transform_json_data(input_path):
    with open(input_path, "r") as transform_file:
        transforms = json.load(transform_file)

    json_transforms = {}
    intrinsics = {}

    all_intrinsics = None 
    if "fl_x" in transforms.keys():
        all_intrinsics = get_transform_intrinsics(transforms, transforms["frames"][0]["file_path"])

    for frame in transforms["frames"]:
        fname = os.path.basename(frame["file_path"])
        transform = frame["transform_matrix"]

        if all_intrinsics is None:
            intrinsics[fname] = get_transform_intrinsics(frame, frame["file_path"])
        else:
            intrinsics[fname] = all_intrinsics

        json_transforms[fname] = transform 

    return json_transforms, intrinsics

def load_transform_data(input_path):
    if os.path.isdir(input_path):
        if os.path.exists(os.path.join(input_path, "images.txt")):
            return load_colmap_txt_data(input_path)
        if os.path.exists(os.path.join(input_path, "images.bin")):
            return load_colmap_bin_data(input_path)
        else:
            raise AttributeError("Unsupported transform data type")
    else:
        file_extension = os.path.splitext(input_path)[1]
        if file_extension == ".json":
            return load_transform_json_data(input_path)
        else:
            raise AttributeError("Unsupported transform data type")