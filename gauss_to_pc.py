import numpy as np
import torch
import sys
import configargparse
import gc
import time
import imageio
#import matplotlib.pyplot as plt

from tqdm import tqdm
from math import cos, sin, pi, sqrt, tan, ceil, exp, floor
from torch.distributions.multivariate_normal import MultivariateNormal
from typing import NamedTuple

from gauss_handler import Gaussians

from transform_dataloader import load_transform_data
from gauss_dataloader import load_gaussians, save_xyz_to_ply

from gauss_render import GaussRenderer
from camera_handler import Camera

COLOR_QUALITY_OPTIONS = {"tiny": 180, "low": 360, "medium": 720, "high": 1280, "ultra": 1920}

class GaussPointCloudSettings(NamedTuple):
    """
    Properties:
        num_points: the number of points to generate for the point cloud
        std_distance: the max distance that a sampled point can be from its gaussian centre before being rejected
        render_colours: whether to render point colours or to use the given gaussian colours (rendering is recommended)
        min_opacity: filter gaussians with an opacity less than the min_opacity
        bounding_box_min: minimum values of the bounding box 
        bounding_box_max: maximum values of the bounding box 
        cull_large_percentage: percentage of the gaussians to remove (from largest to smallest)
        remove_unrendered_gaussians: gaussians that were not rendered, based on the cameras from the transform_path, are not included in the point cloud
        colour_resolution: resolution of the image to render for determining the point colours
        exact_num_points: number of attempts to generate all points for gaussians with the same number of assigned points 
        device: torch device
    """
    num_points: int
    std_distance: float
    camera_skip_rate: int
    render_colours: bool 
    min_opacity: float
    bounding_box_min: list
    bounding_box_max: list
    calculate_normals: bool
    cull_large_percentage: float
    remove_unrendered_gaussians: bool
    colour_resolution: int
    max_sh_degree: int
    exact_num_points: int
    generate_mesh: bool
    device: str

class PointCloudData(NamedTuple):
    points: torch.Tensor
    colours: torch.Tensor
    normals: torch.Tensor

def imwrite(path, image):
    """
    Saves image of the rendered gaussians
    """
    imageio.imwrite(path, ((255 * np.clip(image, 0, 1)).astype(np.uint8)))

def distribute_points(gaussian_sizes, num_points):
    """
    Calculates the number of points per gaussian based on the total gaussian sizes
    """

    total_sum = torch.sum(gaussian_sizes)

    points_ratio = num_points/total_sum

    # Distributes points per gaussian and rounds
    points_per_gaussian = torch.round(gaussian_sizes * points_ratio)
    
    # Increase the number of points to generate to match the given number of points by setting gaussians with 0 points to 1 
    zero_indices = (points_per_gaussian == 0).nonzero()
    zero_indices = zero_indices[:int(min((num_points-points_per_gaussian.sum()).item(), points_per_gaussian[points_per_gaussian == 0].shape[0]))]
    points_per_gaussian[zero_indices] = 1

    return points_per_gaussian

def mahalanobis(means, samples, covs):
    """
    Calculates the mahalanobis distance between a batches of points and the original gaussian centre
    """
    delta = means - samples
    delta = torch.unsqueeze(delta, 2)

    conv_inv = torch.inverse(covs)
    mm_cov_delta = torch.bmm(conv_inv, delta)
    m = torch.bmm(torch.transpose(delta, 1, 2), mm_cov_delta)

    return torch.sqrt(m).squeeze(1).squeeze(1)

def calculate_bin_sizes(points_per_gaussian):
    """
    Crude algorithm for determining when to batch gaussians with similar number of points together
    """

    distribution = torch.bincount(points_per_gaussian)

    distribution = distribution[distribution.nonzero()].squeeze(1)

    # Second derivative of the number of points per gaussian 
    gradients = np.absolute(np.gradient(np.gradient(distribution.cpu().detach().numpy())))

    bin_size = max(len(distribution)//100, 1)

    length = len(gradients) - len(gradients) % bin_size
    gradients = gradients[:length]

    reshaped_gradients = gradients.reshape(-1, bin_size)
        
    summed_gradients = reshaped_gradients.sum(axis=1)

    cut_off = np.max(summed_gradients)//50
    peak = np.argmax(summed_gradients)

    binned_sums = np.nonzero(summed_gradients[peak:] < cut_off)[0]

    start_bin = 1
    if binned_sums.shape[0] != 0:
        start_bin = np.nonzero(summed_gradients[peak:] < cut_off)[0][0]

    #plt.plot(np.linspace(0, summed_arr.shape[0], num=summed_arr.shape[0], dtype=int), summed_arr)
    #plt.show()

    return start_bin, bin_size

def create_new_gaussian_points(num_points_to_sample, means, covariances, colours, std_distance=2, num_attempts=5, normals=None, device="cuda:0"):
    """
    Generates new points for each of the provided gaussians

    Args:
        num_points_to_sample: the number of points to assign to each gaussian
        means: centre of each gaussian
        covariances: covarainces of each gaussian
        colours: colours of each gaussian
        std_distance: the max distance that a sampled point can be from its gaussian centre before being rejected
        num_attempts: number of times to resample each gaussian with number of points less than the num_points_to_sample
        device: torch device

    Returns:
        new_points: sampled 3D points over all gaussians
        new_colours: colours of the 3D points
    """
    
    # Number of points that should be generated over all gaussians
    total_required_points = num_points_to_sample * means.shape[0]

    # Tracks the number of points that have been added for each gaussian
    # Ensures that the exact number of required points is added for each gaussian
    added_points = torch.zeros(means.shape[0], device=device)

    # The maximum number of points that can be added for each gaussian
    max_count = torch.full((means.shape[0],), num_points_to_sample, device=device)
        
    new_points = torch.tensor([], device=device)
    new_colours = torch.tensor([], device=device).type(torch.double)
    new_normals =  torch.tensor([], device=device).type(torch.double) if normals is not None else None

    i = 0

    # Loop until all required points have been sampled or the maximium number of attempts has been exceeded
    while new_points.shape[0] < total_required_points and i < num_attempts:

        # Get gaussians which do not curretly have the maximum number of points to be sampled from
        gaussians_to_add = added_points != num_points_to_sample  
        new_means_for_point = means[gaussians_to_add]
        new_covariances_for_point = covariances[gaussians_to_add]
        new_colours_for_point = colours[gaussians_to_add]
        new_normals_for_point = normals[gaussians_to_add] if normals is not None else None

        gaussians_to_add_idxs = gaussians_to_add.nonzero().squeeze(1)

        # Sample 'num_points_to_sample' number of points for each gaussian
        original_sampled_points = MultivariateNormal(new_means_for_point, new_covariances_for_point).sample((num_points_to_sample,))

        sampled_points = original_sampled_points.transpose(0, 1).contiguous().view(-1, original_sampled_points.size(2))

        repeated_means = torch.repeat_interleave(new_means_for_point, num_points_to_sample, dim=0)
        repeated_covariances = torch.repeat_interleave(new_covariances_for_point, num_points_to_sample, dim=0)

        # Get the mahalanobis distance between the point and its centre gaussian        
        mahalanobis_distances = mahalanobis(repeated_means, sampled_points, repeated_covariances)
        
        # Filter out points with a distance less than the set std_distance
        filtered_samples_idxs = mahalanobis_distances <= std_distance
        filtered_samples = sampled_points[filtered_samples_idxs]

        # Count the number of points per gaussian that have not been rejected
        grouped_idxs = torch.arange(sampled_points.shape[0], device=device)[filtered_samples_idxs].type(torch.float)
        grouped_idxs = torch.floor(torch.div(grouped_idxs, num_points_to_sample))
        counted_idxs, counts = torch.unique(grouped_idxs, return_counts=True)

        # Add gaussians that have not been sampled, due to already having enough points, to the count with a value of 0
        # This ensures that the tensors of the added points and counts are the same size
        all_idxs = torch.arange(new_means_for_point.shape[0], device=device)
        zeroed_indxs = all_idxs[~torch.isin(all_idxs, counted_idxs.type(torch.int))]
        for element in zeroed_indxs:
            counts = torch.cat((counts[:element], torch.tensor([0], device=device), counts[element:]))

        # Occasionally counts will return larger than the size of gaussians to add
        # for a large number of generated points (assuming overflow error)
        counts = counts[:gaussians_to_add_idxs.shape[0]] 

        # Get the difference between the number of points to add (max number of points per gaussian - number of valid points that have been generated)
        # And the number that have been generated. This is the number of points that will be added 
        diffs = torch.min(max_count[gaussians_to_add_idxs]-added_points[gaussians_to_add_idxs], counts).type(torch.int)

        total_current_points = int(diffs.sum().item())

        # Filter points so that only the exact amount of required points (diffs) are added for each gaussian
        indices = torch.arange(len(diffs), device=device) * num_points_to_sample
        expanded_indices = indices.unsqueeze(1) + torch.arange(num_points_to_sample, device=device)
        expanded_indices = expanded_indices.flatten()
        mask = (torch.arange(num_points_to_sample, device=device).unsqueeze(0) < diffs.unsqueeze(1)).flatten()
        filtered_indices = expanded_indices[mask]

        # Empty tensors that will be added to
        current_points = torch.empty((total_current_points, sampled_points.size(1)), dtype=sampled_points.dtype, device=device)
        current_colours = torch.empty((total_current_points, new_colours_for_point.size(1)), dtype=new_colours_for_point.dtype, device=device)

        # Add new sampled points 
        current_points[:] = sampled_points[filtered_indices]
        current_colours[:] = new_colours_for_point.repeat_interleave(diffs, dim=0)

        # Update added points with the new count (and ensure it is not bigger than the number of points to sample)
        added_points[gaussians_to_add_idxs] += counts
        added_points = torch.where(added_points > num_points_to_sample, num_points_to_sample, added_points).type(torch.int)

        new_points = torch.cat((new_points, current_points), 0)
        new_colours = torch.cat((new_colours, current_colours), 0)

        if normals is not None:
            current_normals = torch.empty((total_current_points, new_normals_for_point.size(1)), dtype=new_normals_for_point.dtype, device=device)
            current_normals[:] = new_normals_for_point.repeat_interleave(diffs, dim=0) 
            new_normals = torch.cat((new_normals, current_normals), 0)

        i += 1

    return new_points, new_colours, new_normals

def generate_pointcloud(gaussians, num_points, std_distance=2, exact_num_points=False, calculate_normals=True, num_sample_attempts=5, device="cuda:0"):

    """
    Generates a pointcloud from a set of gaussians

    Args:
        gaussians: A Gaussian object containing the list of gaussians to use for the point cloud
        num_points: The number of point to generate for the point cloud
        std_distance: Max distance a point can be generated from its Gaussian centre
        exact_num_points: Creates the exact number of required points 
        calculate_normals: Set to calculate normals for each point
        num_sample_attempts: The number of attempts to generate the number of points per bin
        device: torch device

    Returns:
        total_points: sampled 3D points over all gaussians
        total_colours: colours of the 3D points
        total_normals: calculated normals for each of the points
    """

    # Calculate Gaussian sizes
    gaussian_sizes = gaussians.get_gaussian_sizes()

    print(f"Distributed Points to Gaussians")
    print()

    # Assign points to gaussians 
    points_per_gaussian = distribute_points(gaussian_sizes, num_points).type(torch.int)

    point_distribution = torch.unique(points_per_gaussian)

    # Bin gaussians together (makes the generation process much faster)
    if not exact_num_points:
        start_bin, bin_size = calculate_bin_sizes(points_per_gaussian)

        point_distribution = torch.cat((point_distribution[:start_bin],  torch.mul(torch.unique(torch.ceil(point_distribution[start_bin:]/ bin_size)), bin_size)), 0)

    total_points = torch.tensor([], device=device)
    total_colours = torch.tensor([], device=device).type(torch.double)
    total_normals = torch.tensor([], device=device).type(torch.double) if calculate_normals else None 

    print(f"Starting Point Cloud Generation")

    # Iterate through different number of points 
    for i in tqdm(range(point_distribution.shape[0]), position=0, leave=True):

        start_range = point_distribution[i]

        if i != point_distribution.shape[0]-1:
            end_range = point_distribution[i+1]
        else:
            end_range = start_range+1

        # Get gaussians with assigned number of points between the start and end
        gaussian_indices = torch.where((points_per_gaussian >= start_range) & (points_per_gaussian < end_range))[0]

        # Number of points to generate for that gaussian
        num_points_for_gaussian = floor(start_range + (end_range-start_range)/2)

        if num_points_for_gaussian <= 0:
            continue

        if gaussian_indices.shape[0] < 1:
            continue

        # All gaussians which have the number of assigned points between the start and end range 
        covariances_for_point = gaussians.covariances[gaussian_indices]
        mean_for_point = gaussians.xyz[gaussian_indices]
        centre_colours = gaussians.colours[gaussian_indices]
        normals_for_point = gaussians.normals[gaussian_indices] if calculate_normals else None
        
        # First point to use is the centre of the gaussian 
        total_points = torch.cat((total_points, mean_for_point), 0)
        total_colours = torch.cat((total_colours, centre_colours), 0)
        if calculate_normals:
            total_normals = torch.cat((total_normals, normals_for_point), 0)

        if num_points_for_gaussian <= 1: 
            continue

        # Sample the rest of the required points
        new_points, new_colours, new_normals = create_new_gaussian_points(num_points_for_gaussian-1, mean_for_point, covariances_for_point,
                                                             centre_colours, std_distance=std_distance, num_attempts=num_sample_attempts, 
                                                             normals=normals_for_point, device=device)

        total_points = torch.cat((total_points, new_points), 0)
        total_colours = torch.cat((total_colours, new_colours), 0)
 
        if calculate_normals:
            total_normals = torch.cat((total_normals, new_normals), 0)

    return total_points, total_colours, total_normals


def convert_3dgs_to_pc(input_path, transform_path, pointcloud_settings):
    """
    Generates a pointcloud from a 3DGS file

    Args:
        input_path: the path to the file containing the gaussian data
        transform_path: the path to the file containing transform data for rendering colours
        pointcloud_settings: contains all configuration information for generating the point cloud

    Returns:
        total_point_cloud: a PointCloudData object containing points for the point cloud generated on the entire scene
        total_colours: a PointCloudData object containing points for the point cloud generated on the surfaces of the scene
    """

    # Transform path has been provided, so use those camera positions and intrinsics 
    if transform_path is not None:
        print("Loading Camera Poses")
        print()

        transforms, intrinsics = load_transform_data(transform_path, skip_rate=pointcloud_settings.camera_skip_rate)

    print("Loading Gaussians from File")
    print()
    
    # Load gaussian data from file
    xyz, scales, rots, colours, opacities = load_gaussians(input_path, max_sh_degree=pointcloud_settings.max_sh_degree)

    gaussians = Gaussians(xyz, scales, rots, colours, opacities)

    # Calculate Gaussian Normals
    if pointcloud_settings.calculate_normals:
        gaussians.calculate_normals()

    # Format gaussians
    gaussians.apply_min_opacity(pointcloud_settings.min_opacity)
    gaussians.apply_bounding_box(pointcloud_settings.bounding_box_min, pointcloud_settings.bounding_box_max)
    gaussians.cull_large_gaussians(pointcloud_settings.cull_large_percentage)

    # Rendered colours has been set 
    if pointcloud_settings.render_colours:

        print("Rendering Gaussian Colours")

        # Initialise the gaussian renderer
        gaussian_renderer = GaussRenderer(gaussians.xyz, torch.unsqueeze(torch.clone(gaussians.opacities), 1).type(torch.float),
                                              gaussians.colours, gaussians.covariances)
        
        if transform_path is not None:

            # Render colours for each camera
            for i in tqdm(range(len(transforms)), position=0, leave=True):

                img_name, transform = list(transforms.items())[i]

                transform = torch.tensor(list(transform), device=pointcloud_settings.device)

                cam_intrinsic = intrinsics[img_name]

                cam_matrix = torch.eye(4, device=pointcloud_settings.device) 

                # Rescale cam parameters to match the given render resolution
                diff = pointcloud_settings.colour_resolution / int(cam_intrinsic[0])

                img_width = int(int(cam_intrinsic[0]) * diff) 
                img_height = int(int(cam_intrinsic[1]) * diff) 

                focal_x = float(cam_intrinsic[2])*diff
                focal_y = float(cam_intrinsic[3])*diff

                camera = Camera(img_width, img_height, focal_x, focal_y, transform)

                # Render a new image
                render = gaussian_renderer.add_img(camera).detach().cpu().numpy()

                #imwrite(f"results\\{i}.png", render)
        else:
            raise Exception("Transforms are required to render colours")   

        # Get new rendered gaussian colours
        gaussians.colours = gaussian_renderer.get_colours()

        # Remove gaussians that were not rendered at all
        if pointcloud_settings.remove_unrendered_gaussians:
            gaussians.filter_gaussians(gaussian_renderer.get_seen_gaussians())

        # Get the predicted surface Gaussians
        if pointcloud_settings.generate_mesh:
            surface_gaussian_idxs = gaussian_renderer.get_surface_gaussians()[gaussian_renderer.get_seen_gaussians()] 

        del gaussian_renderer
    
    else:

        # Convert colours from (0-1) to (0-255)
        gaussians.colours *= 255

        print("Skipping Rendering Gaussian Colours")
    
    print()

    # Number of attempts per point number generation
    num_sample_attempts = 5 if not pointcloud_settings.exact_num_points else 100

    print("Starting Point Cloud Generation for All Gaussians")
    print()

    # Generate points for the point cloud of the entire scene
    points, colours, normals = generate_pointcloud(gaussians, pointcloud_settings.num_points, exact_num_points=pointcloud_settings.exact_num_points, 
                                                                                              std_distance = pointcloud_settings.std_distance, 
                                                                                              device=pointcloud_settings.device, 
                                                                                              calculate_normals=pointcloud_settings.calculate_normals,
                                                                                              num_sample_attempts=num_sample_attempts)

    total_point_cloud = PointCloudData(
        points = points,
        colours = colours,
        normals = normals
    )

    print()

    surface_point_cloud = None

    # Generate surface point cloud if meshing the scene
    if pointcloud_settings.generate_mesh and pointcloud_settings.render_colours:
        print("Starting Point Cloud Generation for Surface Gaussians")
        print()

        # Ensure that only surface gaussians are now included
        gaussians.filter_gaussians(surface_gaussian_idxs)

        avg_points_per_gauss_for_mesh = 25

        # Set the number of mesh points as the large of the number of surface Gaussians multiplied by the average points per Gaussian
        # Or half the number of points set by the user 
        total_mesh_points = min(pointcloud_settings.num_points//2, int(gaussians.xyz.shape[0]*avg_points_per_gauss_for_mesh))

        # Generate points for the point cloud of the entire scene
        points, colours, normals = generate_pointcloud(gaussians, total_mesh_points, exact_num_points=pointcloud_settings.exact_num_points, 
                                                                                     num_sample_attempts=num_sample_attempts,
                                                                                     device=pointcloud_settings.device)

        surface_point_cloud = PointCloudData(
            points = points,
            colours = colours,
            normals = normals
        )

        print()

    # Clear memory 
    torch.cuda.empty_cache()  
    gc.collect()  

    return total_point_cloud, surface_point_cloud

def config_parser():

    parser = configargparse.ArgumentParser()

    parser.add_argument("--input_path",  type=str, required=True, help="Path to ply or splat file to convert to a point cloud")
    parser.add_argument("--output_path",  type=str, default="3dgs_pc.ply", help="Path to output file (must be ply file)")
    parser.add_argument("--transform_path", default=None, type=str, help="Path to COLMAP or Transform file used for loading in camera positions for rendering")

    parser.add_argument("--num_points", type=int, default=10000000, help="Total number of points to generate for the pointcloud")
    parser.add_argument("--exact_num_points", action="store_true", help="Set if the number of generated points should more closely match the num_points argument (slower)")
    
    parser.add_argument("--generate_mesh", action="store_true", help="Set to also generate a mesh based on the created point cloud (requires Open3D)")
    parser.add_argument("--poisson_depth", default=10, type=int, help="The depth used in the poisson surface reconstruction algorithm that is used for meshing (larger value = more quality) ")
    parser.add_argument("--laplacian_iterations", default=10, type=int, help="The number of iterations to perform laplacian mesh smoothing (higher value = smoother mesh)")
    parser.add_argument("--mesh_output_path",  type=str, default="3dgs_mesh.ply", help="Path to mesh output file (must be ply file)")

    parser.add_argument("--clean_pointcloud", action="store_true", help="Set to remove outliers on the point cloud after generation (requires Open3D)")

    parser.add_argument("--camera_skip_rate", type=int, default=0, help="Number of cameras to skip for each rendered camera (reduces compute time- only use if cameras in linear trajectory)")
    
    parser.add_argument("--no_render_colours", action="store_true", help="Skip rendering colours- faster but colours will be strange")
    parser.add_argument("--colour_quality", type=str, default="medium", help="The quality of the colours when generating the point cloud (more quality = slower processing time)")

    parser.add_argument("--bounding_box_min", nargs=3, help="Values for minimum position of gaussians to include in generating the new point cloud")
    parser.add_argument("--bounding_box_max", nargs=3, help="Values for maximum position of gaussians to include in generating the new point cloud")

    parser.add_argument("--std_distance", type=float, default=2.0, help="Maximum distance each point can be from the centre of their gaussian")

    parser.add_argument("--no_calculate_normals",  action="store_true", help="Set to not calculate normals for the points")

    parser.add_argument("--min_opacity", type=float, default=0.0, help="Minimum opacity for gaussians to be included (must be between 0-1)")

    parser.add_argument("--cull_gaussian_sizes", type=float, default=0.0, help="The percentage of gaussians to remove from largest to smallest (used to remove large gaussians)")

    parser.add_argument("--max_sh_degree", type=int, default=3, help="The number spherical harmonics of the loaded point cloud (default 3- change if different number of spherical harmonics are loaded)")

    args = parser.parse_args()

    if args.min_opacity < 0 or args.min_opacity > 1:
        raise AttributeError("Minumum opacity must be between 0 and 1")

    if args.std_distance <= 0:
        raise AttributeError("Std distance must be greater than 0")

    if args.num_points <= 0:
        raise AttributeError("Number of points must be greater than 0")
    
    if args.bounding_box_min is not None:
        try:
            args.bounding_box_min = [float(x) for x in args.bounding_box_min]
        except ValueError:
            raise AttributeError("Bounding Box Min must contain float values")

        if len(args.bounding_box_min) != 3:
            raise AttributeError("Bounding Box Min must have exactly 3 values")

    if args.bounding_box_max is not None:
        try:
            args.bounding_box_max = [float(x) for x in args.bounding_box_max]
        except ValueError:
            raise AttributeError("Bounding Box Max must contain float values")

        if len(args.bounding_box_max) != 3:
            raise AttributeError("Bounding Box Max must have exactly 3 values")

    if args.colour_quality.lower() not in COLOR_QUALITY_OPTIONS.keys():
         raise AttributeError(f"Colour quality must be in the following options {COLOR_QUALITY_OPTIONS.keys()}")

    if args.max_sh_degree < 0:
        raise AttributeError(f"The number of spherical harmonics must be larger than 0")

    if args.camera_skip_rate < 0:
        raise AttributeError(f"The camera skip rate must be larger than 0")

    if args.generate_mesh and args.no_calculate_normals:
        raise AttributeError(f"Normals are required for accurate meshing")

    if args.generate_mesh and args.no_render_colours:
        raise AttributeError(f"Colours are required for meshing")

    if args.generate_mesh and args.transform_path is None:
        raise AttributeError(f"Transforms are required for meshing")

    if not args.no_render_colours and args.transform_path is None:
        raise AttributeError(f"Transforms are required for rendering accurate point colours, set --no_render_colours to True to render with no colour")

    return args

def main():
    args = config_parser()

    # All config info required for generating the point cloud
    pointcloud_settings = GaussPointCloudSettings(
        num_points=args.num_points,
        std_distance=args.std_distance,
        camera_skip_rate=args.camera_skip_rate,
        render_colours=not args.no_render_colours,
        min_opacity=args.min_opacity,
        bounding_box_min=args.bounding_box_min, 
        bounding_box_max=args.bounding_box_max,
        calculate_normals=not args.no_calculate_normals,
        cull_large_percentage=args.cull_gaussian_sizes, 
        colour_resolution=int(COLOR_QUALITY_OPTIONS[args.colour_quality.lower()]),
        max_sh_degree=args.max_sh_degree, 
        exact_num_points = args.exact_num_points,
        generate_mesh = args.generate_mesh,
        remove_unrendered_gaussians = True,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    # Generate point cloud from 3DGS scene
    total_point_cloud, surface_point_cloud = convert_3dgs_to_pc(args.input_path, args.transform_path, pointcloud_settings)
    
    # Clean point cloud if set
    if args.clean_pointcloud:
        print("Cleaning Point Cloud")

        from mesh_handler import clean_point_cloud

        # Clean point cloud using Open3D
        cleaned_points, cleaned_colours, cleaned_normals = clean_point_cloud(total_point_cloud.points, total_point_cloud.colours, 
                                                                             total_point_cloud.normals, device=pointcloud_settings.device)

        total_point_cloud = PointCloudData(
            points = cleaned_points,
            colours = cleaned_colours,
            normals = cleaned_normals
        )

    print("Saving Final Point Cloud")

    # Save point cloud
    save_xyz_to_ply(total_point_cloud.points, args.output_path, rgb_colors=total_point_cloud.colours, 
                                                                normals_points=total_point_cloud.normals, chunk_size=10**6)

    """save_xyz_to_ply(surface_point_cloud.points, "surface_points.ply", rgb_colors=surface_point_cloud.colours,
                                                                normals_points=surface_point_cloud.normals, chunk_size=10**6)"""

    print()

    # Generate mesh from surface point cloud
    if pointcloud_settings.generate_mesh:
        print("Generating Mesh")

        from mesh_handler import generate_mesh

        # Generate and save mesh using Open3D
        generate_mesh(surface_point_cloud.points, surface_point_cloud.colours, surface_point_cloud.normals, args.mesh_output_path, 
                      depth=args.poisson_depth, laplacian_iters=args.laplacian_iterations)

    print()

if __name__ == "__main__":
    main()
