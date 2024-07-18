import numpy as np
import torch
import sys
import configargparse
import gc
import time
import imageio
#import matplotlib.pyplot as plt

from scipy.stats import norm
from math import cos, sin, pi, sqrt, tan, ceil, exp, floor
from torch.distributions.multivariate_normal import MultivariateNormal

from gauss_handler import Gaussian

from transform_dataloader import load_transform_data
from gauss_dataloader import load_gaussians, save_xyz_to_ply

from gauss_render import GaussRenderer
from camera_handler import Camera

COLOR_QUALITY_OPTIONS = {"low": 360, "medium": 720, "high": 1280, "ultra": 1920}

def distribute_points(gaussian_sizes, num_points):
    total_sum = torch.sum(gaussian_sizes)

    points_ratio = num_points/total_sum

    points_per_gaussian = torch.round(gaussian_sizes * points_ratio)

    return points_per_gaussian

def mahalanobis(means, samples, covs):
    delta = means - samples
    delta = torch.unsqueeze(delta, 2)

    conv_inv = torch.inverse(covs)
    mm_cov_delta = torch.bmm(conv_inv, delta)
    m = torch.bmm(torch.transpose(delta, 1, 2), mm_cov_delta)

    return torch.sqrt(m).squeeze(1).squeeze(1)

def calculate_bin_sizes(points_per_gaussian):
    distribution = torch.bincount(points_per_gaussian)

    distribution = distribution[distribution.nonzero()].squeeze(1)

    gradients = np.absolute(np.gradient(np.gradient(distribution.cpu().detach().numpy())))

    bin_size = max(len(distribution)//100, 2)

    length = len(gradients) - len(gradients) % bin_size
    gradients = gradients[:length]

    reshaped_arr = gradients.reshape(-1, bin_size)
        
    summed_arr = reshaped_arr.sum(axis=1)

    cut_off = np.max(summed_arr)//50
    peak = np.argmax(summed_arr)

    binned_sums = np.nonzero(summed_arr[peak:] < cut_off)[0]

    start_bin = 1
    if binned_sums.shape[0] != 0:
        start_bin = np.nonzero(summed_arr[peak:] < cut_off)[0][0]

    #plt.plot(np.linspace(0, summed_arr.shape[0], num=summed_arr.shape[0], dtype=int), summed_arr)
    #plt.show()

    return start_bin, bin_size

def create_new_gaussian_points(num_points_to_sample, means, covariances, centre_colours, std_distance=2, num_attempts=5, device="cuda:0"):
    total_required_points = num_points_to_sample * means.shape[0]

    added_points = torch.zeros(means.shape[0], device=device)
    max_count = torch.full((means.shape[0],), num_points_to_sample, device=device)
        
    new_points = torch.tensor([], device=device)
    new_colours = torch.tensor([], device=device).type(torch.double)

    i = 0

    while new_points.shape[0] < total_required_points and i < num_attempts:

        gaussians_to_add = added_points != num_points_to_sample
            
        new_means_for_point = means[gaussians_to_add]
        new_covariances_for_point = covariances[gaussians_to_add]
        new_colours_for_point = centre_colours[gaussians_to_add]

        gaussians_to_add_idxs = gaussians_to_add.nonzero().squeeze(1)

        original_sampled_points = MultivariateNormal(new_means_for_point, new_covariances_for_point).sample((num_points_to_sample,))

        sampled_points = original_sampled_points.transpose(0, 1).contiguous().view(-1, original_sampled_points.size(2))

        repeated_means = torch.repeat_interleave(new_means_for_point, num_points_to_sample, dim=0)
        repeated_covariances = torch.repeat_interleave(new_covariances_for_point, num_points_to_sample, dim=0)
            
        mahalanobis_distances = mahalanobis(repeated_means, sampled_points, repeated_covariances)
            
        filtered_samples_idxs = mahalanobis_distances <= std_distance

        filtered_samples = sampled_points[filtered_samples_idxs]

        grouped_idxs = torch.arange(sampled_points.shape[0], device=device)[filtered_samples_idxs].type(torch.float)

        grouped_idxs = torch.floor(torch.div(grouped_idxs, num_points_to_sample))
            
        counted_idxs, counts = torch.unique(grouped_idxs, return_counts=True)

        all_idxs = torch.arange(new_means_for_point.shape[0], device=device)

        zeroed_indxs = all_idxs[~torch.isin(all_idxs, counted_idxs.type(torch.int))]

        for element in zeroed_indxs:
            counts = torch.cat((counts[:element], torch.tensor([0], device=device), counts[element:]))

        diffs = torch.min(max_count[gaussians_to_add_idxs]-added_points[gaussians_to_add_idxs], counts).type(torch.int)

        total_current_points = int(diffs.sum().item())

        current_points = torch.empty((total_current_points, sampled_points.size(1)), dtype=sampled_points.dtype, device=device)
        current_colours = torch.empty((total_current_points, new_colours_for_point.size(1)), dtype=new_colours_for_point.dtype, device=device)

        indices = torch.arange(len(diffs), device=device) * num_points_to_sample

        expanded_indices = indices.unsqueeze(1) + torch.arange(num_points_to_sample, device=device)
        expanded_indices = expanded_indices.flatten()

        mask = (torch.arange(num_points_to_sample, device=device).unsqueeze(0) < diffs.unsqueeze(1)).flatten()
        filtered_indices = expanded_indices[mask]

        current_points[:] = sampled_points[filtered_indices]
        current_colours[:] = new_colours_for_point.repeat_interleave(diffs, dim=0)

        added_points[gaussians_to_add_idxs] += counts
        added_points = torch.where(added_points > num_points_to_sample, num_points_to_sample, added_points).type(torch.int)

        new_points = torch.cat((new_points, current_points), 0)
        new_colours = torch.cat((new_colours, current_colours), 0)

        i += 1

    return new_points, new_colours

def imwrite(path, image):
    imageio.imwrite(path, ((255 * np.clip(image, 0, 1)).astype(np.uint8)))

def generate_pointcloud(input_path, num_points, std_distance=2, render_colours=True, transform_path=None,
                        min_opacity=0.0, bounding_box_min=None, bounding_box_max=None, transform_data_path=None, 
                        cull_large_percentage=0.0, remove_unrendered_gaussians=True, colour_resolution=1920, device="cuda:0"):
    
    xyz, scales, rots, colours, opacities = load_gaussians(input_path)

    gaussians = Gaussian(xyz, scales, rots, colours, opacities)

    gaussians.apply_min_opacity(min_opacity)
    gaussians.apply_bounding_box(bounding_box_min, bounding_box_max)
    gaussians.cull_large_gaussians(cull_large_percentage)

    if render_colours:
        print("Rendering Gaussian Colours")

        gaussian_renderer = GaussRenderer(gaussians.xyz, torch.unsqueeze(torch.clone(gaussians.opacities), 1).type(torch.float),
                                              gaussians.colours, gaussians.covariances)
        
        if transform_path is not None:

            print("Rendering colours from transform set")
        
            transforms, intrinsics = load_transform_data(transform_path)

            for i, (img_name, transform) in enumerate(transforms.items()):

                print(i)

                transform = torch.tensor(list(transform), device=device)

                cam_intrinsic = intrinsics[img_name]

                cam_matrix = torch.eye(4, device=device) 

                diff = colour_resolution / int(cam_intrinsic[0])

                img_width = int(int(cam_intrinsic[0]) * diff) 
                img_height = int(int(cam_intrinsic[1]) * diff) 

                focal_x = float(cam_intrinsic[2])*diff
                focal_y = float(cam_intrinsic[3])*diff

                camera = Camera(img_width, img_height, focal_x, focal_y, transform)

                #start_time = time.time()
                render = gaussian_renderer.add_img(camera).detach().cpu().numpy()
                #print(f"time: {time.time() - start_time}")

                #imwrite(f"results\\{i}.png", render)

                print()
        else:
            print("Transforms are required to render colours")

            exit(0)    

        gaussians.gaussian_colours = gaussian_renderer.get_colours()

        if remove_unrendered_gaussians:
            gaussians.filter_gaussians(gaussian_renderer.get_seen_gaussians())

        del gaussian_renderer

    gaussian_sizes = gaussians.get_gaussian_sizes()

    print(f"Distributed points to gaussians")
    points_per_gaussian = distribute_points(gaussian_sizes, num_points).type(torch.int)

    start_bin, bin_size = calculate_bin_sizes(points_per_gaussian)

    point_distribution = torch.unique(points_per_gaussian)
    point_distribution = torch.cat((point_distribution[:start_bin],  torch.mul(torch.unique(torch.ceil(point_distribution[start_bin:]/ bin_size)), bin_size)), 0)

    total_points = torch.tensor([], device=device)
    total_colours = torch.tensor([], device=device).type(torch.double)

    num_attempts = 5

    print(f"Starting point cloud generation")

    for current_i in range(point_distribution.shape[0]-1):

        start_range = point_distribution[current_i]
        end_range = point_distribution[current_i+1]
        
        gaussian_indices = torch.where((points_per_gaussian >= start_range) & (points_per_gaussian < end_range))[0]

        num_points_for_gaussian = floor(start_range + (end_range-start_range)/2)

        if num_points_for_gaussian <= 0:
            continue

        if gaussian_indices.shape[0] < 1:
            continue

        print(current_i)

        covariances_for_point = gaussians.covariances[gaussian_indices]
        mean_for_point = gaussians.xyz[gaussian_indices]
        centre_colours = gaussians.gaussian_colours[gaussian_indices]
        
        total_points = torch.cat((total_points, mean_for_point), 0)
        total_colours = torch.cat((total_colours, centre_colours), 0)

        if num_points_for_gaussian <= 1: 
            continue

        new_points, new_colours = create_new_gaussian_points(num_points_for_gaussian-1, mean_for_point, covariances_for_point,
                                                             centre_colours, std_distance=std_distance, num_attempts=num_attempts, device=device)

        total_points = torch.cat((total_points, new_points), 0)
        total_colours = torch.cat((total_colours, new_colours), 0)

        print()

    torch.cuda.empty_cache()  
    gc.collect()  

    return total_points, total_colours

def config_parser():

    parser = configargparse.ArgumentParser()

    parser.add_argument("--input_path",  type=str, required=True, help="Path to ply or splat file to convert to a point cloud")
    parser.add_argument("--output_path",  type=str, default="3dgs_pc.ply", help="Path to output file (must be ply file)")

    parser.add_argument("--transform_path", type=str, help="Path to COLMAP or Transform file used for loading in camera positions for rendering")
    
    parser.add_argument("--skip_render_colours", action="store_true", help="Skip rendering colours- faster but colours will be strange")
    parser.add_argument("--colour_quality", type=str, default="medium", help="The quality of the colours when generating the point cloud (more quality = slower processing time)")

    parser.add_argument("--bounding_box_min", nargs=3, help="Values for minimum position of gaussians to include in generating the new point cloud")
    parser.add_argument("--bounding_box_max", nargs=3, help="Values for maximum position of gaussians to include in generating the new point cloud")

    parser.add_argument("--num_points", type=int, default=10000000, help="Total number of points to generate for the pointcloud")

    parser.add_argument("--std_distance", type=float, default=2.0, help="Maximum distance each point can be from the centre of their gaussian")

    parser.add_argument("--min_opacity", type=float, default=0.0, help="Minimum opacity for gaussians to be included (must be between 0-1)")

    parser.add_argument("--cull_gaussian_sizes", type=float, default=0.0, help="The percentage of gaussians to remove from largest to smallest (used to remove large gaussians)")

    args = parser.parse_args()

    if args.min_opacity < 0 or args.min_opacity > 1:
        raise AttributeError("Minumum opacity must be between 0 and 1")

    if args.std_distance <= 0:
        raise AttributeError("Std distance must be greater than 0")

    if args.num_points <= 0:
        raise AttributeError("Number of points must be greater than 0")
    
    if  args.bounding_box_min is not None:
        try:
            args.bounding_box_min = [float(x) for x in args.bounding_box_min]
        except ValueError:
            raise AttributeError("Bounding Box Min must contain float values")

        if len(args.bounding_box_min) != 3:
            raise AttributeError("Bounding Box Min must have exactly 3 values")

    if  args.bounding_box_max is not None:
        try:
            args.bounding_box_max = [float(x) for x in args.bounding_box_max]
        except ValueError:
            raise AttributeError("Bounding Box Max must contain float values")

        if len(args.bounding_box_max) != 3:
            raise AttributeError("Bounding Box Max must have exactly 3 values")

    if args.colour_quality.lower() not in COLOR_QUALITY_OPTIONS.keys():
         raise AttributeError(f"Colour quality must be in the following options {COLOR_QUALITY_OPTIONS.keys()}")

    return args

def main():
    args = config_parser()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    total_points, total_colours = generate_pointcloud(args.input_path, args.num_points, std_distance=args.std_distance, render_colours=not args.skip_render_colours,
                                                      transform_path=args.transform_path, min_opacity=args.min_opacity, bounding_box_min=args.bounding_box_min, bounding_box_max=args.bounding_box_max,
                                                      cull_large_percentage=args.cull_gaussian_sizes, colour_resolution=int(COLOR_QUALITY_OPTIONS[args.colour_quality.lower()]), device=device)

    save_xyz_to_ply(total_points, args.output_path, rgb_colors=total_colours, chunk_size=10**6)

if __name__ == "__main__":
    main()
