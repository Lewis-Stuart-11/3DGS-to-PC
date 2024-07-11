import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import configargparse

from scipy.stats import norm
from math import cos, sin, pi, sqrt, tan, ceil, exp, floor
from torch.distributions.multivariate_normal import MultivariateNormal

from transform_dataloader import load_transform_data
from gauss_dataloader import load_gaussians, save_xyz_to_ply

from gauss_render import GaussRenderer
from camera_handler import Camera

scaling_modifier = 1

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = torch.exp(s[:,0])
    L[:,1,1] = torch.exp(s[:,1])
    L[:,2,2] = torch.exp(s[:,2])

    L = R @ L
    return L


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        return actual_covariance

def computeColorFromLowDegSH(sh):
    SH_C0 = 0.28209479177387814
    
    return torch.round(((SH_C0 * sh[:,:,0]) + 0.5) * 255).type(torch.double)

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

def filter_gaussians(xyz, opacities, min_opacity, bounding_box_min=None, bounding_box_max=None):
    valid_gaussians_indices = opacities > (min_opacity)

    if bounding_box_min is not None:
        bounding_box_min_indices = (xyz[:,0] > bounding_box_min[0]) & (xyz[:,1] > bounding_box_min[1]) \
                                  & (xyz[:,2] > bounding_box_min[2])

        valid_gaussians_indices = valid_gaussians_indices & bounding_box_min_indices

    if bounding_box_max is not None:
        bounding_box_max_indices = (xyz[:,0] < bounding_box_max[0]) & (xyz[:,1] < bounding_box_max[1]) \
                                  & (xyz[:,2] < bounding_box_max[2])

        valid_gaussians_indices = valid_gaussians_indices & bounding_box_max_indices

    return valid_gaussians_indices

def regularise_covariances(covariances):
    epsilon = 1e-6
    eye_matrix = epsilon * torch.eye(3, device='cuda:0').expand(covariances.shape[0], 3, 3)
    covariances += eye_matrix
    return covariances

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

    start_bin = np.nonzero(summed_arr[peak:] < cut_off)[0][0]

    #plt.plot(np.linspace(0, summed_arr.shape[0], num=summed_arr.shape[0], dtype=int), summed_arr)
    #plt.show()

    return start_bin, bin_size

def create_new_gaussian_points(num_points_to_sample, means, covariances, centre_colours, std_distance=2, num_attempts=5):
    total_required_points = num_points_to_sample * means.shape[0]

    added_points = torch.zeros(means.shape[0], device="cuda:0")
    max_count = torch.full((means.shape[0],), num_points_to_sample, device="cuda:0")
        
    new_points = torch.tensor([], device="cuda:0")
    new_colours = torch.tensor([], device="cuda:0").type(torch.double)

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

        grouped_idxs = torch.arange(sampled_points.shape[0], device="cuda:0")[filtered_samples_idxs].type(torch.float)

        grouped_idxs = torch.floor(torch.div(grouped_idxs, num_points_to_sample))
            
        counted_idxs, counts = torch.unique(grouped_idxs, return_counts=True)

        all_idxs = torch.arange(new_means_for_point.shape[0], device="cuda:0")

        zeroed_indxs = all_idxs[~torch.isin(all_idxs, counted_idxs.type(torch.int))]

        for element in zeroed_indxs:
            counts = torch.cat((counts[:element], torch.tensor([0], device="cuda:0"), counts[element:]))

        diffs = torch.min(max_count[gaussians_to_add_idxs]-added_points[gaussians_to_add_idxs], counts).type(torch.int)

        total_current_points = int(diffs.sum().item())

        current_points = torch.empty((total_current_points, sampled_points.size(1)), dtype=sampled_points.dtype, device="cuda:0")
        current_colours = torch.empty((total_current_points, new_colours_for_point.size(1)), dtype=new_colours_for_point.dtype, device="cuda:0")

        indices = torch.arange(len(diffs), device="cuda:0") * num_points_to_sample

        expanded_indices = indices.unsqueeze(1) + torch.arange(num_points_to_sample, device="cuda:0")
        expanded_indices = expanded_indices.flatten()

        mask = (torch.arange(num_points_to_sample, device="cuda:0").unsqueeze(0) < diffs.unsqueeze(1)).flatten()
        filtered_indices = expanded_indices[mask]

        current_points[:] = sampled_points[filtered_indices]
        current_colours[:] = new_colours_for_point.repeat_interleave(diffs, dim=0)

        added_points[gaussians_to_add_idxs] += counts
        added_points = torch.where(added_points > num_points_to_sample, num_points_to_sample, added_points).type(torch.int)

        new_points = torch.cat((new_points, current_points), 0)
        new_colours = torch.cat((new_colours, current_colours), 0)

        i += 1

    return new_points, new_colours
    
import imageio

#from gaussian_splatting.utils.camera_utils import Camera
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

def imwrite(path, image):
    imageio.imwrite(path, to8b(image))

def generate_pointcloud(input_path, output_path, num_points, std_distance=2, render_colours=True, transform_path=None,
                        min_opacity=0.5, bounding_box_min=None, bounding_box_max=None, transform_data_path=None, cull_large_percentage=0.025):
    
    xyz, scales, rots, features_all, opacities = load_gaussians(input_path)

    valid_gaussians_indices = filter_gaussians(xyz, opacities, min_opacity,
                                               bounding_box_min=bounding_box_min, bounding_box_max=bounding_box_max)

    xyz = xyz[valid_gaussians_indices]
    scales = scales[valid_gaussians_indices]
    rots = rots[valid_gaussians_indices]
    features_all = features_all[valid_gaussians_indices]
    opacities = opacities[valid_gaussians_indices]

    gaussian_sizes = torch.sum(torch.exp(scales * scaling_modifier), axis=1)

    cull_index = floor(gaussian_sizes.shape[0] *(1-cull_large_percentage))

    sorted_sizes, sorted_indices = torch.sort(gaussian_sizes)

    culled_gaussians = sorted_indices[:cull_index]

    xyz = xyz[culled_gaussians]
    scales = scales[culled_gaussians]
    rots = rots[culled_gaussians]
    features_all = features_all[culled_gaussians]
    opacities = opacities[culled_gaussians]

    gaussian_sizes = torch.sum(torch.exp(scales * scaling_modifier), axis=1)
    
    print(f"Culled- only {xyz.shape[0]} gaussians left")

    covariances = build_covariance_from_scaling_rotation(scales, scaling_modifier, rots)
    covariances = regularise_covariances(covariances)

    if render_colours:
        gaussian_renderer = GaussRenderer(xyz, torch.unsqueeze(torch.clone(opacities), 1).type(torch.float),
                                              torch.transpose(torch.clone(features_all), 1, 2), covariances,
                                              active_sh_degree=0, img_width=1080, img_height=1080)
        
        
        
        if transform_path is not None:

            print("Rendering colours from transform set")
        
            transforms, intrinsics = load_transform_data(transform_path)

            for i, (img_name, transform) in enumerate(transforms.items()):

                #if i % 2 ==0 :
                #    continue

                print(i)

                transform = torch.tensor(list(transform), device="cuda:0")

                cam_intrinsic = intrinsics[img_name]

                cam_matrix = torch.eye(4, device="cuda:0") 
                
                cam_matrix[0][0] = float(cam_intrinsic[2])
                cam_matrix[0][2] = float(cam_intrinsic[4])
                cam_matrix[1][1] = float(cam_intrinsic[3])
                cam_matrix[1][2] = float(cam_intrinsic[5])

                img_width = int(cam_intrinsic[0])
                img_height = int(cam_intrinsic[1])

                transform[2, 3] -= 1.2 #1.8/2 #torch.tensor([0.,0.,1.0])

                camera = Camera(img_width, img_height, cam_matrix, transform)

                render = gaussian_renderer.add_img(camera)

                img = render['render'].detach().cpu().numpy()

                imwrite(f"results\\{i}.png", img)

                print()
        else:
            print("TODO: Generate new cams")

            exit(0)    

        gaussian_colours = gaussian_renderer.get_colours()

    else:
        gaussian_colours = computeColorFromLowDegSH(features_all).type(torch.double)

    print(f"Distributed points to gaussians")
    points_per_gaussian = distribute_points(gaussian_sizes, num_points).type(torch.int)

    start_bin, bin_size = calculate_bin_sizes(points_per_gaussian)

    point_distribution = torch.unique(points_per_gaussian)
    point_distribution = torch.cat((point_distribution[:start_bin],  torch.mul(torch.unique(torch.ceil(point_distribution[start_bin:]/ bin_size)), bin_size)), 0)

    total_points = torch.tensor([], device="cuda:0")
    total_colours = torch.tensor([], device="cuda:0").type(torch.double)

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

        covariances_for_point = covariances[gaussian_indices]
        mean_for_point = xyz[gaussian_indices]
        sh_for_point = features_all[gaussian_indices]

        centre_colours = gaussian_colours[gaussian_indices]
        
        total_points = torch.cat((total_points, mean_for_point), 0)
        total_colours = torch.cat((total_colours, centre_colours), 0)

        if num_points_for_gaussian <= 1: 
            continue

        new_points, new_colours = create_new_gaussian_points(num_points_for_gaussian-1, mean_for_point, covariances_for_point,
                                                             centre_colours, std_distance=std_distance, num_attempts=num_attempts)

        total_points = torch.cat((total_points, new_points), 0)
        total_colours = torch.cat((total_colours, new_colours), 0)

        print()

    total_points_np = total_points.cpu().detach().numpy()
    total_colours_np = total_colours.cpu().detach().numpy()

    print("Saving Point Cloud")

    save_xyz_to_ply(total_points_np, output_path, rgb_colors=total_colours_np)

    print("Finished!")


def config_parser():

    parser = configargparse.ArgumentParser()

    parser.add_argument("--input_path",  type=str, required=True, help="Path to ply or splat file to convert to a point cloud")
    parser.add_argument("--output_path",  type=str, default="3dgs_pc.ply", help="Path to output file (must be ply file)")

    parser.add_argument("--transform_path", type=str, help="")
    
    parser.add_argument("--skip_render_colours", action="store_true", help="Skip rendering colours- faster but colours will be strange")
    parser.add_argument("--colour_quality", type=str, default="high", help="The quality of the colours when generating the point cloud (more quality = slower processing time)")

    parser.add_argument("--bounding_box_min", nargs=3, help="Values for minimum position of gaussians include in point cloud")
    parser.add_argument("--bounding_box_max", nargs=3, help="Values for maximum position of gaussians include in point cloud")

    parser.add_argument("--num_points", type=int, default=1000000, help="Total number of points to generate for the pointcloud")

    parser.add_argument("--std_distance", type=float, default=2.0, help="Maximum distance each point can be from the centre of their gaussian")

    parser.add_argument("--min_opacity", type=float, default=0.0, help="Minimum opacity for gaussians to be included (must be between 0-1)")

    parser.add_argument("--cull_gaussian_sizes", type=float, default=0.0, help="")

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

    return args

def main():
    args = config_parser()

    generate_pointcloud(args.input_path, args.output_path, args.num_points, std_distance=args.std_distance, render_colours=not args.skip_render_colours,
                        transform_path=args.transform_path, min_opacity=args.min_opacity, bounding_box_min=args.bounding_box_min, bounding_box_max=args.bounding_box_max,
                        cull_large_percentage=args.cull_gaussian_sizes)

if __name__ == "__main__":
    main()



    
    
