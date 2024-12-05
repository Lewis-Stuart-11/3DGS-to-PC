# 3D Gaussian Splatting to Point Cloud (or Basic Mesh)

Gaussian Splatting can generate extremely high quality 3D representations of a scene. However, to properly view this reconstruction, specialised gaussian renders are required. Furthermore, a lot of 3D handling software are not compatible with 3D gaussians... but most are compatible with point clouds. 

This repo offers scripts for converting a 3D Gaussian Splatting scene into a dense point cloud. The generated point clouds are high-quality and effectively imitate the original 3DGS scenes. Extra functionality is offered to customise the creation of the point cloud, as well as producing a mesh of the scene.

**Showcase:** *https://www.youtube.com/watch?v=cOXfKRFqqxg*

<p>
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZHA5MXptbjBjOGY1MzVwczFyejIydW1zdmdmejQ0aThkOG8wMXE2YiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/7UknswhXAHe88S93OY/giphy-downsized-large.gif" width="350" />
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExeXdsM3k2Z3JlZ296eDZpOWlwNHc0cjZpZHA1djdoeDU3c3h0a2ZveSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Z9Cd1ENioEBGHxXcbs/giphy-downsized-large.gif" width="350" /> 
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcndwdG9yaGNydmg2cW1ybTQwb2Nnb2luOGswcml2bGc4NGNqY3FwaCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/9M8l0Vv7YJnTFC8evQ/giphy-downsized-large.gif" width="350" />
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZjdhc21tNnJ6OTVibDlqOHB6MzR0dDZ3bmRyczNqN2lpNDM3a2JtZiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/1C4h9ZvYNryH5Zu1oZ/giphy-downsized-large.gif" width="350" /> 
</p>

Credit [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [Torch Splatting](https://github.com/hbb1/torch-splatting/tree/main), which were both used as part of this codebase

## How to run

Firstly, clone this repo:
```bash
git clone https://github.com/Lewis-Stuart-11/3DGS-to-PC
```
Ensure that the original 3D Gaussian Splatting repo has been installed correctly, as this contains all required modules/packages

To run the basic point cloud generation, execute the following:
```bash
python gauss_to_pc.py --input_path "path//to//gaussian_splat"
```
The gaussian splat file can be either a .ply or .splat format.

However, if just the input file is provided, the colours of the point cloud will not look like the original 3D gaussian splats. To generate with authentic colours, include the path to the transform file/folder:

```bash
python gauss_to_pc.py --input_path "path//to//gaussian_splat" --transform_path "path//to//tranasforms"
```

The transform path can either be to a transforms.json file or COLMAP output files.

## Functionality 

| Argument             | Default Value  | Description |
| :---                 |  :----:      |          ---: |
| input_path           | -            |  Path to ply or splat file to convert to a point cloud |
| output_path          | 3dgs_pc.ply  |  Path to output file (must be ply file) |
| transform_path       | -            |  Path to COLMAP or Transform file used for loading in camera positions for rendering colours |
| generate_mesh        | False        |  Set to also generate a mesh based on the created point cloud  |
| poisson_depth        | 12           |  The depth used in the poisson surface reconstruction algorithm that is used for meshing (larger value = more quality)  |
| mesh_output_path     | 3dgs_mesh.ply|  Path to mesh output file (must be ply file) |
| camera_skip_rate     | 0            |  Number of cameras to skip for each rendered image (reduces compute time- only use if cameras in linear trajectory) |
| num_points           | 10000000     |  Total number of points to generate for the pointcloud |
| exact_num_points     | False        |  Set if the number of generated points should more closely match the num_points argument (slower) |
| no_render_colours    | False        |  Skip rendering colours- faster but colours will be strange |
| colour_quality       | medium       |  The quality of the colours when generating the point cloud (more quality = slower processing time). Avaliable options are: tiny, low, medium, high and ultra |
| bounding_box_min     | -            |  Values for minimum position of gaussians to include in generating the new point cloud  |
| bounding_box_max     | -            |  Values for maximum position of gaussians to include in generating the new point cloud  |
| no_calculate_normals | False        |  Set to not calculate and save normals for the points |  
| std_distance         | 2.0          |  Maximum Mahalanobis distance each point can be from the centre of their gaussian |
| min_opacity          | 0.0          |  Minimum opacity for gaussians that will be included (must be between 0-1) |
| cull_gaussian_sizes  | 0.0          |  The percentage of gaussians to remove from largest to smallest (must be between 0-1) |
| max_sh_degrees       | 3            |  The number spherical harmonics of the loaded point cloud (default 3- change if different number of spherical harmonics are loaded) |

## Meshing

This repo offers functionality for generating a mesh from a 3DGS scene. For this to work, open3d must be installed:

```bash
pip install open3d 
```

Our mesh reconstruction works by generating a point cloud only containing the predicted surfaces of the scene (by culling Gaussians that do not contribute more than 25% to the final pixel colour). Poisson surface reconstruction and laplacian smoothing are then used to generate a final mesh of the scene. The meshes that are generated using this method properly represent the entire scene, but often struggle with fine details. For generating a more accurate mesh, we recommend checking out [SuGaR](https://anttwo.github.io/sugar/).

![Comparison of the generated point cloud and mesh for the bulldozer scene](https://i.imgur.com/Lzwhatr.png)

Some tips to improve the results:
1) Set a bounding box to only mesh specific parts of the scene that are you need in the mesh
2) If the final mesh is too sharp, we recommend using some of the features in CloudCompare (e.g. smoothing) to get the desired output.

## How to increase speed

While the generated point clouds have a high accuracy and precise colours, the process can be slower than desired (especially for scenes with millions of Gaussians). There are several ways that speed can be increased without substantially impacting the final quality of the point cloud:
1) Set camera_skip_rate to a value where overlapping images are not rendered (e.g. we set camera_skip_rate = 4 for the mip dataset). We found that setting this value significantly reduced compile time, while not directly impacting the quality of the final reconstruction. Only do this if the camera poses are ordered in a linear trajectory around your scene and the camera poses overlap considerably.
2) Set colour_quality to a lower option. This value is used to determine what resolution to render images of the scene; a lower quality will result in a faster render time.

## How this works

Firstly, the gaussians are loaded from the input file, with the 3D covariance matrices being calculated using the scales and rotations of each gaussian. The original gaussian colours are calculated from the spherical harmonics (with the degree=0 since these points do not change based on direction when they are part of the point cloud). Gaussians are then culled based on the bounding box, size cut off and minimum capacity arguments. Alongside this, the normals of each Gaussians are calculated by taking the smallest axis of the Gaussian; this facilitates meshing of the point cloud later.

There is an issue with using the loaded gaussian colours for generating new points; these colours do not accurately represent the scene. When rendering an image, gaussians that overlap on each pixel each contribute to the final colour of that pixel based on their opacity and order. Hence, a gaussian that is always behind another gaussian may only contribute 10% to the final pixel colour, and thus their colour does not accurately represent the contribution to the scene.

![Comparison of point clouds generated using original gaussian colours vs rendered colours](https://i.imgur.com/Y9ZVZaQ.png)

The fix is to use colours generated by rendering images of the scene and tracking the contributions of each gaussian at each frame. When a gaussian contributes the most to a particular pixel colour compared to other times it is rendered (e.g. the gaussian is closer to a surface at that particular camera perspective) then we assign the colour of that pixel to that gaussian. Hence, gaussians behind other colours will not have erroneous colours and will have the same colour as the rendered images. The results are much better.

![Showcase of how rendering colours works compared to utilising the original gaussian colours](https://i.imgur.com/Kbkp4wI.png)

Now we have all the information required to start sampling points at each gaussian. Firstly, points are distributed to all gaussians based on the volume each gaussian has. Hence, larger gaussians will have more points compared to smaller ones, meaning that areas, such as backgrounds, have proper representation. 

Each of these gaussians are batched, with gaussians with the same number of points to be generated being batched together. Since larger gaussians are less common, the number of gaussians in each batch diminishes as the number of assigned points increases, which is inefficient when generating the points. Hence, after a certain number of points, these batches are 'binned' together. While this does mean that the number of generated points does not exactly match the specified argument, it is much more efficient.

The Torch multivariate normal function is used to sample over the gaussian distribution in batches. However, since a gaussian distribution is not definite, points can be generated that are 'outliers' as they differ too far from the gaussian's centre. Hence, for each point, the Mahalanobis distance is calculated from the centre of each gaussian to its points. If a point has a distance greater than 2 STD, then it is considered an outlier and removed. To ensure that gaussians with lots of random outliers are represented fairly, points that were removed are regenerated and checked again, and removed if they are outliers. This process is repeated until all points have been correctly generated or a max number of attempts has been made (we set this as 5).

Once all the points have been generated for all of the gaussians these points are exported to a .ply file.

## Issues

Currently, we are using an altered version of the gaussian renderer introduced in the Torch Splatting repo. While our alterations allow us to accurately calculate the colours of each point, the entire rendering process is slow compared to the original CUDA implementation (around 2 seconds per render). We plan on eventually implementing this into the original gaussian renderer, but this is a future plan. If anyone is up for the challenge, feel free to implement it and push to this repo :)

Another improvement can be automatically generating camera positions for rendering the colours, rather than requiring a set of camera transforms.
