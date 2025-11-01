# 3D Gaussian Splatting to Point Cloud (or Mesh)

Gaussian Splatting can generate extremely high quality 3D representations of a scene. However, to properly view this reconstruction, specialised gaussian renders are required. Furthermore, a lot of 3D handling software are not compatible with 3D gaussians... but most are compatible with point clouds.

This repo offers scripts for converting a 3D Gaussian Splatting scene into a dense point cloud. The generated point clouds are high-quality and effectively imitate the original 3DGS scenes. Extra functionality is offered to customise the creation of the point cloud, as well as producing a mesh of the scene.

1) **Technical Paper:** *https://arxiv.org/abs/2501.07478*
2) **Research Article:** *https://radiancefields.com/3dgs-to-dense-ply*
3) **Research Article #2:** *https://radiancefields.com/3dgs-to-dense-point-cloud-v2*
3) **Youtube Video (1):** *https://www.youtube.com/watch?v=cOXfKRFqqxg*
4) **Youtube Video (2):** *https://www.youtube.com/watch?v=iB1WDiYxkws*

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZHA5MXptbjBjOGY1MzVwczFyejIydW1zdmdmejQ0aThkOG8wMXE2YiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/7UknswhXAHe88S93OY/giphy-downsized-large.gif" width="45%" />
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExeXdsM3k2Z3JlZ296eDZpOWlwNHc0cjZpZHA1djdoeDU3c3h0a2ZveSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Z9Cd1ENioEBGHxXcbs/giphy-downsized-large.gif" width="45%" /> 
</p>

Credit [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [Torch Splatting](https://github.com/hbb1/torch-splatting/tree/main), which were both used as part of this codebase.

## How to install

Firstly, clone this repo:
```bash
git clone https://github.com/Lewis-Stuart-11/3DGS-to-PC
```
Ensure that the original 3D Gaussian Splatting repo has been installed correctly, as this contains all required modules/packages.

Next you will need to install the CUDA gaussian renderer using the following command:
```bash
pip install ./gaussian-pointcloud-rasterization
```
Ensure that you are inside of this project directory when you run this command. If you cannot install this CUDA extension, then don't worry, we still offer a pure python renderer which does not require any package installation.

To perform meshing, you must install Open3D, which can be installed using the following command:
```bash
pip install open3d 
```

## How to run

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

| Argument                        | Default Value  | Description |
| :---                            |  :----:        |          ---: |
| **input_path**                  | -              |  Path to ply or splat file to convert to a point cloud |
| output_path                     | 3dgs_pc.ply    |  Path to output file (must be ply file) |
| **transform_path**              | -              |  Path to COLMAP or Transform file used for loading in camera positions for rendering colours |
| mask_path                       | -              |  Path to directory containing associated masks (must have the same image names as in transforms) |
| renderer_type                   | cuda           |  The type of renderer to use for determining point colours (currently supports 'cuda' or 'python') |
| **num_points**                  | 10000000       |  Total number of points to generate for the pointcloud |
| exact_num_points                | False          |  Set if the number of generated points should more closely match the num_points argument (slower) |
| no_prioritise_visible_gaussians | False          |  Gaussians that contribute most to the scene are given more points- set to turn this off |
| visibility_threshold            | 0.05           |  Minimum contribution each Gaussian must have to be included in the final point cloud generation (smaller value = less noise) |
| surface_distance_std            | -              | Cull Gaussians that are a minimum of X standard deviations away from the predicted scene surfaces (smaller value = less noise) |
| clean_pointcloud                | False          |  Set to remove outliers on the point cloud after generation (requires Open3D) |
| generate_mesh                   | False          |  Set to also generate a mesh based on the created point cloud  |
| poisson_depth                   | 10             |  The depth used in the poisson surface reconstruction algorithm that is used for meshing (larger value = more quality)  |
| laplacian_iterations            | 10             |  The number of iterations to perform laplacian mesh smoothing (higher value = smoother mesh) |
| mesh_output_path                | 3dgs_mesh.ply  |  Path to mesh output file (must be ply file) |
| camera_skip_rate                | 0              |  Number of cameras to skip for each rendered image (reduces compute time- only use if cameras in linear trajectory) |
| no_render_colours               | False          |  Skip rendering colours- faster but colours will be strange |
| colour_quality                  | high         |  The quality of the colours when generating the point cloud (more quality = slower processing time). Avaliable options are: tiny, low, medium, high, ultra and original |
| bounding_box_min                | -              |  Values for minimum position of gaussians to include in generating the new point cloud  |
| bounding_box_max                | -              |  Values for maximum position of gaussians to include in generating the new point cloud  |
| no_calculate_normals            | False          |  Set to not calculate and save normals for the points |  
| mahalanobis_distance_std        | 2.0            |  Maximum Mahalanobis distance each point can be from the centre of their gaussian |
| min_opacity                     | 0.0            |  Minimum opacity for gaussians that will be included (must be between 0-1) |
| cull_gaussian_sizes             | 0.0            |  The percentage of gaussians to remove from largest to smallest (must be between 0-1) |
| max_sh_degrees                  | 3              |  The number spherical harmonics of the loaded point cloud (default 3- change if different number of spherical harmonics are loaded) |
| quiet                           | False          |  Set to surpress any output print statements

## Tips

### Noise 
If you are experiencing a lot of noise in the final point cloud, we recommend increasing the ```visibility_threshold``` argument to a higher value. This means that Gaussians that contribute less to the rendered images will be removed (which typically represents noise).

Another approach is to set the ```surface_distance_std``` to a value, which will cull Gaussians that have a minimum distance to a predicted surface above a certain threshold, meaning that Gaussians far away from scene surfaces are removed. Decreasing the value will increase the harshness of the culling algorithm.

Finally, the ```clean_pointcloud``` will reduce noisy points using a statistical outlier removal algorithm (although this requires Open3D).

### Meshing

Our meshing approach is quite naive, which can lead to noise results. Some tips to improve the reconstruction:
1) Set the ```poisson_depth``` argument to a higher value (we found that 12 produced the best results, but any higher produced an infeasible mesh)
2) Set a bounding box to only mesh specific parts of the scene that are you need in the mesh
3) If the final mesh is too sharp, we recommend using some of the features in CloudCompare (e.g. smoothing) to get the desired output.

For generating a more accurate mesh, we recommend checking out [SuGaR](https://anttwo.github.io/sugar/).

### Speed
There are several ways that rendering speed can be increased without substantially impacting the final quality of the point cloud:
1) Set ```camera_skip_rate``` to a value where overlapping images are not rendered (e.g. we set camera_skip_rate = 4 for the mip dataset). Only do this if the camera poses are ordered in a linear trajectory around your scene and the camera poses overlap considerably. This can also have an impact in the point distribution quality
2) Set ```colour_quality``` to a lower option. This value is used to determine what resolution to render images of the scene; a lower quality will result in a faster render time.

If you are using the Python renderer, consider using the CUDA renderer instead, as it is much more efficient!

# Citation

```
@InProceedings{A_G_Stuart_2025_ICCV,
    author    = {A G Stuart, Lewis and Morton, Andrew and Stavness, Ian and Pound, Michael P},
    title     = {3DGS-to-PC: 3D Gaussian Splatting to Dense Point Clouds},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2025},
    pages     = {3730-3739}
}
```
