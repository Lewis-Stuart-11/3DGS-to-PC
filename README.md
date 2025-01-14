# 3D Gaussian Splatting to Point Cloud (or Mesh)

Gaussian Splatting can generate extremely high quality 3D representations of a scene. However, to properly view this reconstruction, specialised gaussian renders are required. Furthermore, a lot of 3D handling software are not compatible with 3D gaussians... but most are compatible with point clouds. 

This repo offers scripts for converting a 3D Gaussian Splatting scene into a dense point cloud. The generated point clouds are high-quality and effectively imitate the original 3DGS scenes. Extra functionality is offered to customise the creation of the point cloud, as well as producing a mesh of the scene.

1) **Technical Paper:** *https://arxiv.org/abs/2501.07478*
2) **Research Article:** *https://radiancefields.com/3dgs-to-dense-ply*
2) **Youtube Video:** *https://www.youtube.com/watch?v=cOXfKRFqqxg*

<p>
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZHA5MXptbjBjOGY1MzVwczFyejIydW1zdmdmejQ0aThkOG8wMXE2YiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/7UknswhXAHe88S93OY/giphy-downsized-large.gif" width="350" />
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExeXdsM3k2Z3JlZ296eDZpOWlwNHc0cjZpZHA1djdoeDU3c3h0a2ZveSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Z9Cd1ENioEBGHxXcbs/giphy-downsized-large.gif" width="350" /> 
</p>

Credit [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [Torch Splatting](https://github.com/hbb1/torch-splatting/tree/main), which were both used as part of this codebase.

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
| poisson_depth        | 10           |  The depth used in the poisson surface reconstruction algorithm that is used for meshing (larger value = more quality)  |
| mesh_output_path     | 3dgs_mesh.ply|  Path to mesh output file (must be ply file) |
| clean_pointcloud     | False        |  Set to remove outliers on the point cloud after generation (requires Open3D) |
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
1) Set the ```poisson_depth``` argument to a higher value (we found that 12 produced the best results, but any higher produced a infeasible mesh)
2) Set a bounding box to only mesh specific parts of the scene that are you need in the mesh
3) If the final mesh is too sharp, we recommend using some of the features in CloudCompare (e.g. smoothing) to get the desired output.

## How to increase speed

While the generated point clouds have a high accuracy and precise colours, the process can be slower than desired (especially for scenes with millions of Gaussians). There are several ways that speed can be increased without substantially impacting the final quality of the point cloud:
1) Set camera_skip_rate to a value where overlapping images are not rendered (e.g. we set camera_skip_rate = 4 for the mip dataset). We found that setting this value significantly reduced compile time, while not directly impacting the quality of the final reconstruction. Only do this if the camera poses are ordered in a linear trajectory around your scene and the camera poses overlap considerably.
2) Set colour_quality to a lower option. This value is used to determine what resolution to render images of the scene; a lower quality will result in a faster render time.

# Citation

If you want to know more about how this works, we recommend reading our paper below. Also, if you found our work useful, please consider citing:

```
@misc{stuart20253dgstopcconvert3dgaussian,
      title={3DGS-to-PC: Convert a 3D Gaussian Splatting Scene into a Dense Point Cloud or Mesh}, 
      author={Lewis A G Stuart and Michael P Pound},
      year={2025},
      eprint={2501.07478},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2501.07478}, 
}
```

