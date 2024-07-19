# 3D Gaussian Splatting to Dense Point Cloud

Gaussian Splatting can generate extremely high quality 3D representations of a scene. However, to properly view this reconstruction, specialised gaussian renders are required. Furthermore, a lot of 3D handling software are not compatible with 3D gaussians... but most are compatible with point clouds. 

This repo offers scripts for converting a 3D Gaussian Splatting scene into a dense point cloud. The generated point clouds are high-quality and effectively imitate the original 3DGS scenes. Extra functionality is offered to customise the creation of the point cloud.

<p >
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZHA5MXptbjBjOGY1MzVwczFyejIydW1zdmdmejQ0aThkOG8wMXE2YiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/7UknswhXAHe88S93OY/giphy-downsized-large.gif" width="350" />
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExeXdsM3k2Z3JlZ296eDZpOWlwNHc0cjZpZHA1djdoeDU3c3h0a2ZveSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Z9Cd1ENioEBGHxXcbs/giphy-downsized-large.gif" width="350" /> 
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcndwdG9yaGNydmg2cW1ybTQwb2Nnb2luOGswcml2bGc4NGNqY3FwaCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/9M8l0Vv7YJnTFC8evQ/giphy-downsized-large.gif" width="350" />
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZjdhc21tNnJ6OTVibDlqOHB6MzR0dDZ3bmRyczNqN2lpNDM3a2JtZiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/1C4h9ZvYNryH5Zu1oZ/giphy-downsized-large.gif" width="350" /> 
</p>

Credit to the following repositories, which were used as part of this codebase:

    1. 3D Gaussian Splatting- https://github.com/graphdeco-inria/gaussian-splatting
    2. Torch Splatting- https://github.com/hbb1/torch-splatting/tree/main

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
| :---                 |    :----:      |          ---: |
| input_path           |    -           |  Path to ply or splat file to convert to a point cloud |
| output_path          |    3dgs_pc.ply |  Path to output file (must be ply file) |
| transform_path       |    -           |  Path to COLMAP or Transform file used for loading in camera positions for rendering colours |
| num_points           |    10000000    |  Total number of points to generate for the pointcloud |
| skip_render_colours  |    False       |  Skip rendering colours- faster but colours will be strange |
| colour_quality       |    medium      |  The quality of the colours when generating the point cloud (more quality = slower processing time). Avaliable options are: low, medium, high and ultra |
| bounding_box_min     |    -           |  Values for minimum position of gaussians to include in generating the new point cloud  |
| bounding_box_max     |    -           |  Values for maximum position of gaussians to include in generating the new point cloud  |
| std_distance         |    2.0         |  Maximum Mahalanobis distance each point can be from the centre of their gaussian |
| min_opacity          |    0.0         |  Minimum opacity for gaussians that will be included (must be between 0-1) |
| cull_gaussian_sizes  |    0.0         |  The percentage of gaussians to remove from largest to smallest (must be between 0-1) |

## How this works

Firstly, the gaussians are loaded from the input file, with the 3D covariance matrices being calculated using the scales and rotations of each gaussian. The original gaussian colours are calculated from the spherical harmonics (with the degree=0 since these points do not change based on direction when they are part of the point cloud). Gaussians are then culled based on the bounding box, size cut off and minimum capacity arguments.

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