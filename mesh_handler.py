import numpy as np

def generate_ball_pivoting_mesh():
    import open3d as o3d

    # Estimate radii for ball pivoting
    radii = [0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07] 

    # Generate mesh using the ball pivoting
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        point_cloud, o3d.utility.DoubleVector(radii)
    )

    # Normalise mesh using the laplacian smoothing method
    try:
        cleaned_mesh = mesh.filter_smooth_laplacian(number_of_iterations=filter_iters, filter_scope=o3d.geometry.FilterScope.Vertex)
    except Exception:
        cleaned_mesh = mesh

    return cleaned_mesh

def generate_poisson_mesh(point_cloud, depth=12, filter_iters=10):
    import open3d as o3d

    # Generate mesh using the poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=depth)

    # Remove triangles on the mesh with low density 
    vertices_to_remove = densities < np.quantile(densities, 0.005)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # Normalise mesh using the laplacian smoothing method
    try:
        cleaned_mesh = mesh.filter_smooth_laplacian(number_of_iterations=filter_iters, filter_scope=o3d.geometry.FilterScope.Vertex)
    except Exception:
        cleaned_mesh = mesh

    return cleaned_mesh

def generate_mesh(points, colours, normals, output_path, depth=12, filter_iters=10):
    """
    Generates a mesh from a point cloud

    Args:
        points: a tensor of point positions in the form [x, 3]
        colours: a tensor of point colours in the form [x, 3]
        normals: a tensor of point normals in the form [x, 3]
        output_path: path of the mesh file that will be generated
        depth: the depth of the poisson reconstruction (higher = more quality)
        filter_iters: the number of iterations to perform the laplacian smoothing
    """

    import open3d as o3d

    point_cloud = o3d.geometry.PointCloud()

    # Add points, normals and colours to point cloud object
    point_cloud.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
    point_cloud.colors =  o3d.utility.Vector3dVector(colours.detach().cpu().numpy()/255)
    point_cloud.normals = o3d.utility.Vector3dVector(normals.detach().cpu().numpy())

    mesh = generate_poisson_mesh(point_cloud, depth=depth, filter_iters=filter_iters)

    o3d.io.write_triangle_mesh(output_path, mesh)

    