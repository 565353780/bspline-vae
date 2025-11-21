import numpy as np
import open3d as o3d

def visualize_uv_tangents(points, tangents, scale=0.02):
    """
    points: (N,3) numpy
    tangents: (N,2,3) numpy (du,dv 已归一化)
    """
    N = points.shape[0]

    # base points
    pts = []
    lines = []

    pts.extend(points.tolist())

    # du vectors (red)
    du_start_index = 0
    du_end_index = N
    for i in range(N):
        p = points[i]
        du = tangents[i, 0] * scale
        pts.append((p + du).tolist())
        lines.append([i, N + i])

    # dv vectors (green)
    dv_start_index = N
    dv_end_index = 2 * N
    for i in range(N):
        p = points[i]
        dv = tangents[i, 1] * scale
        pts.append((p + dv).tolist())
        lines.append([i, N + N + i])

    # Open3D objects
    pts = np.array(pts)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # color: du=red, dv=green
    colors = []
    for i in range(N):
        colors.append([1, 0, 0])  # red du
    for i in range(N):
        colors.append([0, 1, 0])  # green dv
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    o3d.visualization.draw_geometries([pcd, line_set])
