import torch
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
    return True

def uv_to_color(uv: torch.Tensor, color_mode="uv") -> torch.Tensor:
    """
    uv: (M, N, 2) in [0,1]
    返回: (M, N, 3) 颜色
    """

    assert uv.ndim == 3 and uv.size(-1) == 2, "uv 必须是 (M,N,2)"

    u = uv[..., 0]
    v = uv[..., 1]

    if color_mode == "uv":
        # R=u, G=v, B=1-u
        r = u
        g = v
        b = 1 - u
        colors = torch.stack([r, g, b], dim=-1)

    elif color_mode == "uv_avg":
        # B = (u+v)/2
        b = 0.5 * (u + v)
        colors = torch.stack([u, v, b], dim=-1)

    elif color_mode == "angle":
        # 用 uv 相对中心的极角映射色相
        centered = uv - 0.5
        ang = torch.atan2(centered[..., 1], centered[..., 0])  # [-pi,pi]
        h = (ang + torch.pi) / (2 * torch.pi)  # [0,1]
        s = torch.ones_like(h)
        v = torch.ones_like(h)

        # HSV -> RGB (pytorch 版)
        colors = hsv_to_rgb_torch(torch.stack([h, s, v], dim=-1))

    else:
        raise ValueError("unknown color_mode")

    return colors.clamp(0, 1)


def hsv_to_rgb_torch(hsv: torch.Tensor) -> torch.Tensor:
    """
    hsv: (..., 3)
    return rgb: (..., 3)
    """

    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    i = torch.floor(h * 6).long()
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    i_mod = i % 6

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    idx = i_mod == 0
    r[idx], g[idx], b[idx] = v[idx], t[idx], p[idx]

    idx = i_mod == 1
    r[idx], g[idx], b[idx] = q[idx], v[idx], p[idx]

    idx = i_mod == 2
    r[idx], g[idx], b[idx] = p[idx], v[idx], t[idx]

    idx = i_mod == 3
    r[idx], g[idx], b[idx] = p[idx], q[idx], v[idx]

    idx = i_mod == 4
    r[idx], g[idx], b[idx] = t[idx], p[idx], v[idx]

    idx = i_mod == 5
    r[idx], g[idx], b[idx] = v[idx], p[idx], q[idx]

    return torch.stack([r, g, b], dim=-1)
