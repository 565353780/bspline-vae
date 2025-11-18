import torch
import numpy as np
import open3d as o3d

def build_triangles(M: int, N: int) -> np.ndarray:
    # 网格中每个 quad 的左上角索引
    # i ∈ [0, M-2], j ∈ [0, N-2]
    ij = np.arange((M-1)*(N-1)).reshape(M-1, N-1)

    # 四个角点：a b
    #           c d
    a = ij * 0 + (np.arange(M-1)[:,None] * N + np.arange(N-1)[None,:])
    b = a + 1
    c = a + N
    d = c + 1

    # 两个三角面
    tri1 = np.stack([a, b, c], axis=-1)
    tri2 = np.stack([c, b, d], axis=-1)

    # 合并为 (T, 3)
    triangles = np.concatenate([tri1.reshape(-1,3), tri2.reshape(-1,3)], axis=0)
    return triangles

def build_double_sided_from_single(M: int, N: int) -> np.ndarray:
    triangles = build_triangles(M, N)

    triangles_rev = triangles[:, ::-1]

    return np.concatenate([triangles, triangles_rev], axis=0)

def buildUVMesh(uv_points: torch.Tensor) -> o3d.geometry.TriangleMesh:
    vertices = uv_points.detach().cpu().numpy().reshape(-1, 3).astype(np.float64)
    sample_num_u, sample_num_v = uv_points.shape[:2]
    triangles = build_triangles(sample_num_u, sample_num_v)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh
