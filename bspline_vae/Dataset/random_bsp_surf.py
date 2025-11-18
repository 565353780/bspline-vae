import torch
from torch.utils.data import Dataset

from bspline_surface.Model.bspline_surface import BSplineSurface

from bspline_vae.Method.rotate import random_rotation_matrices

def genBiggerArray(length: int, device="cpu", dtype=torch.float32):
    increments = torch.rand(length, device=device, dtype=dtype)
    x = torch.cumsum(increments, dim=0)
    x = x / x[-1]          # 归一化到 [0, 1]
    return x

def createUVMat(u_num: int, v_num: int) -> torch.Tensor:
    u = torch.linspace(0, 1, u_num)
    v = torch.linspace(0, 1, v_num)

    uu, vv = torch.meshgrid(u, v, indexing='ij')  # shape: [u_num, v_num]

    uv = torch.stack([uu, vv], dim=-1)  # shape: [u_num * v_num, 2]
    return uv

class RandomBspSurfDataset(Dataset):
    def __init__(
        self,
        epoch_length: int = 1000000,
        degree_u: int = 3,
        degree_v: int = 3,
        size_u: int = 5,
        size_v: int = 5,
        sample_num_u: int = 50,
        sample_num_v: int = 50,
        query_num_u: int = 50,
        query_num_v: int = 50,
    ) -> None:
        self.epoch_length = epoch_length

        self.degree_u = degree_u
        self.degree_v = degree_v
        self.size_u = size_u
        self.size_v = size_v
        self.sample_num_u = sample_num_u
        self.sample_num_v = sample_num_v
        self.query_num_u = query_num_u
        self.query_num_v = query_num_v

        self.sample_num_uv = self.sample_num_u * self.sample_num_v

        self.random_ctrlpts_xy = 0.01 * (torch.rand(self.size_u - 1,
                                            self.size_v - 1,
                                            2,
                                            dtype=torch.float32) - 0.5)

        self.random_ctrlpts_z = torch.rand(self.size_u - 1,
                                        self.size_v - 1,
                                        dtype=torch.float32) - 0.5
        self.random_rot = random_rotation_matrices(1)[0]
        return

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, index):
        bspline_surface = BSplineSurface(
            self.degree_u,
            self.degree_v,
            self.size_u,
            self.size_v,
            self.sample_num_u,
            self.sample_num_v,
            torch.float32,
            'cpu',
        )

        su = bspline_surface.size_u - 1
        sv = bspline_surface.size_v - 1

        # 1. 初始化 ctrlpts 为 tensor
        ctrlpts = torch.zeros((su, sv, 3), dtype=torch.float32)

        # 2. u、v 方向的参数（需你自己把 genBiggerArray 改成 PyTorch 版本）
        u_values = genBiggerArray(su)   # shape [su]
        v_values = genBiggerArray(sv)   # shape [sv]

        surf_length = 1.0

        # 3. 填充 ctrlpts[x, :, 0] 和 ctrlpts[:, y, 1]
        # X 方向 (u)
        ctrlpts[:, :, 0] = surf_length * (u_values - 0.5).unsqueeze(1)   # broadcast to [su,sv]

        # Y 方向 (v)
        ctrlpts[:, :, 1] = surf_length * (v_values - 0.5).unsqueeze(0)   # broadcast to [su,sv]

        # 4. 根据 index 区分初始化模式
        if index == 0:
            # xy 偏移
            ctrlpts[:, :, :2] = ctrlpts[:, :, :2] + self.random_ctrlpts_xy
            # z 方向直接赋值
            ctrlpts[:, :, 2] = self.random_ctrlpts_z

            random_rot = self.random_rot
        else:
            # xy 加很小的噪声
            ctrlpts[:, :, :2] = ctrlpts[:, :, :2] + \
                0.01 * (torch.rand_like(ctrlpts[:, :, :2]) - 0.5)
            # z 随机 [-0.5, 0.5]
            ctrlpts[:, :, 2] = torch.rand((su, sv)) - 0.5

            random_rot = random_rotation_matrices(1)[0]

        ctrlpts = ctrlpts @ random_rot

        bspline_surface.loadParams(ctrlpts=ctrlpts)

        sample_uv = torch.rand(self.sample_num_uv, 2)

        sample_pts = bspline_surface.toUVSamplePoints(sample_uv)

        pts_min = sample_pts.min(dim=0, keepdim=True)[0]
        pts_max = sample_pts.max(dim=0, keepdim=True)[0]

        center = (pts_min + pts_max) / 2  # [1,3]

        # AABB 尺寸
        max_range = (pts_max - pts_min).max()  # max of x/y/z extents

        new_ctrlpts = ((ctrlpts.reshape(-1, 3) - center) / max_range).reshape_as(ctrlpts)

        bspline_surface.loadParams(ctrlpts=new_ctrlpts)

        new_sample_pts = bspline_surface.toUVSamplePoints(sample_uv)

        query_uv = createUVMat(self.query_num_u, self.query_num_v)

        query_pts = bspline_surface.toUVSamplePoints(query_uv.reshape(-1, 2)).reshape(self.query_num_u, self.query_num_v, 3)

        data = {
            #"degree_u": bspline_surface.degree_u,
            #"degree_v": bspline_surface.degree_u,
            #"size_u": bspline_surface.size_u,
            #"size_v": bspline_surface.size_u,
            #"sample_num_u": bspline_surface.sample_num_u,
            #"sample_num_v": bspline_surface.sample_num_u,
            # "ctrlpts": bspline_surface.ctrlpts,
            # "sample_uv": sample_uv,
            "sample_pts": new_sample_pts,
            "query_uv": query_uv,
            "query_pts": query_pts,
        }

        return data
