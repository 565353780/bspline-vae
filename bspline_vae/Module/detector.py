import os
import torch
import open3d as o3d
from torch import nn
from typing import Union

from bspline_vae.Dataset.random_bsp_surf import RandomBspSurfDataset
from bspline_vae.Model.ptv3_uv import PTV3UVNet
from bspline_vae.Method.triangle import buildUVMesh
from bspline_vae.Method.render import uv_to_color


class Detector(object):
    def __init__(
        self,
        model_file_path: Union[str, None] = None,
        use_ema: bool = True,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.dtype = torch.float32

        self.model = PTV3UVNet().to(self.device)

        if model_file_path is not None:
            self.loadModel(model_file_path, use_ema)

        self.bsp_dataset = RandomBspSurfDataset(
            epoch_length=2,
            degree_u=3,
            degree_v=3,
            size_u=5,
            size_v=5,
            sample_num_u=50,
            sample_num_v=50,
            query_num_u=100,
            query_num_v=100,
        )
        return

    def loadModel(self, model_file_path: str, use_ema: bool = True) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][Detector::loadModel]")
            print("\t model file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        state_dict = torch.load(model_file_path, map_location="cpu")

        if use_ema:
            model_state_dict = state_dict["ema_model"]
        else:
            model_state_dict = state_dict["model"]

        self.model.load_state_dict(model_state_dict)
        self.model.eval()

        print("[INFO][Detector::loadModel]")
        print("\t load model success!")
        print("\t model_file_path:", model_file_path)
        return True

    @torch.no_grad()
    def detect(
        self,
        pts: torch.Tensor,
    ) -> torch.Tensor:
        pts_base_shape = pts.shape[:-1]
        pts_device = pts.device
        pts_dtype = pts.dtype

        if pts.ndim < 3:
            pts = pts.unsqueeze(0)

        data_dict = {
            'sample_posterior': False,
            'split': 'val',
            'sample_pts': pts.to(self.device, dtype=self.dtype),
        }

        result_dict = self.model(data_dict)

        pred_uv = result_dict['uv'].reshape(*pts_base_shape, 2).to(pts_device, dtype=pts_dtype)

        return pred_uv

    @torch.no_grad()
    def detectDataset(self, data_idx: int = 1) -> torch.Tensor:
        data_dict = self.bsp_dataset.__getitem__(data_idx)

        gt_uv = data_dict['query_uv']
        gt_pts = data_dict['query_pts']
        bsp_surf = data_dict['bsp_surf']

        pts = gt_pts.reshape(-1, 3)

        pred_uv = self.detect(pts).reshape(*gt_uv.shape)
        pred_pts = bsp_surf.toUVSamplePoints(pred_uv.reshape(-1, 2)).reshape(*gt_pts.shape[:2], 3)

        loss = nn.L1Loss()(pred_uv, gt_uv)
        print('loss uv:', loss)

        gt_uv_mesh = buildUVMesh(gt_pts)
        pred_uv_mesh = buildUVMesh(pred_pts)

        gt_uv_mesh.compute_vertex_normals()
        pred_uv_mesh.compute_vertex_normals()

        gt_uv_colors = uv_to_color(gt_uv, color_mode="uv").reshape(-1, 3).cpu().numpy()
        pred_uv_colors = uv_to_color(pred_uv, color_mode="uv").reshape(-1, 3).cpu().numpy()

        gt_uv_mesh.vertex_colors = o3d.utility.Vector3dVector(gt_uv_colors)
        pred_uv_mesh.vertex_colors = o3d.utility.Vector3dVector(pred_uv_colors)

        gt_uv_pcd = o3d.geometry.PointCloud()
        gt_uv_pcd.points = gt_uv_mesh.vertices
        gt_uv_pcd.normals = gt_uv_mesh.vertex_normals
        gt_uv_pcd.colors = gt_uv_mesh.vertex_colors

        pred_uv_pcd = o3d.geometry.PointCloud()
        pred_uv_pcd.points = pred_uv_mesh.vertices
        pred_uv_pcd.normals = pred_uv_mesh.vertex_normals
        pred_uv_pcd.colors = pred_uv_mesh.vertex_colors

        os.makedirs('./output/', exist_ok=True)
        o3d.io.write_triangle_mesh('./output/gt_uv_mesh.ply', gt_uv_mesh, write_ascii=True)
        o3d.io.write_triangle_mesh('./output/pred_uv_mesh.ply', pred_uv_mesh, write_ascii=True)
        o3d.io.write_point_cloud('./output/gt_uv_pcd.ply', gt_uv_pcd, write_ascii=True)
        o3d.io.write_point_cloud('./output/pred_uv_pcd.ply', pred_uv_pcd, write_ascii=True)

        return pred_uv
