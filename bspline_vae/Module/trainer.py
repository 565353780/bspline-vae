import torch
from torch import nn
from typing import Union

from base_trainer.Module.base_trainer import BaseTrainer

from bspline_vae.Dataset.random_bsp_surf import RandomBspSurfDataset
from bspline_vae.Model.bsp_mlp_vae import BSplineMLPVAE
from bspline_vae.Model.bspline_vae import BSplineVAE
from bspline_vae.Model.ptv3_uv import PTV3UVNet
from bspline_vae.Method.triangle import buildUVMesh


class Trainer(BaseTrainer):
    def __init__(
        self,
        dataset_root_folder_path: str,
        batch_size: int = 5,
        accum_iter: int = 10,
        num_workers: int = 16,
        model_file_path: Union[str, None] = None,
        weights_only: bool = False,
        device: str = "cuda:0",
        dtype=torch.float32,
        warm_step_num: int = 2000,
        finetune_step_num: int = -1,
        lr: float = 2e-4,
        lr_batch_size: int = 256,
        ema_start_step: int = 5000,
        ema_decay_init: float = 0.99,
        ema_decay: float = 0.999,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
        best_model_metric_name: Union[str, None] = None,
        is_metric_lower_better: bool = True,
        sample_results_freq: int = -1,
        use_amp: bool = False,
        quick_test: bool = False,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path

        self.epoch_length: int = 1000000
        self.degree_u: int = 3
        self.degree_v: int = 3
        self.size_u: int = 5
        self.size_v: int = 5
        self.sample_num_u: int = 50
        self.sample_num_v: int = 50
        self.query_num_u: int = 100
        self.query_num_v: int = 100

        self.gt_sample_added_to_logger = False

        self.l1_loss = nn.L1Loss()
        # self.mse_loss = nn.MSELoss()

        super().__init__(
            batch_size,
            accum_iter,
            num_workers,
            model_file_path,
            weights_only,
            device,
            dtype,
            warm_step_num,
            finetune_step_num,
            lr,
            lr_batch_size,
            ema_start_step,
            ema_decay_init,
            ema_decay,
            save_result_folder_path,
            save_log_folder_path,
            best_model_metric_name,
            is_metric_lower_better,
            sample_results_freq,
            use_amp,
            quick_test,
        )
        return

    def createDatasets(self) -> bool:
        # FIXME: skip eval for faster training on slurm
        eval = False

        self.dataloader_dict["rand"] = {
            "dataset": RandomBspSurfDataset(
                self.epoch_length,
                self.degree_u,
                self.degree_v,
                self.size_u,
                self.size_v,
                self.sample_num_u,
                self.sample_num_v,
                self.query_num_u,
                self.query_num_v,
            ),
            "repeat_num": 1,
        }

        if eval:
            self.dataloader_dict["eval"] = {
                "dataset": RandomBspSurfDataset(
                    self.epoch_length,
                    self.degree_u,
                    self.degree_v,
                    self.size_u,
                    self.size_v,
                    self.sample_num_u,
                    self.sample_num_v,
                    self.query_num_u,
                    self.query_num_v,
                ),
            }

        if "eval" in self.dataloader_dict.keys():
            try:
                self.dataloader_dict["eval"]["dataset"].paths_list = self.dataloader_dict[
                    "eval"
                ]["dataset"].paths_list[:4]
            except:
                pass
        return True

    def createModel(self) -> bool:
        mode = 3
        if mode == 1:
            self.model = BSplineVAE().to(self.device)
        elif mode == 2:
            self.model = BSplineMLPVAE().to(self.device)
        elif mode == 3:
            self.model = PTV3UVNet().to(self.device)
        return True

    def getLossDict(self, data_dict: dict, result_dict: dict) -> dict:
        '''
        lambda_pts = 1.0
        lambda_kl = 0.001

        gt_pts = data_dict["query_pts"]
        pred_pts = result_dict["query_pts"]
        kl = result_dict["kl"]

        loss_pts = self.mse_loss(pred_pts, gt_pts)

        loss_kl = torch.mean(kl.float())

        loss = lambda_pts * loss_pts + lambda_kl * loss_kl

        loss_dict = {
            "Loss": loss,
            "LossPts": loss_pts,
            "LossKL": loss_kl,
        }
        '''

        gt_uv = data_dict["sample_uv"]
        pred_uv = result_dict["uv"]
        mask = result_dict.get('mask', None)

        if mask is not None:
            remain_idx_expand = mask.unsqueeze(-1).expand(-1, -1, 2)
            gt_uv = torch.gather(gt_uv, 1, remain_idx_expand).reshape(-1, 2)

        loss_uv = self.l1_loss(pred_uv, gt_uv)

        loss = loss_uv

        loss_dict = {
            "Loss": loss,
            #"LossUV": loss_uv,
        }

        return loss_dict

    def preProcessData(self, data_dict: dict, is_training: bool = False) -> dict:
        '''
        batch_num = data_dict['query_uv'].shape[0]
        data_dict['query_uv'] = data_dict['query_uv'].reshape(batch_num, -1, 2)
        data_dict['query_pts'] = data_dict['query_pts'].reshape(batch_num, -1, 3)
        '''

        if is_training:
            data_dict["sample_posterior"] = True
            data_dict["split"] = "train"
            data_dict["drop_prob"] = 0.2
        else:
            data_dict["sample_posterior"] = False
            data_dict["split"] = "val"
            data_dict["drop_prob"] = 0

        return data_dict

    @torch.no_grad()
    def sampleModelStep(self, model: nn.Module, model_name: str) -> bool:
        # FIXME: skip this since it will occur NCCL error
        return True

        dataset = self.dataloader_dict["rand"]["dataset"]

        model.eval()

        data_dict = dataset.__getitem__(0)
        data_dict["sample_posterior"] = False
        data_dict["split"] = "val"

        # process data here
        u_num, v_num = data_dict['query_uv'].shape[:2]
        data_dict['sample_pts'] = data_dict['sample_pts'].reshape(1, -1, 3).to(self.device, dtype=self.dtype)
        data_dict['query_uv'] = data_dict['query_uv'].reshape(1, -1, 2).to(self.device, dtype=self.dtype)

        result_dict = model(data_dict)

        self.logger.addMesh(
            model_name + "/query",
            buildUVMesh(result_dict['query_pts'].reshape(u_num, v_num, 3)),
            self.step,
        )

        if not self.gt_sample_added_to_logger:
            self.logger.addMesh(
                model_name + "/gt",
                buildUVMesh(data_dict['query_pts']),
                self.step,
            )
            self.gt_sample_added_to_logger = True

        return True
