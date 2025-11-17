import torch
from torch import nn
from typing import Union

from base_trainer.Module.base_trainer import BaseTrainer

from bspline_vae.Dataset.random_bsp_surf import RandomBspSurfDataset
from bspline_vae.Model.bspline_vae import BSplineVAE


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
        self.sample_num_u: int = 40
        self.sample_num_v: int = 40

        self.gt_sample_added_to_logger = False

        self.loss_fn = nn.L1Loss()

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
        self.model = BSplineVAE().to(self.device)
        return True

    def getLossDict(self, data_dict: dict, result_dict: dict) -> dict:
        lambda_query_pts = 1.0
        lambda_kl = 0.001

        gt_query_pts = data_dict["query_pts"]
        pred_query_pts = result_dict["query_pts"]
        kl = result_dict["kl"]

        loss_query_pts = self.loss_fn(pred_query_pts, gt_query_pts)

        loss_kl = torch.mean(kl.float())

        loss = lambda_query_pts * loss_query_pts + lambda_kl * loss_kl

        loss_dict = {
            "Loss": loss,
            "LossQueryPts": loss_query_pts,
            "LossKL": loss_kl,
        }

        return loss_dict

    def preProcessData(self, data_dict: dict, is_training: bool = False) -> dict:
        if is_training:
            data_dict["sample_posterior"] = True
            data_dict["split"] = "train"
        else:
            data_dict["sample_posterior"] = False
            data_dict["split"] = "val"

        return data_dict

    @torch.no_grad()
    def sampleModelStep(self, model: nn.Module, model_name: str) -> bool:
        # FIXME: skip this since it will occur NCCL error
        return True

        dataset = self.dataloader_dict["dino"]["dataset"]

        model.eval()

        data_dict = dataset.__getitem__(1)

        print("[INFO][BaseDiffusionTrainer::sampleModelStep]")
        print("\t start sample shape code....")

        if not self.gt_sample_added_to_logger:
            # render gt here

            # self.logger.addPointCloud("GT_MASH/gt_mash", pcd, self.step)

            self.gt_sample_added_to_logger = True

        # self.logger.addPointCloud(model_name + "/pcd_" + str(i), pcd, self.step)

        return True
