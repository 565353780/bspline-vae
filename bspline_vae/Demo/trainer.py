import sys

sys.path.append("../bspline-surface/")
sys.path.append("../base-trainer/")
sys.path.append("../vecset-vae/")
sys.path.append("../point-cept/")

import os
import torch

from bspline_vae.Module.trainer import Trainer


def demo():
    dataset_root_folder_path = os.environ["HOME"] + "/chLi/Dataset/"
    assert dataset_root_folder_path is not None
    print(dataset_root_folder_path)

    batch_size = 512
    accum_iter = 1
    num_workers = 32
    model_file_path = "./output/v1/model_last.pth"
    model_file_path = None
    weights_only = False
    device = "auto"
    dtype = torch.float32
    warm_step_num = 200
    finetune_step_num = -1
    lr = 2e-5
    lr_batch_size = 256
    ema_start_step = 5000
    ema_decay_init = 0.99
    ema_decay = 0.999
    save_result_folder_path = "auto"
    save_log_folder_path = "auto"
    best_model_metric_name = None
    is_metric_lower_better = True
    sample_results_freq = 1
    use_amp = True
    quick_test = False

    trainer = Trainer(
        dataset_root_folder_path,
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

    model_size = trainer.getModelSize()
    print('model size:', model_size / 1e6, 'M')

    trainer.train()
    return True
