import sys

sys.path.append("../bspline-surface/")
sys.path.append("../point-cept/")

import numpy as np
import open3d as o3d

from bspline_vae.Module.detector import Detector


def demo():
    model_file_path = "./output/v2/model_last.pth"
    use_ema = True
    device = "cuda"

    detector = Detector(
        model_file_path,
        use_ema,
        device,
    )

    pred_uv = detector.detectDataset(1)

    return True
