import sys
sys.path.append('../bspline-surface')
sys.path.append('../vecset-vae')
sys.path.append('../point-cept')

import torch

from bspline_vae.Dataset.random_bsp_surf import RandomBspSurfDataset
from bspline_vae.Model.bspline_vae import BSplineVAE
from bspline_vae.Model.ptv3_uv import PTV3UVNet

def test():
    dataset = RandomBspSurfDataset()

    data_dict = dataset.__getitem__(0)
    data_dict["split"] = 'train'
    data_dict["sample_posterior"] = True

    bspline_vae = PTV3UVNet().cuda()

    print('==== data_dict ====')
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.unsqueeze(0).cuda()
            print(key, value.shape)

    result_dict = bspline_vae(data_dict)

    print('==== result_dict ====')
    for key, value in result_dict.items():
        if isinstance(value, torch.Tensor):
            print(key, value.shape)

    pred_uv = result_dict['uv']

    loss = torch.mean(pred_uv)

    loss.backward()

    return True
