import sys
sys.path.append('../bspline-surface')

from bspline_vae.Dataset.random_bsp_surf import RandomBspSurfDataset
from bspline_vae.Method.render import visualize_uv_tangents

def test():
    dataset = RandomBspSurfDataset()

    data_dict = dataset.__getitem__(1)
    tangents = data_dict["sample_tangents"]
    pts = data_dict["sample_pts"]

    print(tangents[0])
    print(tangents[1])
    print(tangents[2])

    # visualize_uv_tangents(pts, tangents)
    return True
