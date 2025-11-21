import torch
import torch.nn as nn

from pointcept.models.point_transformer_v3.point_transformer_v3m2_sonata import (
    PointTransformerV3,
)


class PTV3UVNet(nn.Module):
    def __init__(
        self,
        use_flash: bool = True,
    ) -> None:
        super().__init__()

        self.use_flash = use_flash

        self.ptv3_encoder = PointTransformerV3(
            in_channels=3,
            enable_flash=False,
            enc_mode=False,
        )

        self.tangent_decoder = nn.Linear(64, 6)
        return

    def forward(
        self,
        data_dict: dict,
    ) -> dict:
        sample_pts = data_dict['sample_pts']
        drop_prob = data_dict.get('drop_prob', 0)

        B, N = sample_pts.shape[:2]

        if drop_prob > 0:
            remain_prob = 1.0 - torch.rand(1, device=sample_pts.device) * drop_prob
            remain_point_num = int(N * remain_prob)
            remain_idx = torch.rand(B, N, device=sample_pts.device).argsort(dim=1)[:, :remain_point_num]

            remain_idx_expand = remain_idx.unsqueeze(-1).expand(-1, -1, 3)  # [B, M, 3]
            sample_pts = torch.gather(sample_pts, 1, remain_idx_expand)

        coords = sample_pts.reshape(-1, 3)

        batch_indices = [
            torch.full((sample_pts.shape[1],), i, dtype=torch.long) for i in range(B)
        ]
        batch = torch.cat(batch_indices, dim=0).cuda()

        data = {
            "coord": coords,
            "feat": coords,
            "batch": batch,
            "grid_size": 0.01,
        }

        point = self.ptv3_encoder(data)

        feat = point.feat

        tangents = self.tangent_decoder(feat)

        result_dict = {
            "tangents": tangents,
        }

        if drop_prob > 0:
            result_dict['mask'] = remain_idx

        return result_dict
