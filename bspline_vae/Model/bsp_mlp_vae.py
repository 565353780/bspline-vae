import math
import torch
import torch.nn as nn

from vecset_vae.Model.diagonal_gaussian_distribution import (
    DiagonalGaussianDistribution,
)
from vecset_vae.Model.Layer.fourier_embedder import FourierEmbedder
from vecset_vae.Model.Transformer.perceiver_1d import Perceiver
from vecset_vae.Model.perceiver_cross_attention_encoder import (
    PerceiverCrossAttentionEncoder,
)


class MLPDecoder(nn.Module):
    """MLP解码器，用于将查询点和潜在特征解码为3D坐标
    
    Args:
        embedder: 用于位置编码的嵌入器
        out_dim: 输出维度，通常为3（表示3D坐标）
        hidden_dim: MLP隐藏层维度
        num_layers: MLP层数
        init_scale: 初始化缩放因子
    """
    def __init__(
        self,
        embedder,
        out_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4,
        init_scale: float = 0.25,
        **kwargs
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.init_scale = init_scale
        
        # 嵌入维度
        self.embed_dim = embedder.out_dim
        
        # 使用自适应池化来处理可变长度的latents
        self.adaptive_pool = nn.AdaptiveAvgPool1d(hidden_dim)
        
        # 构建MLP层
        mlp_layers = []
        input_dim = self.embed_dim + hidden_dim
        
        # 添加隐藏层
        for _ in range(num_layers - 1):
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.LayerNorm(hidden_dim))
            mlp_layers.append(nn.ReLU(inplace=True))
            input_dim = hidden_dim
        
        # 添加输出层
        mlp_layers.append(nn.Linear(hidden_dim, out_dim))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=self.init_scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, queries: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            queries: 查询点坐标，形状为 [B, L, 2]（UV坐标）
            latents: 潜在特征，形状为 [B, M, 64]，其中M是可变的
            
        Returns:
            解码后的3D坐标，形状为 [B, L, out_dim]
        """
        B, L, D = queries.shape
        
        # 对查询点进行位置编码
        embedded_queries = self.embedder(queries)
        
        # 处理可变维度的latents [B, M, 64] -> [B, hidden_dim]
        # 使用自适应池化将任意长度的序列压缩到固定维度
        # 先转置为 [B, 64, M] 以便进行1D池化
        latents_transposed = latents.transpose(1, 2)  # [B, 64, M]
        
        # 对每个通道进行自适应池化，将M长度压缩到hidden_dim
        global_features_pooled = self.adaptive_pool(latents_transposed)  # [B, 64, hidden_dim]
        
        # 对通道维度求和，得到最终的全局特征
        global_features = global_features_pooled.sum(dim=1)  # [B, hidden_dim]
        
        # 将全局特征扩展到与查询点相同的批次维度 [B, hidden_dim] -> [B, L, hidden_dim]
        global_features = global_features.unsqueeze(1).repeat(1, L, 1)
        
        # 拼接查询点编码和全局特征
        combined_features = torch.cat([embedded_queries, global_features], dim=-1)
        
        # 通过MLP解码
        output = self.mlp(combined_features)
        
        return output


class BSplineMLPVAE(nn.Module):
    def __init__(
        self,
        use_downsample: bool = False,
        num_latents: int = 16,
        embed_dim: int = 16,
        width: int = 64,
        point_feats: int = 0,
        embed_point_feats: bool = False,
        query_dim: int = 2,
        out_dim: int = 3,
        num_freqs: int = 8,
        include_pi: bool = False,
        heads: int = 16,
        num_encoder_layers: int = 8,
        num_decoder_layers: int = 4,
        init_scale: float = 0.25,
        qkv_bias: bool = False,
        use_ln_post: bool = True,
        use_flash: bool = True,
        use_checkpoint: bool = True,
        split: str = "val",
    ) -> None:
        super().__init__()

        self.use_downsample = use_downsample
        self.num_latents = num_latents
        self.embed_dim = embed_dim
        self.width = width
        self.point_feats = point_feats
        self.embed_point_feats = embed_point_feats
        self.query_dim = query_dim
        self.out_dim = out_dim
        self.num_freqs = num_freqs
        self.include_pi = include_pi
        self.heads = heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.init_scale = init_scale
        self.qkv_bias = qkv_bias
        self.use_ln_post = use_ln_post
        self.use_flash = use_flash
        self.use_checkpoint = use_checkpoint
        self.split = split

        self.embedder = FourierEmbedder(
            num_freqs=self.num_freqs,
            include_pi=self.include_pi,
        )

        self.init_scale = self.init_scale * math.sqrt(1.0 / self.width)
        self.encoder = PerceiverCrossAttentionEncoder(
            use_downsample=self.use_downsample,
            embedder=self.embedder,
            num_latents=self.num_latents,
            point_feats=self.point_feats,
            input_sharp_pc=False,
            embed_point_feats=self.embed_point_feats,
            width=self.width,
            heads=self.heads,
            layers=self.num_encoder_layers,
            init_scale=self.init_scale,
            qkv_bias=self.qkv_bias,
            use_ln_post=self.use_ln_post,
            use_flash=self.use_flash,
            use_checkpoint=self.use_checkpoint,
        )

        self.pre_kl = nn.Linear(self.width, self.embed_dim * 2)
        self.post_kl = nn.Linear(self.embed_dim, self.width)

        self.transformer = Perceiver(
            n_ctx=self.num_latents,
            width=self.width,
            layers=self.num_decoder_layers,
            heads=self.heads,
            init_scale=self.init_scale,
            qkv_bias=self.qkv_bias,
            use_flash=self.use_flash,
            use_checkpoint=self.use_checkpoint,
        )

        if self.query_dim == 3:
            self.decoder = MLPDecoder(
                embedder=self.embedder,
                out_dim=self.out_dim,
                hidden_dim=256,
                num_layers=4,
                init_scale=self.init_scale,
            )
        else:
            self.decode_embedder = FourierEmbedder(
                num_freqs=self.num_freqs,
                input_dim=self.query_dim,
                include_pi=self.include_pi,
            )
            self.decoder = MLPDecoder(
                embedder=self.decode_embedder,
                out_dim=self.out_dim,
                hidden_dim=256,
                num_layers=4,
                init_scale=self.init_scale,
            )
        return

    def encode(
        self,
        coarse_surface: torch.FloatTensor,
    ):
        """
        Args:
            surface (torch.FloatTensor): [B, N, 3+C]
        Returns:
            shape_latents (torch.FloatTensor): [B, num_latents, width]
            kl_embed (torch.FloatTensor): [B, num_latents, embed_dim]
            posterior (DiagonalGaussianDistribution or None):
        """

        shape_latents = self.encoder(coarse_surface, split=self.split)
        return shape_latents

    def encode_kl_embed(
        self, latents: torch.FloatTensor, sample_posterior: bool = True
    ):
        posterior = None
        moments = self.pre_kl(latents)  # 103，256，768 -》 103，256，128
        posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
        if sample_posterior:
            kl_embed = posterior.sample()  # 1，768，64
        else:
            kl_embed = posterior.mode()

        kl = posterior.kl()
        return kl_embed, kl

    def decode(self, latents: torch.FloatTensor) -> torch.Tensor:
        """
        Args:
            latents (torch.FloatTensor): [B, embed_dim]

        Returns:
            latents (torch.FloatTensor): [B, embed_dim]
        """
        latents = self.post_kl(
            latents
        )  # [B, num_latents, embed_dim] -> [B, num_latents, width]

        latents = self.transformer(latents)
        return latents

    def query(self, queries: torch.FloatTensor, latents: torch.Tensor):
        """
        Args:
            queries (torch.FloatTensor): [B, N, 3]
            latents (torch.FloatTensor): [B, embed_dim]

        Returns:
            logits (torch.FloatTensor): [B, N], tsdf logits
        """
        logits = self.decoder(queries, latents).squeeze(-1)
        return logits

    def forward(
        self,
        data_dict: dict,
    ) -> dict:
        self.split = data_dict["split"]
        coarse_surface = data_dict['sample_pts']
        sample_posterior = data_dict["sample_posterior"]
        query_uv = data_dict['query_uv']

        shape_latents = self.encode(coarse_surface)

        kl_embed, kl = self.encode_kl_embed(
            shape_latents, sample_posterior=sample_posterior
        )

        latents = self.decode(kl_embed)

        query_pts = self.query(query_uv, latents)

        result_dict = {
            "query_pts": query_pts,
            "kl": kl,
        }

        return result_dict
