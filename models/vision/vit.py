import torch
import torch.nn as nn
import timm
from typing import List, Optional, Dict
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ViTEncoder(nn.Module):
    """基于Vision Transformer的分子图像编码器"""
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        output_dim: int = 768,
        img_size: int = 224,
        patch_size: int = 16,
        pool_type: str = "cls",
        **kwargs
    ):
        """
        参数:
            model_name: ViT模型名称
            pretrained: 是否使用预训练权重
            output_dim: 输出特征维度
            img_size: 输入图像尺寸
            patch_size: patch大小
            pool_type: 特征池化方式 (cls/avg)
        """
        super().__init__()
        self.pool_type = pool_type.lower()
        
        # 初始化ViT模型
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # 移除分类头
            img_size=img_size,
            patch_size=patch_size
        )
        
        # 特征投影层
        self.proj = nn.Linear(self.vit.embed_dim, output_dim)
        
        # CLS token处理
        if pool_type == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.vit.embed_dim))
            self.vit.pos_embed = nn.Parameter(
                self._resize_pos_embed(
                    self.vit.pos_embed,
                    (img_size // patch_size, img_size // patch_size))
            )

    def _resize_pos_embed(self, posemb, grid_size):
        """调整位置编码尺寸"""
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        
        gs_old = int(posemb_grid.shape[0] ** 0.5)
        gs_new = grid_size[0]
        
        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bilinear')
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
        
        return torch.cat([posemb_tok, posemb_grid], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: 
            x - 图像张量 [B, C, H, W]
        返回:
            特征向量 [B, output_dim]
        """
        # 添加CLS token (如果使用)
        if self.pool_type == "cls":
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = self.vit.patch_embed(x)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.vit.pos_embed
            x = self.vit.norm_pre(x)
        else:
            x = self.vit.patch_embed(x)
            x = x + self.vit.pos_embed[:, 1:, :]  # 排除CLS token位置编码
            x = self.vit.norm_pre(x)
        
        # Transformer编码
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        
        # 特征池化
        if self.pool_type == "cls":
            x = x[:, 0]  # 取CLS token
        else:
            x = x.mean(dim=1)  # 全局平均池化
        
        return self.proj(x)

    def get_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """获取所有patch的特征 (用于可视化)"""
        x = self.vit.patch_embed(x)
        x = x + self.vit.pos_embed[:, 1:, :]  # 排除CLS token
        x = self.vit.norm_pre(x)
        x = self.vit.blocks(x)
        return self.vit.norm(x)

    def get_config(self) -> Dict:
        """获取配置字典"""
        return {
            "model_name": self.vit.default_cfg["architecture"],
            "output_dim": self.proj.out_features,
            "img_size": self.vit.patch_embed.img_size[0],
            "patch_size": self.vit.patch_embed.patch_size[0],
            "pool_type": self.pool_type
        }