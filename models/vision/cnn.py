import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Optional, Dict
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CNNEncoder(nn.Module):
    """基于CNN的分子图像编码器"""
    
    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        output_dim: int = 2048,
        pool_type: str = "avg",
        freeze_backbone: bool = False,
        **kwargs
    ):
        """
        参数:
            model_name: resnet18/resnet50/efficientnet_v2_s
            pretrained: 是否使用预训练权重
            output_dim: 输出特征维度
            pool_type: 特征池化方式 (avg/max)
            freeze_backbone: 是否冻结主干网络
        """
        super().__init__()
        self.model_name = model_name.lower()
        self.pool_type = pool_type.lower()
        
        # 初始化预训练模型
        self.backbone = self._init_backbone(model_name, pretrained)
        self._adapt_output_layer(output_dim)
        
        # 冻结参数
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info(f"Frozen {model_name} backbone parameters")

    def _init_backbone(self, model_name: str, pretrained: bool) -> nn.Module:
        """初始化CNN主干网络"""
        if "resnet" in model_name:
            model_class = {
                "resnet18": models.resnet18,
                "resnet50": models.resnet50
            }[model_name]
            backbone = model_class(pretrained=pretrained)
            # 移除原始分类头
            backbone.fc = nn.Identity()
            return backbone
        elif "efficientnet" in model_name:
            from torchvision.models import efficientnet_v2_s
            backbone = efficientnet_v2_s(pretrained=pretrained)
            backbone.classifier = nn.Identity()
            return backbone
        else:
            raise ValueError(f"Unsupported CNN model: {model_name}")

    def _adapt_output_layer(self, output_dim: int):
        """调整输出层维度"""
        if "resnet" in self.model_name:
            in_features = 512 if "18" in self.model_name else 2048
            self.proj = nn.Linear(in_features, output_dim)
        else:
            self.proj = nn.Linear(1280, output_dim)  # efficientnet_v2_s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: 
            x - 图像张量 [B, C, H, W]
        返回:
            特征向量 [B, output_dim]
        """
        # 主干网络提取特征
        features = self.backbone(x)
        
        # 全局池化
        if self.pool_type == "max":
            features = F.adaptive_max_pool2d(features, (1, 1))
        else:  # avg pooling
            features = F.adaptive_avg_pool2d(features, (1, 1))
        
        # 展平并投影
        features = torch.flatten(features, 1)
        return self.proj(features)

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """获取最后一层特征图 (用于可视化)"""
        if "resnet" in self.model_name:
            # ResNet特定实现
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            return x
        else:
            raise NotImplementedError("Feature map extraction not implemented for this model")

    def get_config(self) -> Dict:
        """获取配置字典"""
        return {
            "model_name": self.model_name,
            "output_dim": self.proj.out_features,
            "pool_type": self.pool_type
        }