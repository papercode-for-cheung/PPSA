import torch
import torch.nn as nn
from typing import Optional

class DynamicSubstructureGenerator(nn.Module):
    """动态子结构生成器（最终修正版）"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        noise_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        out_dim: Optional[int] = None,  # 可选输出维度
        **kwargs
    ):
        """
        参数:
            input_dim: 输入特征维度（h_ga和h_gb的拼接维度）
            hidden_dim: 隐藏层维度
            noise_dim: 噪声向量维度
            num_layers: 网络层数（至少2层）
            dropout: Dropout概率
            out_dim: 指定输出维度（默认等于hidden_dim）
            kwargs: 接收gnn_out_dim等额外参数用于验证
        """
        super().__init__()
        
        # 参数验证
        assert num_layers >= 2, "至少需要2层网络"
        self.out_dim = out_dim if out_dim is not None else hidden_dim
        self.noise_dim = noise_dim
        
        # 生成器网络构建
        layers = []
        current_dim = input_dim + noise_dim  # 初始输入维度
        
        for i in range(num_layers):
            # 确定下一层维度
            next_dim = hidden_dim if i < num_layers - 1 else self.out_dim
            
            # 添加线性层
            layers.append(nn.Linear(current_dim, next_dim))
            
            # 非最后一层添加激活和Dropout
            if i < num_layers - 1:
                layers.append(nn.LeakyReLU(negative_slope=0.2))
                layers.append(nn.Dropout(p=dropout))
            
            current_dim = next_dim
        
        self.generator = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """Xavier初始化"""
        for layer in self.generator:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, feats: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        前向传播
        输入:
            feats: [batch_size, input_dim]
            noise: [batch_size, noise_dim] 或 None
        返回:
            [batch_size, out_dim]
        """
        if noise is None:
            noise = torch.randn(feats.size(0), self.noise_dim, device=feats.device)
        
        # 维度检查
        if feats.dim() != 2:
            raise ValueError(f"输入必须是2D张量，但得到{feats.dim()}D")
        if noise.size(0) != feats.size(0):
            raise ValueError(f"Batch大小不匹配：feats{feats.size(0)} vs noise{noise.size(0)}")
        if noise.size(1) != self.noise_dim:
            raise ValueError(f"噪声维度应为{self.noise_dim}，但得到{noise.size(1)}")
        
        # 生成过程
        gen_input = torch.cat([feats, noise], dim=1)
        return self.generator(gen_input)  # 直接返回生成结果

    def get_config(self) -> dict:
        """获取配置字典"""
        return {
            'input_dim': self.generator[0].in_features - self.noise_dim,
            'hidden_dim': self.generator[0].out_features,
            'noise_dim': self.noise_dim,
            'out_dim': self.out_dim,
            'num_layers': len([m for m in self.generator if isinstance(m, nn.Linear)])
        }