import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv
from torch_geometric.utils import dropout_adj
from typing import Optional, Dict, List
from torch_geometric.nn import global_mean_pool  # 添加这行导入

class GNNEncoder(nn.Module):
    """可配置的GNN编码器基类"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        out_channels: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        gnn_type: str = 'gat',
        heads: int = 4,
        **kwargs
    ):
        """
        参数:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出维度
            num_layers: GNN层数
            dropout: Dropout概率
            gnn_type: gcn/gat/gin
            heads: GAT的头数
        """
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        self.gnn_type = gnn_type.lower()
        
        # 输入层
        self.add_gnn_layer(in_channels, hidden_channels, heads, first_layer=True)
        
        # 隐藏层
        for _ in range(num_layers - 2):
            self.add_gnn_layer(hidden_channels, hidden_channels, heads)
        
        # 输出层
        self.add_gnn_layer(hidden_channels, out_channels, heads, last_layer=True)
        
        # 全局池化
        self.pool = global_mean_pool

    def add_gnn_layer(
        self, 
        in_dim: int, 
        out_dim: int, 
        heads: int = 1,
        first_layer: bool = False,
        last_layer: bool = False
    ):
        """添加GNN层"""
        if self.gnn_type == 'gat':
            conv = GATConv(
                in_dim, 
                out_dim // heads if not last_layer else out_dim,
                heads=1 if last_layer else heads,
                dropout=self.dropout
            )
        elif self.gnn_type == 'gcn':
            conv = GCNConv(in_dim, out_dim)
        elif self.gnn_type == 'gin':
            nn_seq = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
            conv = GINConv(nn_seq)
        else:
            raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
        
        self.convs.append(conv)
        if not last_layer:
            self.bns.append(nn.BatchNorm1d(out_dim))

    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        参数:
            x: 节点特征 [N, in_channels]
            edge_index: 边索引 [2, E]
            batch: 批索引 [N]
            edge_weight: 边权重 [E]
        返回:
            图级表示 [batch_size, out_channels]
        """
        # 边dropout
        edge_index, edge_weight = dropout_adj(
            edge_index, 
            edge_weight,
            p=self.dropout,
            training=self.training
        )
        
        # GNN消息传递
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 最后一层不使用激活函数
        x = self.convs[-1](x, edge_index, edge_weight)
        
        # 全局池化
        if batch is not None:
            x = self.pool(x, batch)
        return x

    def get_config(self) -> Dict:
        """获取配置字典"""
        return {
            'in_channels': self.convs[0].in_channels,
            'hidden_channels': self.bns[0].num_features,
            'out_channels': self.convs[-1].out_channels,
            'num_layers': len(self.convs),
            'dropout': self.dropout,
            'gnn_type': self.gnn_type
        }