# training/evaluation.py
import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss
)
from typing import Dict, List, Optional
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsTracker:
    """指标跟踪与计算工具类"""
    
    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """计算多种分类指标"""
        y_pred_class = (y_pred > threshold).astype(int)
        
        return {
            'accuracy': accuracy_score(y_true, y_pred_class),
            'auc': roc_auc_score(y_true, y_pred),
            'auprc': average_precision_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred_class, zero_division=0),
            'f1': f1_score(y_true, y_pred_class, zero_division=0),
            'recall': recall_score(y_true, y_pred_class, zero_division=0)  # 可选添加
        }

class Evaluator:
    """模型评估器，支持多指标计算"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        metrics: List[str] = None,
        **kwargs
    ):
        """
        参数:
            model: 待评估模型
            device: 计算设备
            metrics: 需要计算的指标列表
        """
        self.model = model.to(device)
        self.device = device
        self.metrics = metrics or ['accuracy', 'auc', 'auprc', 'precision', 'f1']
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        return_predictions: bool = False,
        **kwargs
    ) -> Dict[str, float]:
        """执行完整评估"""
        y_true, y_pred = [], []
        
        for batch in tqdm(data_loader, desc="Evaluating"):
            # 准备数据
            inputs = {}
            if 'graph_a' in batch:
                inputs['graph_a'] = batch['graph_a'].to(self.device)
            if 'graph_b' in batch:
                inputs['graph_b'] = batch['graph_b'].to(self.device)
            if 'image_a' in batch:
                inputs['image_a'] = batch['image_a'].to(self.device)
            if 'image_b' in batch:
                inputs['image_b'] = batch['image_b'].to(self.device)
            
            # 前向传播
            outputs = self.model(**inputs).sigmoid().cpu().numpy()
            targets = batch['labels'].cpu().numpy()
            
            y_pred.extend(outputs.flatten())
            y_true.extend(targets.flatten())
        
        # 计算指标
        results = MetricsTracker.compute_metrics(
            np.array(y_true),
            np.array(y_pred)
        )
        # 筛选所需指标
        final_results = {
            'loss': log_loss(y_true, y_pred),
            **{k: results[k] for k in self.metrics if k in results}
        }
        
        if return_predictions:
            return final_results, (y_true, y_pred)
        return final_results
    
    def get_confusion_matrix(
        self,
        data_loader: torch.utils.data.DataLoader,
        threshold: float = 0.5
    ) -> Dict[str, int]:
        """获取混淆矩阵"""
        _, (y_true, y_pred) = self.evaluate(
            data_loader,
            return_predictions=True
        )
        y_pred_class = (np.array(y_pred) > threshold).astype(int)
        
        tp = ((y_true == 1) & (y_pred_class == 1)).sum()
        fp = ((y_true == 0) & (y_pred_class == 1)).sum()
        tn = ((y_true == 0) & (y_pred_class == 0)).sum()
        fn = ((y_true == 1) & (y_pred_class == 0)).sum()
        
        return {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }