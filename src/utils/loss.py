import torch
import torch.nn as nn
import numpy as np


class UncertaintyWeightedLoss(nn.Module):
    """
        loss_values = [loss1, loss2, ..., lossN] # 包含各个标量损失张量的列表
        total_weighted_loss = loss_weigher(loss_values) # 计算总加权损失
        total_weighted_loss.backward() # 反向传播
    """

    def __init__(self, num_losses: int):
        """
        Args:
            num_losses (int): 需要加权的损失项数量。
        """
        super().__init__()

        self.num_losses = num_losses

        self.log_vars = nn.Parameter(torch.zeros(num_losses, dtype=torch.float32))

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        """
        计算总加权损失。

        Args:
            losses (list[torch.Tensor]): 一个列表或元组，包含各个独立的标量损失值（未加权的）。
                                          其长度必须等于初始化时的 `num_losses`。

        Returns:
            torch.Tensor: 最终用于最小化的标量损失值。
        """

        total_weighted_loss = 0
        for i in range(self.num_losses):
            log_var = self.log_vars[i]

            precision = torch.exp(-log_var)
            weighted_loss_term = 0.5 * precision * losses[i] + 0.5 * log_var
            total_weighted_loss += weighted_loss_term

        return total_weighted_loss

    @property
    def weights(self) -> np.ndarray:
        """以 numpy 数组形式返回学习到的精度权重 (1/sigma^2)。"""
        with torch.no_grad():
            # weights = precision = exp(-log_var)
            return torch.exp(-self.log_vars).cpu().numpy()

    @property
    def sigmas(self) -> np.ndarray:
        """以 numpy 数组形式返回学习到的标准差 (sigma)。"""
        with torch.no_grad():
            # sigma = sqrt(sigma^2) = sqrt(exp(log_var)) = exp(0.5 * log_var)
            return torch.exp(0.5 * self.log_vars).cpu().numpy()

