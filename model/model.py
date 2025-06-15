import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    残差块模块
    用于构建深层神经网络的基本组件, 通过跳跃连接保持梯度流动和特征传递.
    """
    def __init__(self, num_channels: int) -> None:
        """
        初始化残差块
        参数:
            num_channels: 输入和输出通道数
        """
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        参数:
            x: 输入张量, 形状为 [batch_size, num_channels, height, width]
        返回:
            经过残差块处理后的张量, 形状与输入相同
        """
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


class GoNeuralNetwork(nn.Module):
    """
    围棋神经网络模型
    实现了基于残差网络的围棋AI模型, 包含策略网络和价值网络两个输出头.
    """
    def __init__(self, board_size: int = 9, input_channels: int = 6, 
                 num_channels: int = 256, num_res_blocks: int = 10) -> None:
        """
        初始化围棋神经网络
        参数:
            board_size: 棋盘大小, 默认为9
            input_channels: 输入通道数, 默认为6
            num_channels: 中间层通道数, 默认为256
            num_res_blocks: 残差块数量, 默认为10
        """
        super().__init__()
        self.board_size = board_size
        
        def _initialize_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(_initialize_weights)
        
        # 输入层
        self.conv_input = nn.Sequential(
            nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        # 残差层
        self.res_blocks = nn.ModuleList([ResBlock(num_channels) for _ in range(num_res_blocks)])
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size + 1)  # +1 for pass move
        )
        # 价值头
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        参数:
            x: 输入张量, 形状为 [batch_size, input_channels, board_size, board_size]
        返回:
            tuple[policy, value]:
                policy: 策略输出, 形状为 [batch_size, board_size * board_size + 1]
                value: 局面评估值, 形状为 [batch_size, 1]
        """
        x = self.conv_input(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        # 策略头输出
        policy = self.policy_head(x)
        # 添加一些启发式规则, 有助于在训练前期的稳定性
        batch_size = x.shape[0]
        board_size = self.board_size
        # 1. 天元和星位点有较高优先级
        start_points = torch.zeros((batch_size, board_size * board_size + 1), device=x.device)
        if board_size == 9:
            starts = [20, 24, 28, 40, 44, 48, 60, 64, 68]   # 9 * 9 棋盘的星位点
            start_points[:, starts] = 1.0
        # 2. 边缘点有较低优先级
        edge_penalty = torch.ones((batch_size, board_size * board_size + 1), device=x.device)
        for i in range(board_size):
            for j in range(board_size):
                if i == 0 or i == board_size - 1 or j == 0 or j == board_size - 1:
                    edge_penalty[:, i * board_size + j] = 0.8
        policy = policy * start_points * edge_penalty
        # 3. 输出的策略结果总和为100%
        policy = F.softmax(policy, dim=1)
        # 价值头输出
        value = self.value_head(x)
        return policy, value
