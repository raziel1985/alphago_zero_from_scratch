# Model 目录

本目录包含 AlphaGo Zero 项目的核心AI组件，包括神经网络模型、蒙特卡洛树搜索算法和训练器。

## 目录结构

```
model/
├── __init__.py          # 包初始化文件
├── model.py             # 神经网络模型定义
├── mcts.py              # 蒙特卡洛树搜索算法
├── train.py             # 训练器和训练逻辑
├── checkpoints/         # 模型检查点目录
│   ├── *.pt            # 训练检查点文件
│   ├── latest_model.pt # 最新模型权重
│   └── training_history.json # 训练历史记录
└── README.md           # 本文件
```

## 核心模块

### 1. model.py - 神经网络模型

包含 `GoNeuralNetwork` 类，实现了 AlphaGo Zero 的神经网络架构：

- **输入**: 围棋棋盘状态 (board_size × board_size × 特征通道数)
- **输出**: 
  - 策略头 (Policy Head): 每个位置的落子概率
  - 价值头 (Value Head): 当前局面的胜率评估

**特性**:
- 基于 ResNet 架构的卷积神经网络
- 支持不同棋盘尺寸 (默认 9×9)
- 使用 PyTorch 实现

### 2. mcts.py - 蒙特卡洛树搜索

包含 `MCTS` 类，实现了 AlphaGo Zero 的搜索算法：

- **选择 (Selection)**: 使用 UCB1 公式选择最优路径
- **扩展 (Expansion)**: 扩展搜索树的新节点
- **评估 (Evaluation)**: 使用神经网络评估叶子节点
- **回传 (Backpropagation)**: 更新路径上所有节点的统计信息

**参数**:
- `num_simulations`: 模拟次数 (默认 800)
- `c_puct`: 探索常数 (默认 1.0)
- `dirichlet_alpha`: 狄利克雷噪声参数

### 3. train.py - 训练器

包含 `AlphaGoZeroTrainer` 类，实现了完整的自我对弈训练流程：

**训练流程**:
1. **自我对弈**: AI 与自己对弈生成训练数据
2. **数据收集**: 收集 (状态, 策略, 价值) 三元组
3. **神经网络训练**: 使用收集的数据训练网络
4. **模型评估**: 评估新模型与旧模型的性能
5. **模型更新**: 如果新模型更强则更新

**主要功能**:
- `self_play()`: 执行自我对弈
- `train_network()`: 训练神经网络
- `evaluate_model()`: 模型评估
- `save_checkpoint()`: 保存训练检查点

## 使用方法

### 环境设置

在开始使用之前，请确保安装所有必要的依赖：

```bash
# 在项目根目录安装统一依赖
pip install -r requirements.txt
```

### 运行训练脚本

**正确的运行方式**:
```bash
# 在项目根目录执行
python3 -m model.train
```

> **重要**: 必须使用模块运行方式 (`python3 -m model.train`)，不能直接运行 `python3 model/train.py`。这是因为：
> 1. 代码使用了绝对导入 (`from model.model import ...`)
> 2. 需要正确识别项目的包结构
> 3. 需要访问项目根目录下的 `gym_go` 模块
>
> 直接运行脚本文件会导致模块导入错误。

### 1. 训练新模型

```python
from model import AlphaGoZeroTrainer

# 创建训练器
trainer = AlphaGoZeroTrainer(
    board_size=9,
    num_epochs=100,
    games_per_epoch=20,
    checkpoint_dir='./checkpoints'
)

# 开始训练
trainer.train()
```

### 2. 加载预训练模型

```python
from model import GoNeuralNetwork
from gym_go import govars
import torch

# 创建模型（参数需与训练时保持一致）
model = GoNeuralNetwork(
    board_size=9,
    input_channels=govars.NUM_CHNLS,
    num_channels=32,
    num_res_blocks=6
)

# 加载权重
checkpoint = torch.load('checkpoints/latest_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

### 3. 使用 MCTS 进行搜索

```python
from model import MCTS, GoNeuralNetwork
from gym_go import gogame, govars

# 初始化模型（参数需与训练时保持一致）
model = GoNeuralNetwork(
    board_size=9,
    input_channels=govars.NUM_CHNLS,
    num_channels=32,
    num_res_blocks=6
)
mcts = MCTS(model, num_simulations=800)

# 创建游戏状态
state = gogame.init_game(board_size=9)

# 搜索最佳落子
action_probs = mcts.search(state)
best_action = max(action_probs, key=action_probs.get)
```

## 配置参数

### 训练参数

- `board_size`: 棋盘尺寸 (默认 9)
- `num_epochs`: 训练轮数 (默认 100)
- `games_per_epoch`: 每轮自我对弈局数 (默认 10)
- `batch_size`: 批处理大小 (默认 128)
- `learning_rate`: 学习率 (默认 0.001)
- `replay_buffer_size`: 经验回放缓冲区大小 (默认 500000)

### MCTS 参数

- `num_simulations`: 模拟次数 (默认 800)
- `c_puct`: UCB1 探索常数 (默认 1.0)
- `dirichlet_alpha`: 根节点噪声强度 (默认 0.3)
- `temperature`: 温度参数，控制策略随机性

## 检查点管理

训练过程中会自动保存检查点到 `checkpoints/` 目录：

- `checkpoint_X_YYYYMMDD_HHMMSS.pt`: 训练检查点
- `latest_model.pt`: 最新模型权重
- `training_history.json`: 训练历史和统计信息

## 性能优化

1. **GPU 加速**: 自动检测并使用 CUDA (如果可用)
2. **批处理**: 支持批量推理以提高效率
3. **内存管理**: 使用经验回放缓冲区管理训练数据
4. **并行化**: 支持多进程自我对弈 (可选)

## 依赖项

### 安装依赖

使用项目根目录的统一 `requirements.txt` 文件安装所有依赖：

```bash
# 在项目根目录执行
pip install -r requirements.txt
```

### 核心依赖

- **PyTorch >= 2.0.1**: 深度学习框架
- **NumPy >= 1.21.0**: 数值计算库
- **tqdm >= 4.62.0**: 进度条显示
- **gym_go**: 围棋游戏环境 (项目内部模块)

详细的依赖版本要求请参考项目根目录的 `requirements.txt` 文件。

## 注意事项

1. **内存使用**: 大的 `replay_buffer_size` 会消耗更多内存
2. **训练时间**: 完整训练可能需要数小时到数天
3. **模型收敛**: 建议监控训练损失和胜率变化
4. **检查点恢复**: 支持从检查点恢复训练

## 扩展功能

- 支持不同棋盘尺寸 (9×9, 13×13, 19×19)
- 可调整网络架构参数
- 支持分布式训练 (未来版本)
- 模型压缩和量化 (未来版本)