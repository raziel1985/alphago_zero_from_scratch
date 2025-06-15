# Gym Go 模块

## 代码来源

本模块的代码来源于开源项目 [GymGo](https://github.com/huangeddie/GymGo)。

- **原项目地址**: https://github.com/huangeddie/GymGo
- **原作者**: huangeddie
- **许可证**: 请参考原项目的许可证条款

## 模块说明

本模块实现了围棋游戏的核心逻辑，包括：

### 文件结构

- `__init__.py` - 包初始化文件
- `govars.py` - 围棋游戏常量定义
- `gogame.py` - 围棋游戏主要逻辑实现
- `state_utils.py` - 游戏状态工具函数

### 主要功能

1. **游戏状态管理** (`gogame.py`)
   - 初始化棋盘状态
   - 计算下一步游戏状态
   - 批量处理游戏状态

2. **状态工具函数** (`state_utils.py`)
   - 计算无效落子位置
   - 处理围棋规则（如打劫、提子等）
   - 群组连接关系判断

3. **游戏常量** (`govars.py`)
   - 定义黑白棋子编号
   - 定义游戏状态通道
   - 其他围棋相关常量

## 使用说明

本模块作为围棋AI项目的基础组件，提供了完整的围棋游戏逻辑实现。可以与深度学习模型结合，用于训练和测试围棋AI。

## 致谢

感谢 [GymGo](https://github.com/huangeddie/GymGo) 项目提供的优秀围棋游戏实现。