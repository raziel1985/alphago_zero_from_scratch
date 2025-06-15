# AlphaGo Zero from Scratch

一个从零开始实现的 AlphaGo Zero 围棋AI项目，包含完整的神经网络训练、蒙特卡洛树搜索算法和网页对弈界面。

## 🎯 项目特点

- 🧠 **完整的 AlphaGo Zero 实现**: 包含神经网络、MCTS 搜索和自我对弈训练
- 🎮 **网页对弈界面**: 基于 Flask 的现代化网页应用，支持人机对弈
- 🔧 **模块化设计**: 清晰的代码结构，易于理解和扩展
- 📚 **详细文档**: 每个模块都有完整的说明文档
- 🚀 **易于部署**: 标准化的依赖管理和运行方式

## 📁 项目结构

```
alphago_zero_from_scratch/
├── README.md           # 项目主文档（本文件）
├── requirements.txt    # 项目统一依赖文件
├── __init__.py         # 项目包初始化
├── gym_go/             # 围棋游戏环境
│   ├── gogame.py       # 围棋游戏逻辑
│   ├── govars.py       # 游戏常量定义
│   └── state_utils.py  # 状态处理工具
├── model/              # AI 核心模块
│   ├── model.py        # 神经网络模型
│   ├── mcts.py         # 蒙特卡洛树搜索
│   ├── train.py        # 训练器
│   ├── checkpoints/    # 模型检查点
│   └── README.md       # AI 模块文档
└── web_app/            # 网页应用
    ├── app.py          # Flask 应用
    ├── templates/      # 前端模板
    └── README.md       # 网页应用文档
```

## 🚀 快速开始

### 环境要求

- Python 3.7+
- PyTorch 1.9.0+
- 现代浏览器（用于网页对弈）

### 安装依赖

```bash
# 1. 克隆项目
git clone <repository-url>
cd alphago_zero_from_scratch

# 2. 安装项目依赖（推荐使用虚拟环境）
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 训练 AI 模型

```bash
# 从项目根目录运行训练
python3 -m model.train
```

### 运行网页对弈

```bash
# 从项目根目录运行
PYTHONPATH=. python3 web_app/app.py
```

然后在浏览器中访问 `http://localhost:8080` 开始与 AI 对弈。


## 🧩 核心模块介绍

### 1. gym_go - 围棋游戏环境

提供完整的围棋游戏逻辑，包括：
- 棋盘状态管理
- 落子合法性检查
- 吃子和打劫规则
- 游戏结束判断

### 2. model - AI 核心

#### 神经网络模型 (`model.py`)
- 基于 ResNet 的卷积神经网络
- 双头输出：策略头 + 价值头
- 支持不同棋盘尺寸

#### 蒙特卡洛树搜索 (`mcts.py`)
- 实现 AlphaGo Zero 的 MCTS 算法
- UCB1 公式指导的树搜索
- 神经网络引导的叶子节点评估

#### 训练器 (`train.py`)
- 自我对弈数据生成
- 神经网络训练循环
- 模型评估和检查点管理

### 3. web_app - 网页对弈界面

- 基于 Flask 的后端 API
- 现代化的前端界面
- 实时游戏状态更新
- 支持人机对弈

## 🎮 使用指南

### 训练新模型

1. **开始训练**: `python3 -m model.train`
2. **监控进度**: 观察控制台输出的训练指标
3. **检查点**: 模型会自动保存到 `model/checkpoints/`
4. **评估模型**: 训练过程中会自动进行模型评估

### 网页对弈

1. **启动应用**: `python3 -m web_app.app`
2. **开始游戏**: 浏览器访问 `http://localhost:8080`
3. **人类落子**: 点击棋盘空位
4. **AI 落子**: 点击"AI 落子"按钮
5. **重新开始**: 点击"重新开始"按钮

### 自定义配置

可以修改以下参数来自定义训练和对弈：

**训练参数** (在 `model/train.py` 中):
- `board_size`: 棋盘大小 (默认 9)
- `num_epochs`: 训练轮数 (默认 100)
- `num_simulations`: MCTS 模拟次数 (默认 800)

**对弈参数** (在 `web_app/app.py` 中):
- `num_simulations`: AI 思考深度 (默认 400)
- `c_puct`: 探索参数 (默认 2.0)

## 📊 性能说明

- **训练时间**: 完整训练可能需要数小时到数天
- **内存使用**: 建议至少 4GB RAM
- **GPU 支持**: 自动检测并使用 CUDA（如果可用）
- **AI 强度**: 随训练进度逐步提升

## 🔧 故障排除

### 常见问题

1. **模块导入错误**
   ```bash
   # 确保从项目根目录运行
   python3 -m web_app.app
   ```

2. **端口占用**
   ```bash
   # 修改 web_app/app.py 中的端口号
   app.run(host='0.0.0.0', port=8081, debug=True)
   ```

3. **依赖版本冲突**
   ```bash
   # 使用虚拟环境
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **模型加载失败**
   - 检查 `model/checkpoints/` 目录是否存在
   - 确认模型文件格式正确

## 📚 详细文档

- [AI 模块文档](model/README.md) - 神经网络、MCTS 和训练的详细说明
- [网页应用文档](web_app/README.md) - 网页界面的使用和配置说明

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- AlphaGo Zero 论文作者
- PyTorch 团队
- 围棋开源社区

---

**开始你的围棋 AI 之旅吧！** 🎯🤖