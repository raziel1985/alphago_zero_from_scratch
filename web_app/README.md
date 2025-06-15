# 围棋网页对弈应用

这是一个基于 Flask 的围棋网页应用，可以与使用 MCTS 算法的 AI 进行对弈。

## 功能特点

- 🎯 9x9 围棋棋盘
- 🤖 基于 MCTS 的 AI 对手
- 🎨 现代化的网页界面
- ⚡ 实时游戏状态更新
- 🔄 游戏重置功能

## 安装依赖

```bash
# 在项目根目录安装统一依赖
pip install -r requirements.txt
```

## 运行应用

```bash
# 从项目根目录运行
python3 -m web_app.app
```

服务器启动后，在浏览器中访问 `http://localhost:8080` 即可开始游戏。

### 运行说明
- 应用已移除 `sys.path.append` 语句，采用更标准的模块导入方式
- 必须从项目根目录运行，以确保能正确导入 `gym_go` 和 `model` 模块
- 使用 `PYTHONPATH=.` 环境变量让 Python 能找到项目根目录下的模块

## 游戏说明

### 基本规则
- 黑棋先行
- 点击棋盘空位落子
- 点击"AI 落子"按钮让 AI 下棋
- 连续两次弃权游戏结束

### 操作指南
1. **人类落子**: 直接点击棋盘上的空位
2. **AI 落子**: 点击右侧面板的"AI 落子"按钮
3. **重新开始**: 点击"重新开始"按钮重置游戏

### AI 特点
- 使用蒙特卡洛树搜索 (MCTS) 算法
- 支持加载预训练模型（如果存在）
- 默认进行 400 次模拟搜索
- 自动过滤非法落子

## 技术架构

### 后端 (Flask)
- `app.py`: 主应用文件，包含 API 路由和游戏逻辑
- 集成项目中的 MCTS 类和围棋环境
- 提供 RESTful API 接口

### 前端 (HTML/CSS/JavaScript)
- `templates/index.html`: 单页面应用
- 响应式设计，支持不同屏幕尺寸
- 实时更新游戏状态
- 美观的棋盘和棋子渲染

## API 接口

### GET /
**描述**: 返回游戏主页面  
**返回**: HTML 页面

### POST /api/reset
**描述**: 重置游戏到初始状态  
**请求参数**: 无  
**返回示例**:
```json
{
  "success": true,
  "board": [[0, 0, 0, ...], ...],
  "game_info": {
    "current_player": 0,
    "current_player_name": "黑棋",
    "game_ended": false,
    "winner": null,
    "winner_detail": null,
    "black_area": null,
    "white_area": null
  }
}
```

### POST /api/move
**描述**: 人类玩家落子  
**请求参数**:
```json
{
  "row": 0,
  "col": 0
}
```
**返回示例**:
```json
{
  "success": true,
  "message": "落子成功",
  "board": [[1, 0, 0, ...], ...],
  "game_info": {
    "current_player": 1,
    "current_player_name": "白棋",
    "game_ended": false,
    "winner": null,
    "winner_detail": null,
    "black_area": null,
    "white_area": null
  }
}
```
**错误返回**:
```json
{
  "success": false,
  "message": "非法落子"
}
```

### POST /api/ai_move
**描述**: AI 落子  
**请求参数**: 无  
**返回示例**:
```json
{
  "success": true,
  "message": "AI落子成功",
  "ai_move": {"row": 3, "col": 3},
  "board": [[1, 0, 0, ...], ...],
  "game_info": {
    "current_player": 0,
    "current_player_name": "黑棋",
    "game_ended": false,
    "winner": null,
    "winner_detail": null,
    "black_area": null,
    "white_area": null
  }
}
```
**AI 弃权时**:
```json
{
  "success": true,
  "message": "AI选择弃权",
  "ai_move": {"row": -1, "col": -1},
  "board": [[1, 0, 0, ...], ...],
  "game_info": {...}
}
```

### GET /api/board
**描述**: 获取当前棋盘状态  
**请求参数**: 无  
**返回示例**:
```json
{
  "board": [[0, 1, 2, ...], ...],
  "game_info": {
    "current_player": 0,
    "current_player_name": "黑棋",
    "game_ended": false,
    "winner": null,
    "winner_detail": null,
    "black_area": null,
    "white_area": null
  }
}
```

### POST /judge
**描述**: 判断当前棋盘胜负，返回黑白得分和胜者  
**请求参数**: 无  
**返回示例**:
```json
{
  "black": 45.0,
  "white": 39.5,
  "winner": "black"
}
```



## 文件结构

```
web_app/
├── app.py              # Flask 应用主文件
├── README.md          # 说明文档
└── templates/
    └── index.html     # 前端页面
```

## 注意事项

1. **模型加载**: 应用会尝试加载 `../model/checkpoints/latest_model.pt` 预训练模型，如果不存在则使用随机初始化模型
2. **性能**: AI 思考时间取决于 MCTS 模拟次数，可在 `app.py` 中调整 `num_simulations` 参数
3. **兼容性**: 需要 Python 3.7+ 和现代浏览器支持

## 自定义配置

可以在 `app.py` 中修改以下参数：
- `board_size`: 棋盘大小（默认 9）
- `num_simulations`: MCTS 模拟次数（默认 400）
- `c_puct`: UCB 公式中的探索参数（默认 2.0）

## 故障排除

1. **端口占用**: 如果 8080 端口被占用，可修改 `app.py` 最后一行的端口号
2. **模块导入错误**: 确保从项目根目录运行，并使用 `PYTHONPATH=.` 环境变量
3. **模型加载失败**: 检查模型文件路径和格式是否正确
4. **依赖问题**: 确保所有依赖包版本兼容
5. **相对导入错误**: 不要直接在 `web_app` 目录下运行 `python app.py`，必须从项目根目录运行

享受与 AI 的围棋对弈吧！🎮