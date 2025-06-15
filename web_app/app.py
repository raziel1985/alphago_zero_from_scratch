from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
import os

from gym_go import gogame as environment
from gym_go import govars
from model.model import GoNeuralNetwork
from model.mcts import MCTS

app = Flask(__name__)
CORS(app)  # 允许跨域请求

class GoGameServer:
    def __init__(self, board_size=9):
        self.board_size = board_size
        self.reset_game()
        
        # 初始化AI模型和MCTS
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 使用与train.py相同的模型参数
        from gym_go import govars
        self.model = GoNeuralNetwork(
            board_size=board_size, 
            input_channels=govars.NUM_CHNLS, 
            num_channels=32, 
            num_res_blocks=6
        )
        
        # 尝试加载预训练模型
        checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     'model', 'checkpoints', 'latest_model.pt')
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"加载预训练模型: {checkpoint_path}")
            except Exception as e:
                print(f"加载模型失败: {e}，使用随机初始化模型")
        else:
            print("未找到预训练模型，使用随机初始化模型")
            
        self.model.to(self.device)
        self.model.eval()
        
        # 初始化MCTS
        self.mcts = MCTS(self.model, board_size, num_simulations=400, device=self.device)
        
    def reset_game(self):
        """重置游戏"""
        self.state = environment.init_state(self.board_size)
        self.game_history = []
        
    def get_current_player(self):
        """获取当前玩家"""
        return int(self.state[govars.TURN_CHNL, 0, 0])
        
    def is_valid_move(self, row, col):
        """检查落子是否合法"""
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return False
        action_idx = row * self.board_size + col
        valid_moves = environment.valid_moves(self.state)
        return valid_moves[action_idx] == 1
        
    def make_move(self, row, col):
        """人类玩家落子"""
        if not self.is_valid_move(row, col):
            return False, "非法落子"
            
        action_idx = row * self.board_size + col
        self.state = environment.next_state(self.state, action_idx)
        self.game_history.append((row, col, self.get_current_player()))
        return True, "落子成功"
        
    def make_ai_move(self):
        """AI落子"""
        if environment.game_ended(self.state):
            return None, None, "游戏已结束"
            
        # 使用MCTS获取最佳动作
        action_probs = self.mcts.run(self.state)
        # 选择概率最高的合法动作
        action_idx = np.argmax(action_probs)
        if action_idx == self.board_size * self.board_size:
            # Pass动作
            self.state = environment.next_state(self.state, action_idx)
            return -1, -1, "AI选择弃权"
        else:
            row = int(action_idx // self.board_size)
            col = int(action_idx % self.board_size)
            self.state = environment.next_state(self.state, action_idx)
            self.game_history.append((row, col, self.get_current_player()))
            return row, col, "AI落子成功"
            
    def get_board_state(self):
        """获取当前棋盘状态"""
        board = np.zeros((self.board_size, self.board_size), dtype=int)
        # 0: 空位, 1: 黑棋, 2: 白棋
        board[self.state[govars.BLACK] == 1] = 1
        board[self.state[govars.WHITE] == 1] = 2
        return board.tolist()
        
    def get_game_info(self):
        """获取游戏信息"""
        current_player = self.get_current_player()
        game_ended = environment.game_ended(self.state)
        winner = None
        winner_detail = None
        
        if game_ended:
            # 计算领地和胜负
            black_area, white_area = environment.areas(self.state)
            komi = 6.5  # 白棋贴目6.5子
            result = environment.winning(self.state, komi)
            
            if result == 1:
                winner = "黑棋"
                winner_detail = f"黑棋获胜 (黑:{black_area:.1f} vs 白:{white_area:.1f}+{komi})"
            elif result == -1:
                winner = "白棋"
                winner_detail = f"白棋获胜 (黑:{black_area:.1f} vs 白:{white_area:.1f}+{komi})"
            else:
                winner = "平局"
                winner_detail = f"平局 (黑:{black_area:.1f} vs 白:{white_area:.1f}+{komi})"
                
        return {
            'current_player': current_player,
            'current_player_name': '黑棋' if current_player == 0 else '白棋',
            'game_ended': bool(game_ended),
            'winner': winner,
            'winner_detail': winner_detail,
            'black_area': black_area if game_ended else None,
            'white_area': white_area if game_ended else None
        }

# 全局游戏实例
game_server = GoGameServer()

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/reset', methods=['POST'])
def reset_game():
    """重置游戏"""
    game_server.reset_game()
    return jsonify({
        'success': True,
        'board': game_server.get_board_state(),
        'game_info': game_server.get_game_info()
    })

@app.route('/api/move', methods=['POST'])
def make_move():
    """人类玩家落子"""
    data = request.json
    row = data.get('row')
    col = data.get('col')
    
    if row is None or col is None:
        return jsonify({'success': False, 'message': '缺少行列参数'})
        
    success, message = game_server.make_move(row, col)
    
    response = {
        'success': success,
        'message': message,
        'board': game_server.get_board_state(),
        'game_info': game_server.get_game_info()
    }
    
    return jsonify(response)

@app.route('/api/ai_move', methods=['POST'])
def ai_move():
    """AI落子"""
    row, col, message = game_server.make_ai_move()
    
    response = {
        'success': True,
        'message': message,
        'ai_move': {'row': row, 'col': col},
        'board': game_server.get_board_state(),
        'game_info': game_server.get_game_info()
    }
    
    return jsonify(response)

@app.route('/api/board', methods=['GET'])
def get_board():
    """获取当前棋盘状态"""
    return jsonify({
        'board': game_server.get_board_state(),
        'game_info': game_server.get_game_info()
    })

@app.route('/judge', methods=['POST'])
def judge():
    """
    判断当前棋盘胜负，返回黑白得分和胜者
    """
    # 获取当前状态下的黑白地盘
    black_area, white_area = environment.areas(game_server.state)
    komi = 6.5
    black_score = float(black_area)
    white_score = float(white_area + komi)
    # 判断胜负
    if black_score > white_score:
        winner = "black"
    elif white_score > black_score:
        winner = "white"
    else:
        winner = "draw"
    return jsonify({
        "black": black_score,
        "white": white_score,
        "winner": winner
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)