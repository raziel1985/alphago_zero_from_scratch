import math
import numpy as np
import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

from gym_go import gogame as enviroment


@dataclass
class MCTNode:
    """
    蒙特卡洛树搜索节点类.
    该类表示MCTS搜索树中的一个节点, 存储节点的状态信息和统计数据.

    Attributes:
        prior: 节点的先验概率
        state: 节点对应的游戏状态
        parent: 父节点的引用
        children: 子节点字典, 键为动作坐标, 值为对应的子节点
        visit_count: 节点的访问次数
        value_sum: 节点的价值总和
        is_expanded: 节点是否已被扩展
    """
    prior: float
    state: Optional[np.ndarray] = None
    parent: Optional['MCTNode'] = None
    children: Dict[Tuple[int, int], 'MCTNode'] = None
    visit_count: int = 0
    value_sum: float = 0.0
    is_expanded: bool = False

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = {}
    
    def expand(self, state: np.ndarray, policy: np.ndarray) -> None:
        self.state = state
        self.is_expanded = True
        policy_size = len(policy)
        board_size = int(np.sqrt(policy_size - 1))
        for action in range(policy_size):
            if action == policy_size - 1:   # pass move 弃权
                child_action = (-1, -1)
            else:
                child_action = (action // board_size, action % board_size)
            self.children[child_action] = MCTNode(prior=policy[action], parent=self)

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct: float) -> Tuple[Tuple[int, int], 'MCTNode']:
        best_action = None
        best_child = None
        best_score = float('-inf')
        for action, child in self.children.items():
            # UCB公式: Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a)
            score = child.value + c_puct * child.prior * np.sqrt(self.visit_count) / (1 + child.visit_count)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child
    

class MCTS:
    """
    蒙特卡洛树搜索(MCTS)算法的实现.
    该类实现了AlphaGo Zero中使用的MCTS算法, 用于在围棋游戏中进行决策. 算法通过以下步骤工作:
    1. 选择: 从根节点开始, 使用UCB公式选择最有潜力的动作
    2. 扩展: 对叶子节点进行扩展, 并使用神经网络评估新节点
    3. 模拟: 使用神经网络评估叶子节点的局面
    4. 回溯: 更新搜索路径上所有节点的统计信息
    """

    def __init__(self, model: nn.Module, board_size: int, num_simulations: int = 800, c_puct: float = 2.0, device: str='cpu') -> None:
        self.model = model
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device

    def _normalize_policy(self, policy: np.ndarray, valid_moves: np.ndarray) -> np.ndarray:
        """归一化策略，过滤非法动作"""
        policy = policy * valid_moves
        policy_sum = np.sum(policy)
        if policy_sum > 0:
            return policy / policy_sum
        else:
            # 如果没有合法动作，全部概率赋值给pass动作
            normalized = np.zeros_like(policy)
            normalized[-1] = 1.0
            return normalized
    
    def _action_to_index(self, action) -> int:
        """将动作转换为索引"""
        return (self.board_size * self.board_size if action == (-1, -1) 
                else action[0] * self.board_size + action[1])
    
    def _evaluate_state(self, state: np.ndarray):
        """使用神经网络评估状态"""
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            policy, value = self.model(state_tensor)
            return policy[0].cpu().numpy(), value[0].item()

    def run(self, state: np.ndarray) -> np.ndarray:
        """
        执行蒙特卡洛树搜索.
        Args:
            state: 当前游戏状态, 形状为(6, board_size, board_size)的数组
        Returns:
            动作概率分布, 长度为board_size * board_size + 1的数组
        """
        # 初始化根节点
        root = MCTNode(prior=0)
        policy, _ = self._evaluate_state(state)
        valid_moves = enviroment.valid_moves(state)
        policy = self._normalize_policy(policy, valid_moves)
        # 每次进入run函数时的state(棋盘格局)不同，扩展的子节点会根据当前棋局的状态，探索剩余的有效空间
        root.expand(state, policy)

        # 执行MCTS模拟
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            current_state = state.copy()
            
            # 选择阶段：向下选择直到叶子节点
            while node.is_expanded and not enviroment.game_ended(current_state):
                action, node = node.select_child(self.c_puct)
                action_idx = self._action_to_index(action)
                # 验证动作合法性
                if not enviroment.valid_moves(current_state)[action_idx]:
                    action_idx = self.board_size * self.board_size  # 强制pass
                current_state = enviroment.next_state(current_state, action_idx)
                search_path.append(node)

            # 评估阶段
            if enviroment.game_ended(current_state):
                value = enviroment.winning(current_state)
            else:
                # AlphaGo Zero MCTS: 使用Neural Network Evaluation代替Rollout Simulaion
                new_policy, value = self._evaluate_state(current_state)
                if not node.is_expanded:
                    valid_moves = enviroment.valid_moves(current_state)
                    new_policy = self._normalize_policy(new_policy, valid_moves)
                    node.expand(current_state, new_policy)
            
            # 回溯阶段
            # 调试输出回溯路径，正常训练时可以不输出
            degbug_output = False
            if (degbug_output):
                move_sequence = []
                for i in range(1, len(search_path)):
                    current_node = search_path[i]
                    parent_node = search_path[i-1]
                    action_coord = None
                    for action, child in parent_node.children.items():
                        if child is current_node:
                            action_coord = action
                            break
                    if action_coord == (-1, -1):
                        move_sequence.append("Pass")
                    elif action_coord:
                        move_sequence.append(f"({action_coord[0]}, {action_coord[1]})")
                    else:
                        move_sequence.append("未知")
                if move_sequence:
                    moves_str = ", ".join(move_sequence)
                    print(f"模拟{_+1}: 深度{len(search_path)}, 落子{moves_str}, 价值{value:.3f}")
                else:
                    print(f"模拟{_+1}: 深度{len(search_path)}, 根节点, 价值{value:.3f}")

            for node in reversed(search_path):
                node.value_sum += value
                node.visit_count += 1
                value = -value

        # 计算最终动作概率
        action_probs = np.zeros(self.board_size * self.board_size + 1)
        for action, child in root.children.items():
            if action == (-1, -1):
                action_probs[-1] = child.visit_count    # pass动作的访问次数
            else:
                action_idx = self._action_to_index(action)
                action_probs[action_idx] = child.visit_count      
        # 归一化动作概率，过滤无效动作
        valid_moves = enviroment.valid_moves(state)  
        return self._normalize_policy(action_probs, valid_moves)
