# copy from https://github.com/huangeddie/GymGo/blob/master/gym_go/state_utils.py  # 从GymGo项目复制的state_utils.py
import numpy as np
from scipy import ndimage
from scipy.ndimage import measurements

from gym_go import govars

group_struct = np.array([[[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]],
                         [[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]],
                         [[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]]])  # 群组结构，用于标记棋子连接关系

surround_struct = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]])  # 用于判断上下左右邻居

neighbor_deltas = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # 四个方向的偏移量


def compute_invalid_moves(state: np.ndarray, player: int, ko_protect: tuple = None) -> np.ndarray:  # 计算无效落子位置
    """
    计算无效落子位置。
    参数：
        state (np.ndarray): 当前棋盘状态，形状为 (通道数, 棋盘行, 棋盘列)
        player (int): 当前玩家(0-黑棋, 1-白棋）
        ko_protect (tuple, 可选): 劫点坐标, 若无则为None
    返回：
        np.ndarray: 与棋盘同形状的布尔数组, True表示该位置无效
    从对手的视角更新无效落子
    1.) 对手不能在以下位置落子：
        i.) 如果该位置已有棋子
        i.) 如果该位置受劫保护
    2.) 对手可以在以下位置落子：
        i.) 如果该位置可以提子
    3.) 对手不能在以下位置落子：
        i.) 如果该位置与对手只有一个气的群组相邻，且不与具有多个气的其他群组相邻，并且被完全包围
        ii.) 如果该位置被我方棋子包围，且所有相关的群组都有超过一个气
    """
    # 获取所有棋子和空位
    all_pieces = np.sum(state[[govars.BLACK, govars.WHITE]], axis=0)  # 统计棋盘上所有棋子
    empties = 1 - all_pieces  # 计算空位
    # 初始化无效和有效数组
    possible_invalid_array = np.zeros(state.shape[1:])  # 可能无效的位置
    definite_valids_array = np.zeros(state.shape[1:])  # 确定有效的位置
    # 获取所有群组
    all_own_groups, num_own_groups = measurements.label(state[player])  # 己方群组
    all_opp_groups, num_opp_groups = measurements.label(state[1 - player])  # 对方群组
    expanded_own_groups = np.zeros((num_own_groups, *state.shape[1:]))  # 扩展己方群组到独立通道
    expanded_opp_groups = np.zeros((num_opp_groups, *state.shape[1:]))  # 扩展对方群组到独立通道
    # 将每个群组扩展为单独的通道
    for i in range(num_own_groups):
        expanded_own_groups[i] = all_own_groups == (i + 1)  # 标记每个己方群组
    for i in range(num_opp_groups):
        expanded_opp_groups[i] = all_opp_groups == (i + 1)  # 标记每个对方群组
    # 计算每个群组的"气": 气是与当前群组相邻的空位
    all_own_liberties = empties[np.newaxis] * ndimage.binary_dilation(expanded_own_groups, surround_struct[np.newaxis])  # 己方群组的气
    all_opp_liberties = empties[np.newaxis] * ndimage.binary_dilation(expanded_opp_groups, surround_struct[np.newaxis])  # 对方群组的气
    own_liberty_counts = np.sum(all_own_liberties, axis=(1, 2))  # 己方每个群组的气数
    opp_liberty_counts = np.sum(all_opp_liberties, axis=(1, 2))  # 对方每个群组的气数
    # 可能无效的位置包括：
    # 1. 己方群组有多个气的位置
    # 2. 对方群组只有一个气的位置
    possible_invalid_array += np.sum(all_own_liberties[own_liberty_counts > 1], axis=0)
    possible_invalid_array += np.sum(all_opp_liberties[opp_liberty_counts == 1], axis=0)
    # 确定有效的位置包括：
    # 1. 己方群组只有一个气的位置
    # 2. 对方群组有多个气的位置
    definite_valids_array += np.sum(all_own_liberties[own_liberty_counts == 1], axis=0)
    definite_valids_array += np.sum(all_opp_liberties[opp_liberty_counts > 1], axis=0)
    # 判断是否被完全包围
    surrounded = ndimage.convolve(all_pieces, surround_struct, mode='constant', cval=1) == 4  # 上下左右都有子
    # 所有无效的位置： 已经被占用的空间 + 上下左右都有子的情况下（可能无效的位置 - 确定有效的位置）
    invalid_moves = all_pieces + possible_invalid_array * (definite_valids_array == 0) * surrounded
    # 如果劫保护不为空，则将劫保护的位置设置为无效位置
    if ko_protect is not None:
        invalid_moves[ko_protect[0], ko_protect[1]] = 1  # 劫保护点设为无效
    return invalid_moves > 0  # 返回布尔数组，True为无效落子


def batch_compute_invalid_moves(batch_state: np.ndarray, batch_player: np.ndarray, batch_ko_protect: list) -> np.ndarray:  # 批量计算无效落子
    """
    批量计算无效落子位置。
    参数：
        batch_state (np.ndarray): 批量棋盘状态，形状为 (批量, 通道数, 棋盘行, 棋盘列)
        batch_player (np.ndarray): 每个状态的当前玩家，形状为 (批量,)
        batch_ko_protect (list): 每个状态的劫点坐标或None
    返回：
        np.ndarray: (批量, 棋盘行, 棋盘列) 的布尔数组, True表示无效
    从对手的视角批量更新无效落子
    1.) 对手不能在以下位置落子：
        i.) 如果该位置已有棋子
        i.) 如果该位置受劫保护
    2.) 对手可以在以下位置落子：
        i.) 如果该位置可以提子
    3.) 对手不能在以下位置落子：
        i.) 如果该位置与对手只有一个气的群组相邻，且不与具有多个气的其他群组相邻，并且被完全包围
        ii.) 如果该位置被我方棋子包围，且所有相关的群组都有超过一个气
    """
    batch_idcs = np.arange(len(batch_state))  # 批量索引

    # 获取所有棋子和空位
    batch_all_pieces = np.sum(batch_state[:, [govars.BLACK, govars.WHITE]], axis=1)  # 批量所有棋子
    batch_empties = 1 - batch_all_pieces  # 批量空位

    # 初始化无效和有效数组
    batch_possible_invalid_array = np.zeros(batch_state.shape[:1] + batch_state.shape[2:])  # 可能无效
    batch_definite_valids_array = np.zeros(batch_state.shape[:1] + batch_state.shape[2:])  # 确定有效

    # 获取所有群组
    batch_all_own_groups, _ = measurements.label(batch_state[batch_idcs, batch_player], group_struct)  # 己方群组
    batch_all_opp_groups, _ = measurements.label(batch_state[batch_idcs, 1 - batch_player], group_struct)  # 对方群组

    batch_data = enumerate(zip(batch_all_own_groups, batch_all_opp_groups, batch_empties))  # 批量数据
    for i, (all_own_groups, all_opp_groups, empties) in batch_data:
        own_labels = np.unique(all_own_groups)  # 己方群组标签
        opp_labels = np.unique(all_opp_groups)  # 对方群组标签
        own_labels = own_labels[np.nonzero(own_labels)]  # 去除0标签
        opp_labels = opp_labels[np.nonzero(opp_labels)]  # 去除0标签
        expanded_own_groups = np.zeros((len(own_labels), *all_own_groups.shape))  # 扩展己方群组
        expanded_opp_groups = np.zeros((len(opp_labels), *all_opp_groups.shape))  # 扩展对方群组

        # 将每个群组扩展为单独的通道
        for j, label in enumerate(own_labels):
            expanded_own_groups[j] = all_own_groups == label  # 标记己方群组

        for j, label in enumerate(opp_labels):
            expanded_opp_groups[j] = all_opp_groups == label  # 标记对方群组

        # 计算所有群组的气
        all_own_liberties = empties[np.newaxis] * ndimage.binary_dilation(expanded_own_groups,
                                                                          surround_struct[np.newaxis])  # 己方群组气
        all_opp_liberties = empties[np.newaxis] * ndimage.binary_dilation(expanded_opp_groups,
                                                                          surround_struct[np.newaxis])  # 对方群组气

        own_liberty_counts = np.sum(all_own_liberties, axis=(1, 2))  # 己方群组气数
        opp_liberty_counts = np.sum(all_opp_liberties, axis=(1, 2))  # 对方群组气数

        # 可能无效：己方群组有多个气，对方群组只有一个气
        # 确定有效：己方群组只有一个气，对方群组有多个气，或未被包围
        batch_possible_invalid_array[i] += np.sum(all_own_liberties[own_liberty_counts > 1], axis=0)
        batch_possible_invalid_array[i] += np.sum(all_opp_liberties[opp_liberty_counts == 1], axis=0)

        batch_definite_valids_array[i] += np.sum(all_own_liberties[own_liberty_counts == 1], axis=0)
        batch_definite_valids_array[i] += np.sum(all_opp_liberties[opp_liberty_counts > 1], axis=0)

    # 所有无效位置 = 已有棋子 + （可能无效-确定有效 且被包围）
    surrounded = ndimage.convolve(batch_all_pieces, surround_struct[np.newaxis], mode='constant', cval=1) == 4  # 被包围
    invalid_moves = batch_all_pieces + batch_possible_invalid_array * (batch_definite_valids_array == 0) * surrounded  # 无效

    # 劫保护
    for i, ko_protect in enumerate(batch_ko_protect):
        if ko_protect is not None:
            invalid_moves[i, ko_protect[0], ko_protect[1]] = 1  # 劫点无效
    return invalid_moves > 0  # 返回布尔数组


def update_pieces(state: np.ndarray, adj_locs: np.ndarray, player: int) -> list:  # 提子逻辑，移除无气的对方棋子
    """
    提子逻辑，移除无气的对方棋子。
    参数：
        state (np.ndarray): 当前棋盘状态
        adj_locs (np.ndarray): 新落子点的邻居坐标，形状为 (N, 2)
        player (int): 当前玩家编号
    返回：
        list: 被提掉的对方群组坐标列表
    """
    opponent = 1 - player  # 对手编号
    killed_groups = []  # 被提掉的群组

    all_pieces = np.sum(state[[govars.BLACK, govars.WHITE]], axis=0)  # 所有棋子
    empties = 1 - all_pieces  # 空位

    all_opp_groups, _ = ndimage.measurements.label(state[opponent])  # 对方所有群组

    # 遍历所有邻居的对方群组
    all_adj_labels = all_opp_groups[adj_locs[:, 0], adj_locs[:, 1]]  # 邻居群组标签
    all_adj_labels = np.unique(all_adj_labels)  # 唯一标签
    for opp_group_idx in all_adj_labels[np.nonzero(all_adj_labels)]:  # 遍历非0群组
        opp_group = all_opp_groups == opp_group_idx  # 当前群组
        liberties = empties * ndimage.binary_dilation(opp_group)  # 计算气
        if np.sum(liberties) <= 0:  # 没有气
            # 被提掉的群组
            opp_group_locs = np.argwhere(opp_group)  # 群组坐标
            state[opponent, opp_group_locs[:, 0], opp_group_locs[:, 1]] = 0  # 移除棋子
            killed_groups.append(opp_group_locs)  # 记录被提群组

    return killed_groups  # 返回被提群组


def batch_update_pieces(batch_non_pass: np.ndarray, batch_state: np.ndarray, batch_adj_locs: list, batch_player: np.ndarray) -> list:  # 批量提子
    """
    批量提子，移除无气的对方棋子。
    参数：
        batch_non_pass (np.ndarray): 非pass动作的批量索引
        batch_state (np.ndarray): 批量棋盘状态
        batch_adj_locs (list): 每个状态的邻居坐标列表
        batch_player (np.ndarray): 每个状态的当前玩家编号
    返回：
        list: 每个状态被提掉的对方群组坐标列表
    """
    batch_opponent = 1 - batch_player  # 批量对手
    batch_killed_groups = []  # 批量被提群组

    batch_all_pieces = np.sum(batch_state[:, [govars.BLACK, govars.WHITE]], axis=1)  # 批量所有棋子
    batch_empties = 1 - batch_all_pieces  # 批量空位

    batch_all_opp_groups, _ = ndimage.measurements.label(batch_state[batch_non_pass, batch_opponent],
                                                         group_struct)  # 批量对方群组

    batch_data = enumerate(zip(batch_all_opp_groups, batch_all_pieces, batch_empties, batch_adj_locs, batch_opponent))  # 批量数据
    for i, (all_opp_groups, all_pieces, empties, adj_locs, opponent) in batch_data:
        killed_groups = []  # 当前状态被提群组

        # 遍历所有邻居的对方群组
        all_adj_labels = all_opp_groups[adj_locs[:, 0], adj_locs[:, 1]]  # 邻居群组标签
        all_adj_labels = np.unique(all_adj_labels)  # 唯一标签
        for opp_group_idx in all_adj_labels[np.nonzero(all_adj_labels)]:  # 遍历非0群组
            opp_group = all_opp_groups == opp_group_idx  # 当前群组
            liberties = empties * ndimage.binary_dilation(opp_group)  # 计算气
            if np.sum(liberties) <= 0:  # 没有气
                # 被提掉的群组
                opp_group_locs = np.argwhere(opp_group)  # 群组坐标
                batch_state[batch_non_pass[i], opponent, opp_group_locs[:, 0], opp_group_locs[:, 1]] = 0  # 移除棋子
                killed_groups.append(opp_group_locs)  # 记录被提群组

        batch_killed_groups.append(killed_groups)  # 记录每个状态的被提群组

    return batch_killed_groups  # 返回批量被提群组


def adj_data(state: np.ndarray, action2d: np.ndarray, player: int) -> tuple:  # 获取指定位置的邻居信息和是否被包围
    """
    获取指定位置的邻居信息和是否被对方包围。
    参数：
        state (np.ndarray): 当前棋盘状态
        action2d (np.ndarray): 当前落子二维坐标 (行, 列)
        player (int): 当前玩家编号
    返回：
        tuple: (邻居坐标数组, 是否被对方包围的布尔值)
    """
    neighbors = neighbor_deltas + action2d  # 计算邻居坐标
    valid = (neighbors >= 0) & (neighbors < state.shape[1])  # 判断邻居是否越界
    valid = np.prod(valid, axis=1)  # 只保留合法邻居
    neighbors = neighbors[np.nonzero(valid)]  # 过滤非法邻居

    opp_pieces = state[1 - player]  # 对方棋子
    surrounded = (opp_pieces[neighbors[:, 0], neighbors[:, 1]] > 0).all()  # 是否被对方包围

    return neighbors, surrounded  # 返回邻居和包围信息


def batch_adj_data(batch_state: np.ndarray, batch_action2d: np.ndarray, batch_player: np.ndarray) -> tuple:  # 批量获取邻居和包围信息
    """
    批量获取邻居和包围信息。
    参数：
        batch_state (np.ndarray): 批量棋盘状态
        batch_action2d (np.ndarray): 每个状态的落子二维坐标
        batch_player (np.ndarray): 每个状态的当前玩家编号
    返回：
        tuple: (批量邻居坐标列表, 批量是否被包围的布尔列表)
    """
    batch_neighbors, batch_surrounded = [], []  # 批量邻居和包围
    for state, action2d, player in zip(batch_state, batch_action2d, batch_player):  # 遍历批量
        neighbors, surrounded = adj_data(state, action2d, player)  # 获取单个邻居
        batch_neighbors.append(neighbors)  # 加入列表
        batch_surrounded.append(surrounded)  # 加入列表
    return batch_neighbors, batch_surrounded  # 返回批量邻居和包围


def set_turn(state: np.ndarray) -> None:  # 切换当前回合
    """
    切换当前回合。
    参数：
        state (np.ndarray): 当前棋盘状态
    返回：
        None
    """
    state[govars.TURN_CHNL] = 1 - state[govars.TURN_CHNL]  # 0变1，1变0


def batch_set_turn(batch_state: np.ndarray) -> None:  # 批量切换回合
    """
    批量切换回合。
    参数：
        batch_state (np.ndarray): 批量棋盘状态
    返回：
        None
    """
    batch_state[:, govars.TURN_CHNL] = 1 - batch_state[:, govars.TURN_CHNL]  # 批量切换
