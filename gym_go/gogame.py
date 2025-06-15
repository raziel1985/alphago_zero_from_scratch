# copy from https://github.com/huangeddie/GymGo/blob/master/gym_go/gogame.py  # 从GymGo项目复制的gogame.py
import numpy as np  
from scipy import ndimage
from sklearn import preprocessing

from gym_go import state_utils, govars

"""
游戏状态是一个numpy数组
* 所有值都是0或1

* 形状为 [NUM_CHNLS, SIZE, SIZE]

0 - 黑棋位置
1 - 白棋位置
2 - 当前轮到谁下棋 (0-黑棋, 1-白棋)
3 - 无效落子（包括打劫保护）
4 - 上一步是否为pass
5 - 游戏是否结束
"""


def init_state(size: int) -> np.ndarray:  # 创建并返回指定大小的初始游戏状态（空棋盘）
    """
    创建并返回指定大小的初始游戏状态（空棋盘）。
    参数：
        size (int): 棋盘大小
    返回：
        np.ndarray: 初始状态数组，形状为(NUM_CHNLS, size, size)
    """
    state = np.zeros((govars.NUM_CHNLS, size, size))  # 创建全零数组，表示空棋盘
    return state  # 返回初始状态


def batch_init_state(batch_size: int, board_size: int) -> np.ndarray:  # 创建并返回一批指定大小的初始游戏状态，用于批量处理
    """
    创建并返回一批指定大小的初始游戏状态，用于批量处理。
    参数：
        batch_size (int): 批量大小
        board_size (int): 棋盘大小
    返回：
        np.ndarray: 批量初始状态数组，形状为(batch_size, NUM_CHNLS, board_size, board_size)
    """
    batch_state = np.zeros((batch_size, govars.NUM_CHNLS, board_size, board_size))  # 创建批量全零数组
    return batch_state  # 返回批量初始状态


def next_state(state: np.ndarray, action1d: int, canonical: bool = False) -> np.ndarray:  # 根据给定的动作，计算下一个游戏状态
    """
    根据给定的动作，计算下一个游戏状态。
    参数：
        state (np.ndarray): 当前棋盘状态
        action1d (int): 动作的一维索引
        canonical (bool): 是否转为规范形式
    返回：
        np.ndarray: 新的棋盘状态
    """
    state = np.copy(state)  # 深拷贝，避免修改原始状态
    board_shape = state.shape[1:]  # 棋盘尺寸
    pass_idx = np.prod(board_shape)  # pass的索引编号是棋盘的大小上限
    passed = action1d == pass_idx  # 是否pass
    action2d = action1d // board_shape[0], action1d % board_shape[1]  # 将一维动作转为二维坐标
    player = turn(state)  # 获取当前玩家
    previously_passed = prev_player_passed(state)  # 上一手是否pass
    ko_protect = None  # 劫保护初始化
    if passed:  # 如果本手是pass
        state[govars.PASS_CHNL] = 1  # 将PASS通道所有元素都设为1
        if previously_passed:  # 连续两次pass
            state[govars.DONE_CHNL] = 1  # 游戏结束
    else:
        state[govars.PASS_CHNL] = 0  # 非pass，PASS通道清零
        assert state[govars.INVD_CHNL, action2d[0], action2d[1]] == 0, ("Invalid move", action2d)  # 检查落子是否合法
        state[player, action2d[0], action2d[1]] = 1  # 在指定位置放下当前玩家的棋子
        adj_locs, surrounded = state_utils.adj_data(state, action2d, player)  # 获取邻居和是否被包围
        killed_groups = state_utils.update_pieces(state, adj_locs, player)  # 提子
        if len(killed_groups) == 1 and surrounded:  # 如果只提掉一个且是单子且被包围
            killed_group = killed_groups[0]
            if len(killed_group) == 1:
                ko_protect = killed_group[0]  # 设置劫保护
    state[govars.INVD_CHNL] = state_utils.compute_invalid_moves(state, player, ko_protect)  # 更新无效落子
    state_utils.set_turn(state)  # 切换回合
    if canonical:
        state = canonical_form(state)  # 转为规范形式
    return state  # 返回新状态


def batch_next_states(batch_states: np.ndarray, batch_action1d: np.ndarray, canonical: bool = False) -> np.ndarray:  # next_state 的批量版本，同时处理多个状态和动作
    """
    批量计算下一个游戏状态。
    参数：
        batch_states (np.ndarray): 批量棋盘状态，形状为(batch, NUM_CHNLS, size, size)
        batch_action1d (np.ndarray): 批量动作索引，形状为(batch,)
        canonical (bool): 是否转为规范形式
    返回：
        np.ndarray: 批量新棋盘状态
    """
    batch_states = np.copy(batch_states)  # 深拷贝
    board_shape = batch_states.shape[2:]  # 棋盘尺寸
    pass_idx = np.prod(board_shape)  # pass的索引
    batch_pass = np.nonzero(batch_action1d == pass_idx)  # 找到所有pass动作
    batch_non_pass = np.nonzero(batch_action1d != pass_idx)[0]  # 非pass动作索引
    batch_prev_passed = batch_prev_player_passed(batch_states)  # 上一手是否pass
    batch_game_ended = np.nonzero(batch_prev_passed & (batch_action1d == pass_idx))  # 游戏结束的索引
    batch_action2d = np.array([batch_action1d[batch_non_pass] // board_shape[0],
                               batch_action1d[batch_non_pass] % board_shape[1]]).T  # 非pass动作转为二维
    batch_players = batch_turn(batch_states)  # 获取每个状态当前玩家
    batch_non_pass_players = batch_players[batch_non_pass]  # 非pass动作的玩家
    batch_ko_protect = np.empty(len(batch_states), dtype=object)  # 劫保护数组
    batch_states[batch_pass, govars.PASS_CHNL] = 1  # 所有pass动作的PASS通道设为1
    batch_states[batch_game_ended, govars.DONE_CHNL] = 1  # 游戏结束通道设为1
    batch_states[batch_non_pass, govars.PASS_CHNL] = 0  # 非pass动作的PASS通道设为0
    assert (batch_states[batch_non_pass, govars.INVD_CHNL, batch_action2d[:, 0], batch_action2d[:, 1]] == 0).all()  # 检查所有非pass动作是否合法
    batch_states[batch_non_pass, batch_non_pass_players, batch_action2d[:, 0], batch_action2d[:, 1]] = 1  # 批量落子
    batch_adj_locs, batch_surrounded = state_utils.batch_adj_data(batch_states[batch_non_pass], batch_action2d,
                                                                  batch_non_pass_players)  # 获取邻居和是否被包围
    batch_killed_groups = state_utils.batch_update_pieces(batch_non_pass, batch_states, batch_adj_locs,
                                                          batch_non_pass_players)  # 批量提子
    for i, (killed_groups, surrounded) in enumerate(zip(batch_killed_groups, batch_surrounded)):
        if len(killed_groups) == 1 and surrounded:
            killed_group = killed_groups[0]
            if len(killed_group) == 1:
                batch_ko_protect[batch_non_pass[i]] = killed_group[0]  # 设置劫保护
    batch_states[:, govars.INVD_CHNL] = state_utils.batch_compute_invalid_moves(batch_states, batch_players,
                                                                                batch_ko_protect)  # 批量更新无效落子
    state_utils.batch_set_turn(batch_states)  # 批量切换回合
    if canonical:
        batch_states = batch_canonical_form(batch_states)  # 转为规范形式
    return batch_states  # 返回批量新状态


def invalid_moves(state: np.ndarray) -> np.ndarray:  # 返回一个二进制向量，表示哪些动作是无效的
    """
    返回一个二进制向量，表示哪些动作是无效的。
    参数：
        state (np.ndarray): 当前棋盘状态
    返回：
        np.ndarray: 一维二进制向量，1表示无效动作
    """
    # 如果游戏结束，所有动作都无效
    if game_ended(state):
        return np.zeros(action_size(state))  # 返回全0
    return np.append(state[govars.INVD_CHNL].flatten(), 0)  # 否则返回无效动作向量


def valid_moves(state: np.ndarray) -> np.ndarray:  # 返回一个二进制向量，表示哪些动作是有效的
    """
    返回一个二进制向量，表示哪些动作是有效的。
    参数：
        state (np.ndarray): 当前棋盘状态
    返回：
        np.ndarray: 一维二进制向量，1表示有效动作
    """
    return 1 - invalid_moves(state)  # 有效动作=1-无效动作


def batch_invalid_moves(batch_state: np.ndarray) -> np.ndarray:  # 批量获取无效动作
    """
    批量获取无效动作。
    参数：
        batch_state (np.ndarray): 批量棋盘状态
    返回：
        np.ndarray: (batch, 动作数) 的二进制向量
    """
    n = len(batch_state)  # 批量大小
    batch_invalid_moves_bool = batch_state[:, govars.INVD_CHNL].reshape(n, -1)  # 展平成二维
    batch_invalid_moves_bool = np.append(batch_invalid_moves_bool, np.zeros((n, 1)), axis=1)  # 添加pass动作
    return batch_invalid_moves_bool  # 返回批量无效动作


def batch_valid_moves(batch_state: np.ndarray) -> np.ndarray:  # 批量获取有效动作
    """
    批量获取有效动作。
    参数：
        batch_state (np.ndarray): 批量棋盘状态
    返回：
        np.ndarray: (batch, 动作数) 的二进制向量
    """
    return 1 - batch_invalid_moves(batch_state)  # 有效=1-无效


def children(state: np.ndarray, canonical: bool = False, padded: bool = True) -> np.ndarray:  # 生成当前状态的所有可能的后继状态（子节点）
    """
    生成当前状态的所有可能的后继状态（子节点）。
    参数：
        state (np.ndarray): 当前棋盘状态
        canonical (bool): 是否转为规范形式
        padded (bool): 是否填充无效动作
    返回：
        np.ndarray: 所有子节点状态，形状为(动作数, NUM_CHNLS, size, size)
    """
    valid_moves_bool = valid_moves(state)  # 获取有效动作
    n = len(valid_moves_bool)  # 动作总数
    valid_move_idcs = np.argwhere(valid_moves_bool).flatten()  # 有效动作索引
    batch_states = np.tile(state[np.newaxis], (len(valid_move_idcs), 1, 1, 1))  # 批量复制状态
    children = batch_next_states(batch_states, valid_move_idcs, canonical)  # 生成所有后继状态
    if padded:
        padded_children = np.zeros((n, *state.shape))  # 创建填充数组
        padded_children[valid_move_idcs] = children  # 有效动作填充
        children = padded_children  # 无效动作为零
    return children  # 返回所有子节点


def action_size(state: np.ndarray = None, board_size: int = None) -> int:  # 计算动作空间的大小（棋盘位置数量 + 1 个 pass 动作）
    """
    计算动作空间的大小（棋盘位置数量 + 1 个 pass 动作）。
    参数：
        state (np.ndarray, 可选): 当前棋盘状态
        board_size (int, 可选): 棋盘大小
    返回：
        int: 动作总数
    """
    if state is not None:
        m, n = state.shape[1:]
    elif board_size is not None:
        m, n = board_size, board_size
    else:
        raise RuntimeError('No argument passed')  # 没有传入参数报错
    return m * n + 1  # 返回动作总数


def prev_player_passed(state: np.ndarray) -> bool:  # 检查上一个玩家是否选择了 pass
    """
    检查上一个玩家是否选择了 pass。
    参数：
        state (np.ndarray): 当前棋盘状态
    返回：
        bool: True表示上一步为pass
    """
    return np.max(state[govars.PASS_CHNL] == 1) == 1  # 返回是否pass


def batch_prev_player_passed(batch_state: np.ndarray) -> np.ndarray:  # 批量检查上一个玩家是否pass
    """
    批量检查上一个玩家是否pass。
    参数：
        batch_state (np.ndarray): 批量棋盘状态
    返回：
        np.ndarray: (batch,) 布尔数组
    """
    return np.max(batch_state[:, govars.PASS_CHNL], axis=(1, 2)) == 1  # 返回批量pass


def game_ended(state: np.ndarray) -> int:  # 检查游戏是否结束。当两名玩家连续 pass 时，游戏结束
    """
    检查游戏是否结束。当两名玩家连续 pass 时，游戏结束。
    参数：
        state (np.ndarray): 当前棋盘状态
    返回：
        int: 0/1，0表示未结束，1表示已结束
    """
    m, n = state.shape[1:]
    return int(np.count_nonzero(state[govars.DONE_CHNL] == 1) == m * n)  # 全部DONE_CHNL为1则结束


def batch_game_ended(batch_state: np.ndarray) -> np.ndarray:  # 批量检查游戏是否结束
    """
    批量检查游戏是否结束。
    参数：
        batch_state (np.ndarray): 批量棋盘状态
    返回：
        np.ndarray: (batch,) 0/1数组
    """
    return np.max(batch_state[:, govars.DONE_CHNL], axis=(1, 2))  # 返回批量结束标志


def winning(state: np.ndarray, komi: float = 0) -> int:  # 根据区域和贴目（komi）判断胜负
    """
    根据区域和贴目（komi）判断胜负。
    参数：
        state (np.ndarray): 当前棋盘状态
        komi (float): 贴目
    返回：
        int: 1黑胜，-1白胜，0平
    """
    black_area, white_area = areas(state)  # 计算黑白面积
    area_difference = black_area - white_area  # 面积差
    komi_correction = area_difference - komi  # 贴目修正
    return np.sign(komi_correction)  # 返回胜负（1黑胜，-1白胜，0平）


def batch_winning(state: np.ndarray, komi: float = 0) -> np.ndarray:  # 批量判断胜负
    """
    批量判断胜负。
    参数：
        state (np.ndarray): 批量棋盘状态
        komi (float): 贴目
    返回：
        np.ndarray: (batch,) 1黑胜，-1白胜，0平
    """
    batch_black_area, batch_white_area = batch_areas(state)  # 批量面积
    batch_area_difference = batch_black_area - batch_white_area  # 批量面积差
    batch_komi_correction = batch_area_difference - komi  # 批量贴目修正
    return np.sign(batch_komi_correction)  # 批量胜负


def turn(state: np.ndarray) -> int:  # 返回当前轮到哪个玩家
    """
    返回当前轮到哪个玩家。
    参数：
        state (np.ndarray): 当前棋盘状态
    返回：
        int: 当前玩家编号 (govars.BLACK/govars.WHITE)
    """
    return int(np.max(state[govars.TURN_CHNL]))  # 返回当前玩家


def batch_turn(batch_state: np.ndarray) -> np.ndarray:  # 批量获取当前轮到谁下棋
    """
    批量获取当前轮到谁下棋。
    参数：
        batch_state (np.ndarray): 批量棋盘状态
    返回：
        np.ndarray: (batch,) 当前玩家编号数组
    """
    return np.max(batch_state[:, govars.TURN_CHNL], axis=(1, 2)).astype(np.int)  # 返回每个状态的当前玩家编号


def liberties(state: np.ndarray) -> tuple:  # 计算黑棋和白棋的气（liberty）位置
    """
    计算黑棋和白棋的气（liberty）位置。
    参数：
        state (np.ndarray): 当前棋盘状态
    返回：
        tuple: (黑棋气的布尔数组, 白棋气的布尔数组)
    """
    blacks = state[govars.BLACK]  # 黑棋位置
    whites = state[govars.WHITE]  # 白棋位置
    all_pieces = np.sum(state[[govars.BLACK, govars.WHITE]], axis=0)  # 所有棋子

    liberty_list = []  # 存储每方的气
    for player_pieces in [blacks, whites]:  # 遍历黑白棋
        liberties = ndimage.binary_dilation(player_pieces, state_utils.surround_struct)  # 膨胀找出气
        liberties *= (1 - all_pieces).astype(np.bool)  # 只保留空位
        liberty_list.append(liberties)  # 加入列表

    return liberty_list[0], liberty_list[1]  # 返回黑棋和白棋的气


def num_liberties(state: np.ndarray) -> tuple:  # 计算黑棋和白棋的气的数量
    """
    计算黑棋和白棋的气的数量。
    参数：
        state (np.ndarray): 当前棋盘状态
    返回：
        tuple: (黑棋气数, 白棋气数)
    """
    black_liberties, white_liberties = liberties(state)  # 获取气的位置
    black_liberties = np.count_nonzero(black_liberties)  # 黑棋气数
    white_liberties = np.count_nonzero(white_liberties)  # 白棋气数
    return black_liberties, white_liberties  # 返回气数


def areas(state: np.ndarray) -> tuple:
    """
    计算黑棋和白棋的领地面积。
    参数：
        state (np.ndarray): 当前棋盘状态
    返回：
        tuple: (黑棋总面积, 白棋总面积)
    """
    all_pieces = np.sum(state[[govars.BLACK, govars.WHITE]], axis=0)  # 所有棋子
    empties = 1 - all_pieces  # 空位
    empty_labels, num_empty_areas = ndimage.measurements.label(empties)  # 标记所有空区域
    black_area, white_area = np.sum(state[govars.BLACK]), np.sum(state[govars.WHITE])  # 初始面积
    for label in range(1, num_empty_areas + 1):  # 遍历每个空区域
        empty_area = empty_labels == label  # 当前空区域
        neighbors = ndimage.binary_dilation(empty_area)  # 找邻居
        black_claim = False  # 是否被黑棋包围
        white_claim = False  # 是否被白棋包围
        if (state[govars.BLACK] * neighbors > 0).any():  # 有黑棋邻居
            black_claim = True
        if (state[govars.WHITE] * neighbors > 0).any():  # 有白棋邻居
            white_claim = True
        if black_claim and not white_claim:  # 只被黑棋包围
            black_area += np.sum(empty_area)
        elif white_claim and not black_claim:  # 只被白棋包围
            white_area += np.sum(empty_area)
    return black_area, white_area  # 返回面积


def batch_areas(batch_state: np.ndarray) -> tuple:
    """
    批量计算黑白面积。
    参数：
        batch_state (np.ndarray): 批量棋盘状态
    返回：
        tuple: (黑棋面积数组, 白棋面积数组)
    """
    black_areas, white_areas = [], []  # 存储结果
    for state in batch_state:
        ba, wa = areas(state)  # 计算单个状态面积
        black_areas.append(ba)
        white_areas.append(wa)
    return np.array(black_areas), np.array(white_areas)  # 返回批量面积


def canonical_form(state: np.ndarray) -> np.ndarray:
    """
    将状态转换为规范形式（始终从黑棋视角）。
    参数：
        state (np.ndarray): 当前棋盘状态
    返回：
        np.ndarray: 规范化后的棋盘状态
    """
    state = np.copy(state)  # 深拷贝
    if turn(state) == govars.WHITE:  # 如果当前是白棋回合
        channels = np.arange(govars.NUM_CHNLS)
        channels[govars.BLACK] = govars.WHITE
        channels[govars.WHITE] = govars.BLACK
        state = state[channels]  # 交换黑白通道
        state_utils.set_turn(state)  # 切换回合
    return state  # 返回规范状态


def batch_canonical_form(batch_state: np.ndarray) -> np.ndarray:
    """
    批量规范化。
    参数：
        batch_state (np.ndarray): 批量棋盘状态
    返回：
        np.ndarray: 批量规范化后的棋盘状态
    """
    batch_state = np.copy(batch_state)  # 深拷贝
    batch_player = batch_turn(batch_state)  # 获取每个状态当前玩家
    white_players_idcs = np.nonzero(batch_player == govars.WHITE)[0]  # 找到白棋回合的索引
    channels = np.arange(govars.NUM_CHNLS)
    channels[govars.BLACK] = govars.WHITE
    channels[govars.WHITE] = govars.BLACK
    for i in white_players_idcs:
        batch_state[i] = batch_state[i, channels]  # 交换黑白通道
        batch_state[i, govars.TURN_CHNL] = 1 - batch_player[i]  # 切换回合
    return batch_state  # 返回批量规范状态


def random_symmetry(image: np.ndarray) -> np.ndarray:
    """
    生成棋盘状态的随机对称变换。
    参数：
        image (np.ndarray): (C, BOARD_SIZE, BOARD_SIZE) 的numpy数组, C为通道数
    返回：
        np.ndarray: 随机对称变换后的棋盘
    """
    orientation = np.random.randint(0, 8)  # 随机选择一种对称方式
    if (orientation >> 0) % 2:
        image = np.flip(image, 2)  # 水平翻转
    if (orientation >> 1) % 2:
        image = np.flip(image, 1)  # 垂直翻转
    if (orientation >> 2) % 2:
        image = np.rot90(image, axes=(1, 2))  # 旋转90度
    return image  # 返回变换后的棋盘


def all_symmetries(image: np.ndarray) -> list:
    """
    生成棋盘状态的所有对称变换。
    参数：
        image (np.ndarray): (C, BOARD_SIZE, BOARD_SIZE) 的numpy数组, C为通道数
    返回：
        list: 包含8种对称变换的棋盘状态列表
    """
    symmetries = []  # 存储所有对称变换
    for i in range(8):
        x = image
        if (i >> 0) % 2:
            x = np.flip(x, 2)  # 水平翻转
        if (i >> 1) % 2:
            x = np.flip(x, 1)  # 垂直翻转
        if (i >> 2) % 2:
            x = np.rot90(x, axes=(1, 2))  # 旋转90度
        symmetries.append(x)  # 加入列表
    return symmetries  # 返回所有对称变换


def random_weighted_action(move_weights: np.ndarray) -> int:
    """
    根据给定的权重随机选择一个动作。
    参数：
        move_weights (np.ndarray): 动作权重数组，形状为(动作数,)
    返回：
        int: 采样得到的动作索引
    """
    move_weights = preprocessing.normalize(move_weights[np.newaxis], norm='l1')  # 归一化权重
    return np.random.choice(np.arange(len(move_weights[0])), p=move_weights[0])  # 按概率采样


def random_action(state: np.ndarray) -> int:
    """
    从当前状态的有效动作中随机选择一个。
    参数：
        state (np.ndarray): 当前棋盘状态
    返回：
        int: 随机采样得到的动作索引
    """
    invalid_moves = state[govars.INVD_CHNL].flatten()  # 获取无效动作
    invalid_moves = np.append(invalid_moves, 0)  # 添加pass动作
    move_weights = 1 - invalid_moves  # 有效动作权重为1
    return random_weighted_action(move_weights)  # 随机采样


def str(state: np.ndarray) -> str:
    """
    将游戏状态转换为字符串表示，用于打印和显示。
    参数：
        state (np.ndarray): 当前棋盘状态
    返回：
        str: 棋盘字符串
    """
    board_str = ''  # 棋盘字符串
    size = state.shape[1]  # 棋盘大小
    board_str += '\t'
    for i in range(size):
        board_str += '{}'.format(i).ljust(2, ' ')
    board_str += '\n'
    for i in range(size):
        board_str += '{}\t'.format(i)
        for j in range(size):
            if state[0, i, j] == 1:
                board_str += '○'
                if j != size - 1:
                    if i == 0 or i == size - 1:
                        board_str += '═'
                    else:
                        board_str += '─'
            elif state[1, i, j] == 1:
                board_str += '●'
                if j != size - 1:
                    if i == 0 or i == size - 1:
                        board_str += '═'
                    else:
                        board_str += '─'
            else:
                if i == 0:
                    if j == 0:
                        board_str += '╔═'
                    elif j == size - 1:
                        board_str += '╗'
                    else:
                        board_str += '╤═'
                elif i == size - 1:
                    if j == 0:
                        board_str += '╚═'
                    elif j == size - 1:
                        board_str += '╝'
                    else:
                        board_str += '╧═'
                else:
                    if j == 0:
                        board_str += '╟─'
                    elif j == size - 1:
                        board_str += '╢'
                    else:
                        board_str += '┼─'
        board_str += '\n'
    black_area, white_area = areas(state)  # 计算面积
    done = game_ended(state)  # 是否结束
    ppp = prev_player_passed(state)  # 上一步是否pass
    t = turn(state)  # 当前玩家
    if done:
        game_state = 'END'
    elif ppp:
        game_state = 'PASSED'
    else:
        game_state = 'ONGOING'
    board_str += '\tTurn: {}, Game State (ONGOING|PASSED|END): {}\n'.format('BLACK' if t == 0 else 'WHITE', game_state)
    board_str += '\tBlack Area: {}, White Area: {}\n'.format(int(black_area), int(white_area))
    return board_str  # 返回棋盘字符串
