# copy from https://github.com/huangeddie/GymGo/blob/master/gym_go/govars.py  # 从GymGo项目复制的govars.py
ANYONE = None  # 任何人（未指定玩家）
NOONE = -1  # 没有人（无玩家）

BLACK = 0  # 黑棋编号
WHITE = 1  # 白棋编号
TURN_CHNL = 2  # 当前轮到谁下棋的通道编号
INVD_CHNL = 3  # 无效落子（包括打劫保护）的通道编号
PASS_CHNL = 4  # 上一步是否为pass的通道编号
DONE_CHNL = 5  # 游戏是否结束的通道编号

NUM_CHNLS = 6  # 总通道数
