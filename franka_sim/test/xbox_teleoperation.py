import time
from argparse import Action

import pygame
import numpy as np
import mujoco
import mujoco.viewer
from franka_sim import envs
from xbox_input import XboxInput




# 初始化机器人仿真环境
env = envs.PandaAssembleGearGymEnv(action_scale=(0.1, 1))
action_spec = env.action_space
xbox = XboxInput()
m = env.model
d = env.data

reset = False
KEY_SPACE = 32




# 用于重置环境
def key_callback(keycode):
    global reset
    if keycode == KEY_SPACE:
        reset = True


env.reset()
with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
    start = time.time()
    running = True
    while running and viewer.is_running():
        # 处理手柄输入
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN and event.button == 11:  # 停止按钮
                running = False
                print("退出")

        if reset:
            env.reset()
            reset = False
        else:
            # 获取手柄输入并计算控制动作
            action = xbox.get_action()
            print("action: ", action)

            # 让机器人仿真环境执行该动作
            step_start = time.time()
            env.step(action)
            viewer.sync()
            time_until_next_step = env.control_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

pygame.quit()
