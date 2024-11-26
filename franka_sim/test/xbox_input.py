import pygame
import numpy as np


class XboxInput:
    """
        XboxInput Class
        
        This class captures inputs from an Xbox controller and maps them to robotic control actions 
        suitable for the Franka Research 3 Robot simulation in MuJoCo. The inputs are processed to 
        ensure precision and include dead zones to filter out minor noise.
        
        Author: parzivar
        Date: 2024-11-30
        Version: 1.0
        
        Input Mapping:
        --------------
        1. Left Joystick:
           - Axis 0: Controls translation along the X-axis (scaled by 0.01, with a dead zone).
           - Axis 1: Controls translation along the Y-axis (scaled by 0.01, with a dead zone).
        
        2. Buttons for Z-axis Movement:
           - Y Button (Button 4): Moves the end-effector up along the Z-axis (fixed increment).
           - A Button (Button 0): Moves the end-effector down along the Z-axis (fixed increment).
        
        3. Right Joystick:
           - Axis 3: Controls pitch angle adjustment (scaled by 0.1, with a dead zone).
           - Axis 2: Controls roll angle adjustment (scaled by 0.1, with a dead zone).
        
        4. Trigger Inputs for Yaw Adjustment:
           - Left Trigger (Axis 4): Adjusts yaw angle to the left (input mapped from [-1, 1] to [0, 1]).
           - Right Trigger (Axis 5): Adjusts yaw angle to the right (input mapped from [-1, 1] to [0, 1]).
           - The combined input from the triggers is used for smooth yaw control.
        
        5. Gripper Controls:
           - X Button (Button 3): Opens the gripper (decreases gripper state by 0.1 per press).
           - B Button (Button 1): Closes the gripper (increases gripper state by 0.1 per press).
        
        Special Features:
        -----------------
        1. Dead Zone Detection:
           - Dead zones are applied to joystick inputs to avoid small unintentional movements.
           - Inputs below the dead zone threshold are treated as zero.
        
        2. Scaling:
           - Translational movements (X, Y, Z) are scaled for precision control.
           - Rotational movements (pitch, roll, yaw) are scaled independently.
           - Gripper control values remain unscaled for direct mapping.
        
        3. Input Preprocessing:
           - Trigger inputs are preprocessed to map the raw range of [-1, 1] to [0, 1], ensuring usability for control purposes.
        
        Usage:
        ------
        1. Initialize the XboxInput class.
        2. Call `get_action()` to retrieve the processed action array for the robot.
        3. Use the action array in the robot control step function.
    
    """
    def __init__(self):
        # 初始化 Pygame 和 Joystick
        pygame.init()
        pygame.joystick.init()

        # 检查是否有手柄连接
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("没有检测到手柄")
        else:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"已连接手柄: {self.joystick.get_name()}")

    def apply_dead_zone(self, value: float, threshold: float = 0.2) -> float:
        """
        应用死区检测。
        如果绝对值小于阈值，则返回 0，否则返回原始值。
        Args:
            value (float): 原始输入值
            threshold (float): 死区阈值
        Returns:
            float: 过滤后的值
        """
        return value if abs(value) >= threshold else 0.0

    def preprocess_trigger(self, value: float) -> float:
        """
        预处理扳机输入，将[-1, 1]映射到[0, 1]
        Args:
            value (float): 原始输入值（范围[-1, 1]）
        Returns:
            float: 处理后的值（范围[0, 1]）
        """
        return (value + 1) / 2  # 映射公式

    def get_action(self):
        """
        获取手柄输入，映射为控制动作
        Returns:
            np.ndarray: 包含 XYZ 移动和姿态控制的动作 [x, y, z, roll, pitch, yaw, gripper]
        """
        action = np.zeros(7)

        # 左摇杆控制 XY 平面移动，加入死区检测
        action[0] = self.apply_dead_zone(self.joystick.get_axis(0))  # 左摇杆X轴
        action[1] = -1 * self.apply_dead_zone(self.joystick.get_axis(1))  # 左摇杆Y轴

        # Y/A 按键控制 Z 轴移动
        if self.joystick.get_button(4):  # 按下Y
            action[2] = 1  # 向上
        elif self.joystick.get_button(0):  # 按下A
            action[2] = -1  # 向下

        # 右摇杆和扳机控制姿态，加入死区检测
        action[5] = -self.apply_dead_zone(self.joystick.get_axis(3)) * 0.5  # 修改：右摇杆X轴（俯仰）
        action[4] = self.apply_dead_zone(self.joystick.get_axis(2))  # 右摇杆Y轴（偏航）

        # 左右扳机叠加控制偏航 (轴4 和 轴5)，加入死区检测
        action[3] = self.preprocess_trigger(self.apply_dead_zone(self.joystick.get_axis(4))) - \
                    self.preprocess_trigger(self.apply_dead_zone(self.joystick.get_axis(5)))

        # 夹爪控制
        if self.joystick.get_button(3):  # 按下X
            action[6] = -0.1  # 张开
        elif self.joystick.get_button(1):  # 按下B
            action[6] = 0.1  # 闭合

        # 分别缩放不同部分
        action[:3] *= 0.01  # 缩放 XYZ 轴的移动
        action[3:6] *= 0.05  # 缩放姿态控制
        # action[6] 保持原始值，无需缩放

        return action
