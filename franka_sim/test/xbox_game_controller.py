import pygame


# 初始化 Pygame 和 Joystick
pygame.init()
pygame.joystick.init()

# 检查是否有手柄连接
if pygame.joystick.get_count() == 0:
    print("没有检测到手柄")
else:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"已连接手柄: {joystick.get_name()}")

# 循环检测手柄输入
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.JOYBUTTONDOWN and event.button == 11:
            running = False 
            print("退出")
        
        # 检测按键按下
        if event.type == pygame.JOYBUTTONDOWN:
            print(f"按键 {event.button} 按下")

        # 检测摇杆移动
        if event.type == pygame.JOYAXISMOTION:
            axis_value = joystick.get_axis(event.axis)
            print(f"摇杆 {event.axis} 移动，值: {axis_value}")

pygame.quit()
