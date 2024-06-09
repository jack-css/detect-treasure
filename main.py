import mkroad as MkRoad
import numpy as np
import tracking
import time
from Emakefun_MotorHAT import Emakefun_MotorHAT, Emakefun_Servo



def FindMinRoad(start, point_flag, step):
    print(step)
    step = step[start]
    step[start] = 500
    print(len(point_flag))
    for i in range(len(point_flag) - 1):
        print(i)
        if point_flag[i] == 1:
            step[i] = 500
    print(step)
    end_appoint = step.index(min(step))
    print(step[end_appoint])
    return end_appoint


# 寻找两个宝藏的对应点
def DealBao(point_flag, c_zone, index):
    for i in range(len(c_zone) - 1):
        if c_zone[i][0] + c_zone[index][0] == 20 and c_zone[i][1] + c_zone[index][1] == 20:
            point_flag[i] = 1
            return point_flag


def MarkFour(index, c_zone):
    x_flag = c_zone[index][0] - 10
    y_flag = c_zone[index][1] - 10
    c_zone[index] = [0, 0]
    for i in range(len(c_zone)):
        if ((c_zone[i][0] - 10) * x_flag) > 0 and ((c_zone[i][1] - 10) * y_flag > 0):
            return i


def main():
    # 将摄像头180度舵机设为0度
    init_camera_rotion(90)
    [road_all_turn, c_zone, step] = MkRoad.Mkroad()
    init_camera_rotion(105)
    #write(road_all_turn)
    # 队伍颜色
    # blue 2
    # red 1
    group_color = 1
    for i in range(len(road_all_turn)):
        j = i
        if road_all_turn[i][j] == 0:
            j = j + 1
        while j < len(road_all_turn):
            temp_old = road_all_turn[i][j]
            temp = []
            for k in range(len(temp_old) - 1, -1, -1):
                temp.append(temp_old[k])
            road_all_turn[len(road_all_turn) - i - 1][len(road_all_turn) - j - 1] = temp
            j = j + 1
    print(road_all_turn[5][2])
    # print(c_zone)
    # print(step)
    # 15.1 to 13.19
    # [[15, 1], [19, 1], [19, 9], [16, 9], [15, 9]], [[5, 5], [1, 5], [1, 9], [3, 9], [3, 11], [1, 11], [1, 13], [3, 13]
    # [3, 15], [5, 15], [5, 13], [11, 13], [11, 15], [13, 15], [13, 13],
    # [15, 13], [15, 10], [15, 9]]
    # c_zone = [[5, 11], [5, 19], [7, 1], [15, 15], [15, 9], [15, 1], [5, 5], [13, 19]]
    # print('zong', road_all_turn)
    # 起点与终点为1，宝藏点为0
    point_flag = [0 for i in range(len(c_zone) - 2)]
    point_flag.append(1)
    point_flag.append(1)
    point_flag = list(point_flag)
    index = FindMinRoad(8, point_flag, step)
    # write(turns,[index],[8])
    print(road_all_turn[index][8])
    # 建立数组
    # SendRoad(road_all_turn[index][8], ser)
    road_temp = road_all_turn[index][8]
    tracking.main()
    return 
    while road_temp:
        # 循迹
        drive.Tracking()
    # 走过记为一
    bao_flag = TellPic.main()
    point_flag[index] = 1
    # 如果是真宝藏
    # 如果是对面宝藏
    # 红三为0，红圆为1，蓝三为2，蓝圆为3
    if bao_flag == group_color:
        # 对面的不用去了
        point_flag = DealBao(point_flag, c_zone, index)
    if bao_flag // 2 == group_color - 1:
        i_index = MarkFour(index, c_zone)
        point_flag[i_index] = 1
    # 进入循环前的赋值
    # 起点
    start_index = index
    while sum(point_flag) < len(point_flag):
        # 终点
        end_index = FindMinRoad(start_index, point_flag, step)
        SendRoad(road_all_turn[start_index][end_index], ser)
        bao_flag = TellPic.main()
        if bao_flag == group_color:
            # 对面的不用去了
            point_flag = DealBao(point_flag, c_zone, end_index)
        if bao_flag // 2 == group_color - 1:
            i_index = MarkFour(end_index, c_zone)
            point_flag[i_index] = 1
        # 终点记为走过
        point_flag[end_index] = 1
        start_index = end_index


# 初始化摄像头180度舵机类        
def init_camera_rotion(rotion):
    mh = Emakefun_MotorHAT(addr=0x60)
    myServo3 = mh.getServo(3)
    speed = 8
    myServo3.writeServoWithSpeed(rotion, speed)
    time.sleep(0.1)
    return 0


# def main19():
#     [road_all_turn, c_zone, turns] = MkRoad.Mkroad()
#     #write(road_all_turn)
#     write(turns)
#     tracking.main()

if __name__ == "__main__":
    main()