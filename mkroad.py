import time

import cv2
import numpy as np
from math import ceil
import math
from numpy import sort


# 判断是否入队
class Road:
    def __init__(self, zone):
        # 迷宫地图
        self.road_flag = None
        self.zone = zone
        # 记录路径的栈
        self.road_stack = []
        self.road_account = len(self.zone)
        self.step = None
        # treasure
        self.c_zone = None

    def BoolIn(self, location):
        """
        1   是不能入队
        :param location:
        :return:
        """
        x, y = location
        # print(x, y)
        # 越界返回
        if x < 0 or y < 0:
            return 1
        if x >= self.road_account or y >= self.road_account:
            return 1
        if self.zone[x][y] == 0:
            return 1
        return 0

    # 分发寻找两点之间的路径的任务
    def AppointFdRd(self, appoints):
        roads = [[0 for i in range(len(appoints))] for i in range(len(appoints))]
        self.step = [[0 for i in range(len(appoints))] for i in range(len(appoints))]
        for i in range(0, len(appoints)):
            for j in range(i + 1, len(appoints)):
                print(appoints[i], appoints[j])
                start_appoint = appoints[i]
                end_appoint = appoints[j]
                t_appoints = [start_appoint]
                step = 0
                self.road_flag = [[0 for i in range(len(self.zone))] for i in range(len(self.zone))]
                self.road_flag[start_appoint[0]][start_appoint[1]] = -1
                while 1:
                    step = step + 1
                    am = self.FdRdBetween2P(t_appoints, end_appoint, step)
                    t_appoints = (am[0])
                    # print(self.road_stack)
                    # print('返回的', appoints)
                    if am[1] == 1:
                        # print('找到')
                        # print(self.road_flag)
                        # 将栈置空
                        road = self.OutRoad()
                        # print(road)
                        self.step[i][j] = len(road)
                        self.step[j][i] = len(road)
                        roads[i][j] = road
                        self.road_stack = []
                        break
        return roads

    # 寻找两点之间的距离
    def FdRdBetween2P(self, appoints, end_appoint, step):
        x = [-1, 1, 0, 0]
        y = [0, 0, -1, 1]
        # 记录走过的路
        new_appoints = []
        for appoint in appoints:
            # print(new_appoints)
            for i in range(4):
                t_appoint = [appoint[0] + x[i], appoint[1] + y[i]]
                self.road_stack.append(t_appoint)
                if t_appoint == end_appoint:
                    print('找到')
                    self.road_stack.append(t_appoint)
                    # print(self.road_stack)
                    self.road_flag[t_appoint[0]][t_appoint[1]] = step
                    return [new_appoints, 1]
                if self.BoolIn(t_appoint) == 0:
                    # 没走过则计入
                    if self.road_flag[t_appoint[0]][t_appoint[1]] * 1 == 0:
                        self.road_stack.append(t_appoint)
                        new_appoints.append(t_appoint)
                        # 记录步数
                        self.road_flag[t_appoint[0]][t_appoint[1]] = step
                else:
                    self.road_stack.pop()
        return [new_appoints, 0]

    # 输出路径
    def OutRoad(self):
        stack_road = []
        temp = self.road_stack[-1]
        stack_road.append(temp)
        # print(self.road_stack)
        for i in range(len(self.road_stack) - 1, -1, -1):
            if abs(self.road_stack[i][0] - temp[0]) + abs(self.road_stack[i][1] - temp[1]) == 1:
                temp = self.road_stack[i]
                stack_road.append(temp)
        return stack_road

    def GetTurn(self, datas):
        road_all = [[0 for i in range(len(datas))] for i in range(len(datas))]

        def BoolCross(position):
            x = [-1, 1, 0, 0]
            y = [0, 0, -1, 1]
            sum = 0
            for i in range(4):
                new_position = [position[0] + x[i], position[1] + y[i]]
                sum = int(self.BoolIn(new_position)) + sum
            return sum < 2

        # 具体处理细则
        def Deal(data):
            if data == 0:
                return 0
            # 返回值
            data_temp = []
            x = data[1][0] - data[0][0]
            y = data[1][1] - data[0][1]
            # for i in range(1, len(data)):
            for i in range(2, len(data) - 1):
                new_x = data[i][0] - data[i - 1][0]
                new_y = data[i][1] - data[i - 1][1]
                # print(new_y, new_x, x, y)
                if new_x != x or new_y != y or BoolCross(data[i - 1]):
                    x = new_x
                    y = new_y
                    # print(data[i-2], data[i-1], data[i])
                    data_temp.append(GetMes(data[i - 2], data[i - 1], data[i]))
                    # print(GetMes(data[i-2], data[i-1], data[i]))
                    # print(data_temp)
            # data_temp.append(data[-1])
            return data_temp

        for i in range(len(datas)):
            data_m = datas[i]
            if data_m == 0:
                continue
            for j in range(len(data_m)):
                temp = Deal(data_m[j])
                # if temp != 0:
                #     temp.append(self.c_zone[i])
                # print(self.c_zone[i])
                road_all[i][j] = temp
        return road_all


class Pic():
    def __init__(self, img):
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 划分有w_num列
        self.w_num = 21
        self.h_num = 21
        # w是在x轴分
        # 宽度（长度）
        # 透视变换矩阵
        self.box = None
        # 最大矩形顶点
        self.approx = None
        # treasure
        self.c_zone = None
        # 出入口
        """
        0 是左下角和右上角
        1 是右下角和左上角
        """
        self.entrance = 1

    # 获取起始点
    def GetRoadStart(self):
        entry = []
        if self.entrance == 0:
            # 入口
            entry.append([1, 1])
            # 出口
            entry.append([19, 19])
        else:
            # 入口
            entry.append([19, 1])
            # 出口
            entry.append([1, 19])
        return entry

    # 二值化迷宫
    def getBoard1(self, imm, circles):
        # 误差加
        w_add_e = 10.8
        h_add_e = 11.2
        # print(imm.shape[0], imm.shape[1])
        ww = imm.shape[0] / self.w_num
        # 高度
        hh = imm.shape[1] / self.h_num
        # 记录宝藏的地点
        c_zone = []
        ww_c = (imm.shape[0] - 0) / 21
        hh_c = (imm.shape[1] - 0) / 21
        for circle in circles:
            y = round((circle[0] - 10) / hh_c)
            print('y', circle[0] / ww_c)
            # x = round(circle[1] / hh)
            x = round((circle[1] - 10) / ww_c)
            print('x', circle[1] / hh_c)
            c_zone.append([int(x), int(y)])
        # img = cv2.circle(imm, (320, 355), 3, (155, 25, 125))
        # print(imm.shape)
        board = [[0 for w in range(self.w_num)] for h in range(self.h_num)]  # 创建数组
        # print(ww * w_num, hh * h_num)
        for h in range(self.h_num):
            for w in range(self.w_num):
                if imm[round(w * ww + w_add_e)][round(hh * h + h_add_e)] > 0:  # 坐标转换
                    # print(imm[math.ceil(w * ww)][math.floor(hh * h) + 15], w, h)
                    # 先行后列
                    board[h][w] = 255
                img = cv2.circle(imm, (round(h * hh + h_add_e), round(w * ww + w_add_e)), 3, (155, 25, 125))
        cv2.imshow('img', img)
        # cv2.waitKey(0)
        # if (imm[10 * h + 12, 10 * w + 57] - np.uint8(255)).any():
        # 这个board是19*21
        # print(board)
        # cv2.imshow('origin_pic', img)
        # cv2.waitKey(0)
        # board = MakeMatrix(board)
        r_value = [board, c_zone]
        return r_value

    # 获取定位点的坐标
    def ImgMoreS(self, gaussian_index):
        img = self.img
        origin_img = img.copy()
        # imgBlur = cv2.GaussianBlur(img, (gaussian_index, gaussian_index), 1)  # 高斯模糊
        imgBlur = cv2.bilateralFilter(img, 5, 4, 5)
        # img = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 1)
        # img = cv2.erode(img, (3,3), 5)
        img = cv2.dilate(img, (3, 3), iterations=2)
        # cv2.imshow('contous', imgBlur)
        # cv2.waitKey(0)
        img = cv2.Canny(imgBlur, 25, 255)  # Canny算子边缘检测
        img[50:420, 120:520] = 0
        # img = cv2.bilateralFilter(img, 5, 4, 5)
        # img = cv2.dilate(img, (5, 5), iterations=2)
        blank = np.zeros(img.shape[:2])

        # img = cv2.dilate(img, (5, 5), iterations=2)
        blank = np.zeros(img.shape[:2], dtype='uint8')
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 寻找轮廓点
        for cnt in contours:
            img = cv2.drawContours(blank, cnt, -1, 255, 5)
            # cv2.imshow('contous', img)
            cv2.waitKey(120)
        rect = []
        for cnt in contours:
            perimeter = cv2.arcLength(cnt, True)  # 计算轮廓周长
            approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)  # 获取轮廓角点坐标
            CornerNum = len(approx)  # 轮廓角点的数量
            area = cv2.contourArea(cnt)  # 计算轮廓内区域的面积
            if CornerNum == 4:
                if area > 100:
                    # print(area)
                    # print(approx[0][0], approx[-1][0])
                    cv2.drawContours(blank, cnt, -1, (125, 125, 255), 5)
                    # cv2.imshow('bf', blank)
                    # cv2.waitKey(0)
                    x, y, w, h = cv2.boundingRect(approx)
                    if w + h > 100:
                        continue
                    approx = sum(approx)[0]
                    box = [approx[0] / 4, approx[1] / 4]
                    rect.append(box)
        # print(img.shape)
        # print(len(contours))
        print('透视矫正点是')
        print(rect)
        if len(rect) != 4:
            print('矫正点检测错误')
            return None
        return rect

    # 寻找最大轮廓

    def DeMaxR(self, img):
        # imgBlur = cv2.GaussianBlur(img, (5, 5), 1)  # 高斯模糊
        imgBlur = cv2.bilateralFilter(img, 5, 10, 5)
        # cv2.equalizeHist(imgBlur, imgBlur)
        # _, imgBlur = cv2.threshold(imgBlur, 165, 255, cv2.THRESH_BINARY)
        _, imgBlur = cv2.threshold(imgBlur, 105, 255, cv2.THRESH_BINARY)
        # 增强
        # imgBlur = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
        imgBlur = cv2.bitwise_not(imgBlur)
        # # imgBlur = img
        # # imgCanny = cv2.Canny(imgBlur, 120, 255)  # Canny算子边缘检测
        imgCanny = imgBlur
        cv2.imshow('demax', imgCanny)
        #cv2.waitKey(0)
        # cv2.imshow('Canny', imgCanny)
        # cv2.waitKey(0)
        contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 寻找轮廓点
        perimeters = [cv2.arcLength(cnt, True) for cnt in contours]
        print(sort(perimeters)[-3:])
        for cnt in contours:
            perimeter = cv2.arcLength(cnt, True)  # 计算轮廓周长
            # print(perimeter)
            if perimeter < 2000:
                continue
            img = cv2.drawContours(img, cnt, -1, (0, 0, 0), 10)
            # print(cnt)
            cv2.imshow('evlunkou', img)
            #cv2.waitKey(0)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)  # 获取轮廓角点坐标
            # print(approx)
            self.approx = approx
            print('最大轮廓矫正点数量')
            # print(approx)
            print(len(self.approx))
            return 0
        # cv2.waitKey(0)
        # print(len(contours))
        # img = cv2.drawContours(img, contours, -1, (255, 0, 255), 3)
        # cv2.imshow('evlunkou', img)
        # cv2.waitKey(0)
        # 按轮廓长度正排序
        return 0


# 腐蚀
def ImgErode(imm):
    # ret, binary = cv2.threshold(imm, 165, 255, cv2.THRESH_BINARY)  # 二值化
    cv2.equalizeHist(imm, imm)
    # imm = cv2.bilateralFilter(imm, 5, 4, 5)
    cv2.imshow('pic1', imm)
    #cv2.waitKey(0)
    ret, binary = cv2.threshold(imm, 25, 255, cv2.THRESH_BINARY)  # 二值化
    # binary = cv2.adaptiveThreshold(imm, 65, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 3)
    # binary = cv2.bitwise_not(binary)
    # binary = cv2.Canny(binary, 120, 255)
    # binary = cv2.bilateralFilter(binary, 5, 15, 8)
    cv2.imshow('pic1', binary)
    # cv2.waitKey(0)
    a = imm.shape
    # 高
    width = math.floor(a[0] / 21)
    # 宽
    lenth = math.floor(a[1] / 21)
    print('erode')
    print(width, lenth)
    k = np.ones((21, 21), np.uint8)  # 定义核
    binary = cv2.erode(binary, k)  # 腐蚀
    # binary = cv2.dilate(binary, (3, 3),  iterations=13)
    # cv2.imshow('pic1', binary)
    # cv2.waitKey(0)
    return binary


# 逆时针，左下开始
def Perspective_transform(box, original_img):
    # 获取画框宽高(x=orignal_W,y=orignal_H)
    orignal_W = math.ceil(np.sqrt((box[3][1] - box[2][1]) ** 2 + (box[3][0] - box[2][0]) ** 2))
    orignal_H = math.ceil(np.sqrt((box[3][1] - box[0][1]) ** 2 + (box[3][0] - box[0][0]) ** 2))
    # 原图中的四个顶点,与变换矩阵
    pts1 = np.float32([box[0], box[1], box[2], box[3]])
    pts2 = np.float32(
        [[int(orignal_W + 1), int(orignal_H + 1)], [0, int(orignal_H + 1)], [0, 0], [int(orignal_W + 1), 0]])

    # 生成透视变换矩阵；进行透视变换
    M = cv2.getPerspectiveTransform(pts1, pts2)
    result_img = cv2.warpPerspective(original_img, M, (int(orignal_W + 3), int(orignal_H + 1)))
    return result_img


# 寻找宝藏
def FindC(img):
    #     cv2.imshow('ww', img)
    #     cv2.waitKey(0)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, param1=20, param2=20, minRadius=5, maxRadius=15)
    print('有圆', len(circles[0]))
    # print(circles)
    i = 0
    for am in circles[0]:
        cv2.circle(img, (math.floor(am[0]), math.floor(am[1])), math.floor(am[2]), (125, 0, 255), 3)
        cv2.putText(img, str(i), (math.floor(am[0]), math.floor(am[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 128, 128), 2, cv2.LINE_AA)
        i += 1
    cv2.imshow('find_circles', img)
    # cv2.waitKey(0)
    return circles[0]


# 验证圆形是否在坐标上
def make_correct(img, circles):
    # cv2.imwrite('./1.jpg', img)
    if len(circles) != 8:
        print('未找全全部宝藏')
        return 1
    # cv2.imshow('correct', img)
    # cv2.waitKey(0)
    # print(circles)
    for am in circles:
        # print(img[math.floor(am[0]), math.floor(am[1])])
        # cv2.circle(img, math.floor(am[0]), math.floor(am[1]), math.floor(am[2]), (125, 255, 255), 6)
        if img[math.floor(am[1]), math.floor(am[0])] < 50:
            cv2.circle(img, (math.floor(am[0]), math.floor(am[1])), 5, (125, 255, 255), 6)
        else:
            # print(img[math.floor(am[1]), math.floor(am[0])])
            # print(img[math.floor(am[0]) + 2, math.floor(am[1]) + 2])
            # print(type(math.floor(am[1])), math.floor(am[2]))
            # print(am)
            cv2.circle(img, (int(math.floor(am[0])), int(math.floor(am[1]))), 5, (125, 255, 255), 6)
            return 1
    # cv2.imwrite('./1.jpg', img)
    # cv2.imshow('erb', img)
    # cv2.waitKey(0)
    return 0


# 将四个点再切割
def BoxDeal(box):
    m = np.mat(box)
    a = [200, 200]
    sign = np.sign(m - a)
    plus = [18, 10]
    # m = m + np.multiply(plus, -sign)
    m = m
    box = np.array(m)
    return box


# 将点排序
def SortBox(box):
    media = [0, 0]
    for i in range(len(box)):
        media = [media[0] + box[i][0], media[1] + +box[i][1]]
    media = [media[0] / 4, media[1] / 4]
    for i in range(len(box)):
        if box[i][0] - media[0] > 0 and box[i][1] - media[1] > 0:
            box_ru = box[i]
            # print('ru', box_ru)
        elif box[i][0] - media[0] > 0 and box[i][1] - media[1] < 0:
            box_rd = box[i]
            # print('rd', box_rd)
        elif box[i][0] - media[0] < 0 and box[i][1] - media[1] > 0:
            box_lu = box[i]
            # print('lu', box_lu)
        elif box[i][0] - media[0] < 0 and box[i][1] - media[1] < 0:
            box_ld = box[i]
            # print('ld', box_ld)
    box = []
    print('修正后的透视矫正点是')
    print(box_ld, box_rd, box_ru, box_lu)
    box.append(box_ld)
    box.append(box_rd)
    box.append(box_ru)
    box.append(box_lu)
    return box


# 从摄像头采集图片
def Get_img_FromCamera():
    # 1是第一号摄像头
    WIDTH = 480
    HEIGHT = 640
    # 120 , 160的整数被
    cap = cv2.VideoCapture(0)  # 打开摄像头
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # 关闭自动曝光
    cap.set(cv2.CAP_PROP_EXPOSURE, 84)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 24);      #亮度 1
    a = 100
    while 1:
        if not cap.isOpened():
            print("无法打开摄像头")
            exit()
        # 建议使用XVID编码,图像质量和文件大小比较都兼顾的方案
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # get a frame
        # print(cv2.CAP_PROP_EXPOSURE)
        ret, frame = cap.read()
        frame = MakeNor(frame)
        # print(frame.shape[:])
        # frame = cv2.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示
        # show a frame
        cv2.imshow("capture", frame)  # 生成摄像头窗口
        cv2.waitKey(1)
        #
        if cv2.waitKey(1) & 0xFF == ord('/'):  # 如果按下q 就截图保存并退出
            cv2.imwrite("test.png", frame)  # 保存路径
            return frame


# 去曲率
def MakeNor(img):
    # img = cv2.imread('../test0.png')
    P = [[474.04040403, 0., 322.48613684],
         [0., 631.34136164, 254.86748751],
         [0., 0., 1.]]
    K = [[-0.70438594, 0.5540529, 0.00289683, 0.00382296, -0.2173833]]
    img_distort = cv2.undistort(img, np.array(P), np.array(K))
    # img_diff = cv2.absdiff(img, img_distort)
    # cv2.imshow('img', img)
    # cv2.imshow('img_distort', img_distort)
    # cv2.imshow('img_absdiff', img_diff)
    cv2.imwrite('test1.png', img_distort)
    # cv2.waitKey(0)
    return img_distort


# 获取透视矫正定位点
def GetBox(c_pic):
    rect = c_pic.ImgMoreS(5)
    if rect:
        pass
    else:
        rect = c_pic.ImgMoreS(3)
    c_pic.box = SortBox(rect)
    c_pic.img = Perspective_transform(c_pic.box, c_pic.img)
    # cv2.imshow('weg', c_pic.img)
    # cv2.waitKey(0)
    c_pic.img = cv2.resize(c_pic.img, (0, 0), None, 1.5, 1.5)


def CheckBao(c_zone, img):
    ww = img.shape[0] / 21
    hh = img.shape[1] / 21
    for temp_s in c_zone:
        print(temp_s)
        location = (int((temp_s[1] + 0) * hh + 12), int((temp_s[0] + 0) * ww) + 10)
        cv2.circle(img, location, 5, (125, 125, 255), 5)
    # cv2.imshow('check,bao', img)
    # cv2.waitKey(0)


def DrawRoad(road, img, c_zone):
    ww = img.shape[0] / 21
    hh = img.shape[1] / 21
    temp = np.array([9, 3, 5, 8, 7, 6, 2, 4, 1, 10])-1
    for i in range(10):
        cv2.putText(img, str(i+1), (int((c_zone[temp[i]][1] - 1) * hh + 12), int((c_zone[temp[i]][0] + 1) * ww + 10)), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 120, 255), 2, cv2.LINE_AA)
    print(c_zone[1], 'errse')
    for i in range(9):
        if temp[i] < temp[i+1]:
            ii = temp[i]
            jj = temp[i+1]
        else:
            ii = temp[i+1]
            jj = temp[i]
        for m in range(1, len(road[ii][jj])):
            temp_s = road[ii][jj][m - 1]
            temp_e = road[ii][jj][m]
            location_s = (int((temp_s[1] + 0) * hh + 12), int((temp_s[0] + 0) * ww + 10))
            location_e = (int((temp_e[1] + 0) * hh + 12), int((temp_e[0] + 0) * ww + 10))
            # cv2.putText(img, str(i), location_s, cv2.FONT_HERSHEY_COMPLEX, 2, (255, 120, 255), 2, cv2.LINE_AA)
            cv2.line(img, location_s, location_e, (125, 125, 255), 5, 4)
    # for i in range(len(road) - 1):
    #     j = i
    #     # if road[i][j] == 0:
    #     #     continue
    #     while j < len(road):
    #         # print(road[i][j])
    #         if road[i][j] == 0:
    #             j = j + 1
    #             continue
    #         for m in range(1, len(road[i][j])):
    #             temp_s = road[i][j][m - 1]
    #             temp_e = road[i][j][m]
    #             location_s = (int((temp_s[1] + 0) * hh + 12), int((temp_s[0] + 0) * ww + 10))
    #             location_e = (int((temp_e[1] + 0) * hh + 12), int((temp_e[0] + 0) * ww + 10))
    #             cv2.line(img, location_s, location_e, (125, 125, 255), 5, 4)
    #             # cv2.imshow('draw_line', img)
    #             # cv2.waitKey(5)
    #         # cv2.waitKey(0)
    #         j = j + 1
    # cv2.waitKey(0)


# 精细化裁剪
def CropImg(approx, img):
    # print(approx)
    sub_num = len(approx)
    print(sub_num)
    temp_x = []
    temp_y = []
    for i in approx:
        temp_x.append(i[0][0])
        temp_y.append(i[0][1])
    print('pic_y', img.shape[1])
    temp_x = sort(temp_x)
    temp_y = sort(temp_y)
    x_min = temp_x[0] + 30
    x_max = temp_x[-1] - 30
    y_min = temp_y[0] - 14
    y_max = temp_y[-1] + 14
    print('精细矫正点是')
    print(x_min, x_max, y_min, y_max)

    cropped_image = img[y_min: y_max, x_min:x_max]
    return cropped_image


# 将测迷宫
def CheckZone(c_zone, zone):
    zone_real = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 0, 255, 255, 255, 0],
                 [0, 255, 0, 255, 0, 0, 0, 255, 0, 255, 0, 255, 0, 0, 0, 255, 0, 255, 0, 0, 0],
                 [0, 255, 0, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 255, 0, 255, 0, 255, 255, 255, 0],
                 [0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 255, 0, 0, 0, 255, 0],
                 [0, 255, 0, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 0, 255, 0, 255, 255, 255, 0],
                 [0, 0, 0, 255, 0, 255, 0, 255, 0, 0, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0],
                 [0, 255, 255, 255, 0, 255, 0, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 0, 255, 0],
                 [0, 255, 0, 0, 0, 255, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 255, 0],
                 [0, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0],
                 [0, 255, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 255, 0],
                 [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 0, 255, 0],
                 [0, 255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 255, 0, 0, 0, 255, 0],
                 [0, 255, 0, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 0, 255, 0, 255, 255, 255, 0],
                 [0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 255, 0, 255, 0, 255, 0, 0, 0],
                 [0, 255, 255, 255, 0, 255, 0, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 0, 255, 0],
                 [0, 255, 0, 0, 0, 255, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0],
                 [0, 255, 255, 255, 0, 255, 0, 255, 255, 255, 255, 255, 0, 255, 255, 255, 0, 255, 0, 255, 0],
                 [0, 0, 0, 255, 0, 255, 0, 0, 0, 255, 0, 255, 0, 255, 0, 0, 0, 255, 0, 255, 0],
                 [0, 255, 255, 255, 0, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    # 将数组行列转换
    zone = list(map(list, zip(*zone)))
    # print(one)
    # print(zone)
    # for i in zone:
    # print(i)
    error = 0
    try:
        for i in range(21):
            j = i
            while j < 21:
                if zone[i][j] == zone[20 - i][20 - j] and zone[i][j] == zone_real[i][j]:
                    j = j + 1
                else:
                    print(zone[i][j])
                    if [i, j] not in c_zone:
                        zone[i][j] = zone_real[i][j]
                        zone[20 - i][20 - j] = zone_real[i][j]
                        error = error + 1
                    else:
                        zone[i][j] = 0
                        zone[20 - i][20 - j] = 0
                    print(i, j)
                    print(zone[i][j])
                    j = j + 1
    # for i in zone:
    #     print(i)
    except IndexError:
        print('error, maybe exceed index\nthe now index is', i, j)
    print(f'矩阵纠错{error}个')
    return zone


def GetMes(former, now, later):
    # flag[0]:1为竖直，0为水平
    # flag[1]:0为增大的方向，1为减小方向
    if now[0] - former[0] > 0:
        # 按列，竖直下走
        flag = '10'
    elif now[0] - former[0] < 0:
        # 按列，竖直上走
        flag = '11'
    else:
        if now[1] - former[1] > 0:
            # 按行走，水平右走
            flag = '01'
        if now[1] - former[1] < 0:
            # 按行走，水平左走
            flag = '00'
        # 按列走
    if flag[1] == '0':
        if later[int(flag[0])] - now[int(flag[0])] > 0:
            # 左转
            return 1
        elif later[int(flag[0])] - now[int(flag[0])] < 0:
            return 2
        else:
            return 3
    else:
        if later[int(flag[0])] - now[int(flag[0])] < 0:
            # 左转
            return 1
        elif later[int(flag[0])] - now[int(flag[0])] > 0:
            return 2
        else:
            return 3

def Turn_Over_Road(arrany):
    for i in range(len(arrany)):
        if arrany[i] == 1:
            arrany[i] = 2
        elif arrany[i] == 2:
            arrany[i] = 1
        else:
            continue
    return arrany


def WriteMes(roads):
    print(roads)
    # indexes = [[4, 8], [5, 4], [6,5],[9,6]]
    indexes = [[2,8],[5,2],[1,5],[9,1]]
    with open('./turn.txt', 'w') as f:
        for i in indexes:
            print(i)
            if i[0] > i[1]:
                f.write(str(Turn_Over_Road(roads[i[1]][i[0]])[::-1]))
            else:
                f.write(str(roads[i[0]][i[1]]))
            f.write('\n')

    # with open('./turn.txt', 'w') as f:
    #     f.write()


def Mkroad():
    flag = 1
    if flag:
        # path = '/home/ubuntu/Desktop/test1.png'
        # path = '/home/ubuntu/Desktop/test0.png'
        # path = 'test1.png'
        # path = "distorted_res.png"
        # path = "./test2.png"
        path = './test.png'
        # path = './pic/test0.png'
        img = cv2.imread(path)
        # img = MakeNor(img)
    else:
        img = Get_img_FromCamera()
        # img = MakeNor(img)
    # 建立图像处理类
    c_pic = Pic(img.copy())
    # 透视矫正定位点
    GetBox(c_pic)
    # 获取精细裁剪点
    # # 图像预处理
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_findc = Perspective_transform(c_pic.box, img)
    img = cv2.resize(img_findc, (0, 0), None, 1.5, 1.5)
    # # 得到精细裁剪点
    c_pic.DeMaxR(img.copy())
    # img = Get_img_FromCamera()
    # img = MakeNor(img)
    # img = cv2.imread(path)

    # cv2.imshow('erb', imgCanny)
    # cv2.waitKey(0)
    # box = BoxDeal(box)
    # box = [[48.0, 414.75], [435.25, 421.25], [435.75, 140.25], [105.5, 93.75]]

    # # 透视变换
    # cv2.imshow('tranform', img)
    # cv2.waitKey(0)
    # board = getBoard1(img,)
    # print(board)
    # 寻找宝藏
    # c_pic.DeMaxR(img)
    # # 精细化裁剪
    img = CropImg(c_pic.approx, c_pic.img)
    # cv2.imshow('crop_img', img)
    # cv2.waitKey(0)
    circles = FindC(img.copy())
    # flag = make_correct(img, circles)
    # if flag:
    #     print('宝藏检测有误')
    # else:
    #     print('宝藏检测无误')
    img = ImgErode(img)
    # for circle in circles:
    #     cv2.circle(img, (int(circle[0]), int(circle[1])), 5, (255, 255, 255), 5)
    # cv2.imshow('eagb', img)
    # cv2.waitKey(0)
    # cv2.imshow('zone_erode', img)
    # cv2.waitKey(0)
    [zone, c_zone] = c_pic.getBoard1(img, circles)
    # with open('./zone.txt', 'w') as f:
    #     f.write(str(zone))
    zone = CheckZone(c_zone, zone)
    print('宝藏坐标化后的坐标')
    print(c_zone)
    road = Road(zone)
    road.c_zone = c_zone
    appoints = c_zone
    CheckBao(c_zone, img)
    entries = c_pic.GetRoadStart()
    for entry in entries:
        appoints.append(entry)
    roads_all = road.AppointFdRd(appoints)
    # print(roads_all)
    # 转弯点
    road_all_turn = road.GetTurn(roads_all)
    # 转弯指示
    # print(road_all_turn)
    WriteMes(road_all_turn)
    # with open('./road.txt', 'w') as f:
    #     for i in road_all_turn:
    #         f.write(str(i))
    #         f.write('\n')
    # print(road_all_turn)
    temp = roads_all
    print(temp)
    print(roads_all[2][4])
    DrawRoad(temp, img,c_zone)
    # cv2.destroyAllWindows()
    cv2.imshow('zone_sample', img)
    cv2.waitKey(0)

    # cv2.waitKey(0)
    with open('./step.txt', 'w') as f:
        f.write(str([i for i in range(10)]))
        f.write('\n')
        for i in range(10):
            f.write(str(road.step[i]))
            f.write('\n')

    return [road_all_turn, c_zone, road.step]

    # print(road.step)
    # cv2.imshow('wev'.img)
    # cv2.waitKey(0)
    # print(road_all_turn[2][4])
    # 第二个宝藏点到第四个宝藏点
    # print(c_zone[2], c_zone[4])


if __name__ == "__main__":
    Mkroad()
