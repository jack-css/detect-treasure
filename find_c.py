import time
from matplotlib import pyplot as plt
import cv2
import numpy as np
from numpy import sort, random


# 去曲率

def MakeNor(image):
    # img = cv2.imread('../test0.png')
    P = [[2.58100780e+03, 0, 320],
         [0, 2.58732767e+03, 240],
         [0, 0, 1]]
    K = [-1.51904924e+01, 3.45247566e+02, 3.40699221e-02, -1.07431419e-01,
         -6.08582432e+03]
    img_distort = cv2.undistort(image, np.array(P), np.array(K))
    # img_diff = cv2.absdiff(img, img_distort)
    # cv2.imshow('img', img)
    # cv2.imshow('img_distort', img_distort)
    # cv2.imshow('img_absdiff', img_diff)
    # cv2.waitKey(0)
    return img_distort


# 开启摄像头
def Get_img_FromCamera():
    # 1是第一号摄像头
    WIDTH = 480
    HEIGHT = 640
    # , cv2.CAP_DSHOW
    cap = cv2.VideoCapture(0)  # 打开摄像头
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # 关闭自动曝光
    #cv2.CAP_PROP_EXPOSURE = 25
    print(cap.isOpened())
    while 1:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 60);      #亮度 1
        
        #cap.set(cv2.CAP_PROP_CONTRAST,40);        #对比度 40
        #cap.set(cv2.CAP_PROP_SATURATION, 50);     #饱和度 50
        #cap.set(cv2.CAP_PROP_HUE, 25);            #色调 50
        cap.set(cv2.CAP_PROP_EXPOSURE, 180);       #曝光 50
        #print(cap.get(cv2.CAP_PROP_EXPOSURE))
        
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        # 设置摄像头设备帧率,如不指定,默认600
        # cap.set(cv2.CAP_PROP_FPS, 600)
        # 建议使用XVID编码,图像质量和文件大小比较都兼顾的方案
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # get a frame
        # for _ in range(200):
        #     cap.grab()  # 只读取，但不使用帧数据
        ret, img = cap.read()
        # cv2.waitKey(20)
        # img = cv2.flip(img, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示
        img = MakeNor(img)
        # 调整亮度和对比度
       
        # cv2.waitKey(2)
        # cv2.imshow('wse', img) # alpha = 1.0  # 亮度增益
        # beta = 30    # 亮度偏移量
        # img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        result =  tell_color(img)
        print('result is ', result)
        if result is not None:
            return result


def get_roi(cnts, img2):
    """"
    img1:感兴趣部分
    img2；原图像
    """
    # print(img2.shape[:])
    blank = np.zeros(img2.shape[:2], dtype='uint8')
    mask = cv2.drawContours(blank, cnts, -1, 255, -1)
    print(cv2.countNonZero(mask))

    # cv2.imshow('function', img2)
    if cv2.countNonZero(mask) > 1e4:
        return cv2.bitwise_and(img2, img2, mask=mask)
    return None


def tell_shape(img, color):
    # cv2.imwrite('./1.png', img)
    print('color is', color)
    """
    img 是感兴趣图像那部分
    """
    origin_img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 5)
    cv2.imshow('erg', cv2.Canny(img, 50, 255))
    cv2.waitKey(2)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 4, param1=50, param2=50, minRadius=10, maxRadius=120)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        if len(circles[0]) > 1:
            (x, y, r) = circles[0]
            cv2.putText(img, f'{color}  circle', (250 + random.randint(0, 1) * 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        255, 2)
            cv2.imshow('tell_shape', cv2.circle(img, (x, y), r, 255, 10))
            cv2.waitKey(2)
            # cv2.destroyAllWindows()
            # print('找到圆')
            # for circle in circles[0]:
            #     cv2.circle(img, (int(circle[0]), int(circle[1])), int(circle[2]), (125, 0, 255), 5)
            # cv2.imshow('origin_finc', img)
            # cv2.waitKey(10)
            return 0
    hsv_image = cv2.cvtColor(origin_img, cv2.COLOR_BGR2HSV)
    # 定义HSV范围
    lower_range = np.array([50, 50, 85])
    upper_range = np.array([140, 135, 225])
    # lower_range = np.array([50, 50, 85])
    # upper_range = np.array([140, 135, 225])
    # lower_range = np.array([80, 50, 80])
    # upper_range = np.array([150, 180, 150])
    # 创建掩码
    mask = cv2.inRange(hsv_image, lower_range, upper_range)

    # 通过掩码提取图像
    result_image = cv2.bitwise_and(origin_img, origin_img, mask=mask)

    # 显示原始图像和结果图像
    cv2.imshow('Result Image', result_image)
    cv2.waitKey(2)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_HSV2BGR)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
    contours = cv2.findContours(result_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # 遍历轮廓
    # print(len(contours))
    for contour in contours:
        # 计算轮廓的周长
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 50:
            continue
        # 近似轮廓
        # approx = cv2.approxPolyDP(contour, epsilon, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        # 如果近似的轮廓有三个顶点，认为是三角形
        # print(len(approx))
        if len(approx) == 3:
            # print(perimeter)
            img = cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)
            cv2.putText(img, f'{color}  tri-angle', (250 + random.randint(0, 1) * 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        255, 2)
            cv2.imshow('tell_shape', img)
            # cv2.waitKey(1200)
            return 1
    print('未检测到')
    img = cv2.drawContours(img, contours, -1, 255, -1)
    # cv2.imshow('tri', img)
    # cv2.waitKey(1200)
    # cv2.destroyAllWindows()


def bool_rectangle(cnts1):
    for cnt in cnts1:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # print(len(approx))
        if len(approx) == 4:
            # print('this is a rectangle')
            # cv2.drawContours(new_red, cnt, -1, (255, 0, 0), 15)
            # print(epsilon)
            if epsilon > 4.5:
                return 1
    return 0


def tell_color(img):
    # img = cv2.GaussianBlur(img, (5, 5), 1)
    # 存一个传进来的图
    img = cv2.bilateralFilter(img, 15, 40, 15)
    img_origin = img.copy()
    cv2.imshow('origin_blur', img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ball_color = 'red'
    color_dist = {  # 'red': {'Lower': np.array([0, 160, 120]), 'Upper': np.array([10, 255, 255])},
        'red': {'Lower': np.array([0, 70, 50]), 'Upper': np.array([30, 255, 245])},
        'red1': {'Lower': np.array([160, 50, 50]), 'Upper': np.array([190, 255, 245])},
        'blue': {'Lower': np.array([100, 60, 46]), 'Upper': np.array([150, 255, 255])},
        'green': {'Lower': np.array([20, 43, 35]), 'Upper': np.array([30, 255, 255])},
    }
    inRange_hsv1 = cv2.inRange(img.copy(), color_dist[ball_color]['Lower'], color_dist[ball_color]['Upper'])
    inRange_hsv11 = cv2.inRange(img.copy(), color_dist['red1']['Lower'], color_dist['red1']['Upper'])
    inRange_hsv1 = cv2.bitwise_or(inRange_hsv1, inRange_hsv11)
    # 蓝色
    inRange_hsv2 = cv2.inRange(img.copy(), color_dist['blue']['Lower'], color_dist['blue']['Upper'])
    # cv2.imshow('red_img_origin_red', inRange_hsv1)
    blank = np.zeros(img.shape[:2], dtype='uint8')
    cnts1 = cv2.findContours(inRange_hsv1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # 蓝色
    cnts2 = cv2.findContours(inRange_hsv2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # img_red = cv2.drawContours(blank.copy(), cnts1, -1, 255, -1)
    cv2.waitKey(2)
    # 0是圆， 1是三角
    if bool_rectangle(cnts1):
        new_red = get_roi(cnts1, img_origin.copy())
        if new_red is not None:
            new_red = cv2.resize(new_red, dsize=None, fx=1.0, fy=0.8, interpolation=cv2.INTER_LINEAR)
            # cv2.imshow('img_red', new_red)
            result = tell_shape(new_red, 'red')
            if result is not None:
                # return [0, result]
                return 0 + result * 2
        return None
    if bool_rectangle(cnts2):
        new_blue = get_roi(cnts2, img_origin.copy())
        if new_blue is not None:
            new_blue = cv2.resize(new_blue, dsize=None, fx=1.0, fy=0.8, interpolation=cv2.INTER_LINEAR)

            # cv2.imshow('img_red', new_red)
            result = tell_shape(new_blue, 'blue')
            if result is not None:
                # return [1, result]
                return 1 + result * 2
        return None
    return None


# 判断识别是否有效,并返回有效的值
def get_deal_figure(array):
    if not array:
        print('sg')
        return None
    count = np.bincount(array)
    # print(count)
    index = np.argmax(count)
    # print(index)
    print('count is ', count[index])
    if (count[index] / len(array)) > 0.8:
        # print('yes, it is success')
        print('the max appear figure is ', index)
        # red cicle is tuibao(forward)
        if index == 0:
            return 1
        else:
            return 0      
    else:
        return None

def main():
    group_color = 1
    results = []
    # while len(flag) < 600:   
    while get_deal_figure(results) is None:
        results = []
        if len(results) < 6:
            results.append(Get_img_FromCamera())
            # print('res', len(results))
            # if len(results) > 1:
            #     if sum(results) - len(results) < 2:
            #         return 1
            #     else:
            #         return 0
            # flag.append(Get_img_FromCamera())
        cv2.destroyAllWindows()
        return get_deal_figure(results)


if __name__ == '__main__':
    # 0 is red   1 is blue
    # portion is 1
    # 0 is circle 1 is tri_angle
    # portion is 2
    from camera_init import pwm_light
    pwm_light.main()
    main()
