import numpy as np
import cv2


# 从摄像头采集图片
def Get_img_FromCamera():
    # 1是第一号摄像头
    # WIDTH = 780
    # HEIGHT = 1280
    cap = cv2.VideoCapture(0)  # 打开摄像头
    cap.set(cv2.CAP_PROP_EXPOSURE, -6.3)  # 设置曝光值，可以根据实际情况调整
    cap.set(cv2.CAP_PROP_EXPOSURE, 84)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 24);      #亮度 1
    height = 480
    a = 100
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 1)  # 亮度 1
    cap.set(cv2.CAP_PROP_HUE, 15)  # 色调 50
    # 关闭自动曝光和自动增益控制功能
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # 关闭自动曝光
    cap.set(cv2.CAP_PROP_GAIN, 0)  # 关闭自动增益
    while 1:
        if not cap.isOpened():
            print("无法打开摄像头")
            exit()

        # 设置摄像头设备帧率,如不指定,默认600
        # cap.set(cv2.CAP_PROP_FPS, 1080)
        # 建议使用XVID编码,图像质量和文件大小比较都兼顾的方案
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # get a frame
        ret, frame = cap.read()
        print(frame.shape[:])
        # frame = cv2.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示
        # show a frame
        cv2.line(frame, (frame.shape[1] // 2, 1), (frame.shape[1] // 2, frame.shape[0]), (0, 255, 0), 5)
        try:
            cv2.imshow("capture", frame)  # 生成摄像头窗口
            cv2.waitKey(1)
        except cv2.error:
            print('camera can not capture imagine')
        if cv2.waitKey(1) & 0xFF == ord('/'):  # 如果按下q 就截图保存并退出
            cv2.imwrite("test.png", frame)  # 保存路径
            cap.release()
            return frame

        # cap.release()


# 准备物体点和图像点
object_points = []  # 存储物体点
image_points = []  # 存储图像点

# 从文件或实际采集的图像中加载图像
# image = cv2.imread('test2.png')
image = Get_img_FromCamera()
gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

# 定义棋盘格尺寸
board_size = (9, 9)  # 内部角点数目

# 查找棋盘格角点
found, corners = cv2.findChessboardCorners(gray, board_size, None)
# 如果找到棋盘格角点
if found:
    # 添加物体点
    print('sew')
    object_points.append(np.zeros((board_size[0] * board_size[1], 3), np.float32))
    object_points[-1][:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    cv2.drawChessboardCorners(image, board_size, corners, found)
    # 添加图像点
    image_points.append(corners)
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    cv2.drawChessboardCorners(image, (4, 11), corners2, found)
cv2.imshow('wrg', image)
cv2.waitKey(0)
# 进行相机标定
ret, K, D, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

# 输出相机参数
print("相机内部参数矩阵：")
print(K)
print("\n畸变系数：")
print(D)
