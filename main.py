import cv2
import numpy as np

frameWidth = 1200 # 宽度
frameHeight = 1200 # 高度
cap = cv2.VideoCapture(0) # 调用摄像头
cap.set(3, frameWidth)
cap.set(4, frameHeight)


def empty(a):
    pass

# 阈值调参窗口
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 640)
cv2.createTrackbar("Threshold1", "Parameters", 261, 500, empty)
cv2.createTrackbar("Threshold2", "Parameters", 20, 500, empty)
#cv2.createTrackbar("Area", "Parameters", 5000, 30000, empty)

'''
镜像输出视频（性能极低，不建议使用）
def video_mirror_output(video):
    new_img = np.zeros_like(video)
    h, w = video.shape[0],video.shape[1]
    for row in range(h):
        for i in range(w):
            new_img[row,i] = video[row,w-i-1]
    return new_img
'''

# 图像堆叠
def stackImages(scale, imgArray: [list]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

# 获取轮廓
def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # 获取轮廓

    for cnt in  contours:
        area = cv2.contourArea(cnt) # 计算轮廓面积
        # areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > 1000:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            # print(len(approx))
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5) # 绘制矩形

            cv2.putText(imgContour, "Points:" + str(len(approx)), (x + w+20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2) # 放置文字
            cv2.putText(imgContour, "Area:" + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2) # 放置文字


while True:
    success, img = cap.read()

    imgContours = img.copy()
    imgBlur = cv2.GaussianBlur(img, (7, 7), 1) # 高斯模糊
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY) # 获取灰度图像

    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters") # 获取阈值1滑动条位置
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters") # 获取阈值2滑动条位置
    # print(threshold1, threshold2)

    imgCanny = cv2.Canny(img, threshold1, threshold2) # Canny算法对输入图像进行边缘检测

    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1) # 膨胀算法使图像扩大一圈，给图像中的对象边界添加像素，其运算过程就是用3X3的结构元素，扫描二值图像的每一个像素，用结构元素与其覆盖的二值图像做“与”运算，如果都为0，结构图像的该像素为0，否则为1。结果就是使二值图像扩大一圈。

    getContours(imgDil, imgContours)

    imgStack = stackImages(0.8, ([img, imgGray, imgCanny], [imgDil, imgContours, imgContours]))

    ret = True
    # video_mirror_output(img) 性能太低
    '''
    镜像画面(性能较好)
    if ret:
        imgStack = cv2.flip(imgStack, 180)  # 视频旋转cv2.flip(frame, 1)第一个参数表示要旋转的视频，第二个参数表示旋转的方向，0表示绕x轴旋转，大于0的数表示绕y轴旋转，小于0的负数表示绕x和y轴旋转
    '''
    cv2.imshow("Contour Detect", imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
