import cv2
import matplotlib.pyplot as plt
import numpy as np

#----------------------直接查看图像-----------------------------
# cap = cv2.VideoCapture(0)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
# for i in range(50):
#         ret, frame = cap.read()
# while(1):
#     # get a frame
    
#     ret, frame = cap.read()
#     # show a frame
#     frame = frame[60:420,:630]
#     # cv2.imshow("aaa",frame)

#     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#     # cv2.imshow("aaa",gray)
#     # 2-mode
#     ret1, binary  = cv2.threshold(gray, 23, 255, cv2.THRESH_BINARY)
#     # cv2.imshow("aaa",binary)
    
#     # 开运算
#     opened_pic = cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)
#     # cv2.imshow("aaa",opened_pic)
#     #寻找边界
#     contours, boundary,ret2 = cv2.findContours(opened_pic,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
#     cv2.drawContours(opened_pic,boundary,-1,(0,0,255),3)
#     # cv2.imshow("bbb",opened_pic)
#     #计算坐标
    
#     x_min = np.zeros((len(boundary),1))
#     x_max = np.zeros((len(boundary),1)) 
#     y_max = np.zeros((len(boundary),1))
#     y_min = np.zeros((len(boundary),1))
#     color = (0,255,0)
#     for i in range(len(boundary)):
#         x_min[i] = boundary[i][0][0][0]
#         x_max[i] = boundary[i][0][0][0]
#         y_min[i] = boundary[i][0][0][1]
#         y_max[i] = boundary[i][0][0][1]
#         for j in range(len(boundary[i])):
#             if boundary[i][j][0][0] > x_max[i]:
#                 x_max[i] = boundary[i][j][0][0]
#             if boundary[i][j][0][0] < x_min[i]:
#                 x_min[i] = boundary[i][j][0][0]
#             if boundary[i][j][0][1] > y_max[i]:
#                 y_max[i] = boundary[i][j][0][1]
#             if boundary[i][j][0][1] < y_min[i]:
#                 y_min[i] = boundary[i][j][0][1]
#     for i in range(len(boundary)-1):
#         i = i + 1
#         yuanxin_x = int((x_max[i] - x_min[i])/2+x_min[i])
#         yuanxin_y = int((y_max[i] - y_min[i])/2+y_min[i])
#         yuanxin = (yuanxin_x,yuanxin_y)
#         cv2.circle(frame,yuanxin,10,color)
#     cv2.waitKey(50)
#     cv2.imshow("get_characteristic",frame)

# cap.release()


# -------------------直方图--------------------------------
# image = cv2.imread('2100-1475.jpg')

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# plt.subplot(131), plt.imshow(image, "gray")
# plt.title("source image"), plt.xticks([]), plt.yticks([])
# plt.subplot(132), plt.hist(image.ravel(), 256)
# plt.title("Histogram"), plt.xticks([]), plt.yticks([])
# ret1, th1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# plt.subplot(133), plt.imshow(th1, "gray")
# plt.title("2-Mode Method"), plt.xticks([]), plt.yticks([])
# plt.show()

#----------------------保存到xls中---------------------------------
import xlwt
import os
workbook = xlwt.Workbook(encoding = 'ascii')
worksheet = workbook.add_sheet('My Worksheet')
def read_pic(filepath,number):
    try:
        # get a frame
        frame = cv2.imread(filepath)
        # show a frame
        frame = frame[60:420,:630]
        # cv2.imshow("aaa",frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # cv2.imshow("aaa",gray)
        # 2-mode
        ret1, binary  = cv2.threshold(gray, 23, 255, cv2.THRESH_BINARY)
        # cv2.imshow("aaa",binary)
        
        # 开运算
        opened_pic = cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)
        # cv2.imshow("aaa",opened_pic)
        #寻找边界
        contours, boundary,ret2 = cv2.findContours(opened_pic,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
        cv2.drawContours(opened_pic,boundary,-1,(0,0,255),3)
        if len(boundary) > 7:
            return False
        # cv2.imshow("bbb",opened_pic)
        #计算坐标
        
        x_min = np.zeros((len(boundary),1))
        x_max = np.zeros((len(boundary),1)) 
        y_max = np.zeros((len(boundary),1))
        y_min = np.zeros((len(boundary),1))
        color = (0,255,0)
        for i in range(len(boundary)):
            x_min[i] = boundary[i][0][0][0]
            x_max[i] = boundary[i][0][0][0]
            y_min[i] = boundary[i][0][0][1]
            y_max[i] = boundary[i][0][0][1]
            for j in range(len(boundary[i])):
                if boundary[i][j][0][0] > x_max[i]:
                    x_max[i] = boundary[i][j][0][0]
                if boundary[i][j][0][0] < x_min[i]:
                    x_min[i] = boundary[i][j][0][0]
                if boundary[i][j][0][1] > y_max[i]:
                    y_max[i] = boundary[i][j][0][1]
                if boundary[i][j][0][1] < y_min[i]:
                    y_min[i] = boundary[i][j][0][1]
        for i in range(len(boundary)-1):
            i = i + 1
            yuanxin_x = int((x_max[i] - x_min[i])/2+x_min[i])
            yuanxin_y = int((y_max[i] - y_min[i])/2+y_min[i])
            # yuanxin = (yuanxin_x,yuanxin_y)
            # cv2.circle(frame,yuanxin,10,color)
            worksheet.write(number, 3+i, label = yuanxin_x)
            worksheet.write(number, 9+i, label = yuanxin_y)
        return True
        # cv2.waitKey(50)
        # cv2.imshow("get_characteristic",frame)
    except Exception:
        return False

# workbook = xlwt.Workbook(encoding = 'ascii')
# worksheet = workbook.add_sheet('My Worksheet')
# worksheet.write(0, 0, label = 'Row 0, Column 0 Value')


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
step = 100
Num = 0
for MotorA in range(2500,1850,-step):
    MotorB = 500
    for MotorC in range(500,1150,step):
        MotorD = 2500
        if not read_pic("%d-%d-%d-%d.jpg" % (MotorA,MotorB,MotorC,MotorD),Num):
            print("%d-%d-%d-%d.jpg" % (MotorA,MotorB,MotorC,MotorD))
            continue
        worksheet.write(Num, 0, label = MotorA)
        worksheet.write(Num, 1, label = MotorB)
        worksheet.write(Num, 2, label = MotorC)
        worksheet.write(Num, 3, label = MotorD)
        
        Num += 1

    for MotorD in range(2500,1850-step,-step):
        MotorC = 500
        if not read_pic("%d-%d-%d-%d.jpg" % (MotorA,MotorB,MotorC,MotorD),Num):
            print("%d-%d-%d-%d.jpg" % (MotorA,MotorB,MotorC,MotorD))
            continue
        worksheet.write(Num, 0, label = MotorA)
        worksheet.write(Num, 1, label = MotorB)
        worksheet.write(Num, 2, label = MotorC)
        worksheet.write(Num, 3, label = MotorD)
        
        Num += 1

for MotorB in range(500+step,1150,step):
    MotorA = 2500

    for MotorC in range(500,1150,step):
        MotorD = 2500
        if not read_pic("%d-%d-%d-%d.jpg" % (MotorA,MotorB,MotorC,MotorD),Num):
            print("%d-%d-%d-%d.jpg" % (MotorA,MotorB,MotorC,MotorD))
            continue
        worksheet.write(Num, 0, label = MotorA)
        worksheet.write(Num, 1, label = MotorB)
        worksheet.write(Num, 2, label = MotorC)
        worksheet.write(Num, 3, label = MotorD)
        
        Num += 1

    for MotorD in range(2500-step,1850,-step):
        MotorC = 500
        if not read_pic("%d-%d-%d-%d.jpg" % (MotorA,MotorB,MotorC,MotorD),Num):
            print("%d-%d-%d-%d.jpg" % (MotorA,MotorB,MotorC,MotorD))
            continue
        worksheet.write(Num, 0, label = MotorA)
        worksheet.write(Num, 1, label = MotorB)
        worksheet.write(Num, 2, label = MotorC)
        worksheet.write(Num, 3, label = MotorD)
        
        Num += 1
    

workbook.save('Excel_Workbook220.xls')
print('done')

#-------------------------读取xls，查看结果---------------------------------------