#coding=utf-8
import cv2
import serial
import time
#列出可用串口
import serial.tools.list_ports

def sendData(motor,pos):
    # return control byte
    # the 'motor' has only two choice : '0' or '1'
    # the range of 'pos' is 500 - 2500, 500 means 0 degree, and 2500 means 90(?i m not sure) degree. 
    #对输入进行限制
    if motor == 0:
        MotorSend = '00'
    elif motor == 1:
        MotorSend = '01'
    elif motor == 2:
        MotorSend = '02'
    elif motor == 3:
        MotorSend = '03'
    else:
        return
    DataL1 = str(hex(pos & 0xff))
    DataH1 = str(hex(pos>>8))
    try:
        DataL = DataL1[2] + DataL1[3]
    except Exception:
        DataL = '0' + DataL1[2]
    try:
        DataH = DataH1[2] + DataH1[3]
    except Exception:
        DataH = '0' + DataH1[2]
    sendStr = 'ff 02 '+MotorSend + ' ' + DataL + ' ' + DataH
    strByte = bytes.fromhex(sendStr)
    return strByte


plist = list(serial.tools.list_ports.comports())
if len(plist) <= 0:
    print("没有发现端口!")
else:
    plist_0 = list(plist[0])
    serialName = plist_0[0]
    serialFd = serial.Serial(serialName, 9600, stopbits=1, parity=serial.PARITY_SPACE,timeout=60)
if serialFd.isOpen():
    pass
else:
    serialFd.open()
    print("可用端口名>>>", serialFd.name)


#通过发送bytes对舵机控制。
#写入舵机速度
serialFd.write(bytes.fromhex('ff 01 00 04 00')) 
serialFd.write(bytes.fromhex('ff 01 01 04 00'))
serialFd.write(bytes.fromhex('ff 01 02 04 00'))
serialFd.write(bytes.fromhex('ff 01 03 04 00'))
#使舵机位置初始化
serialFd.write(sendData(0,2500))
serialFd.write(sendData(1,500))
serialFd.write(sendData(2,500))
serialFd.write(sendData(3,2500))
#test
serialFd.write(sendData(3,2400))
serialFd.write(sendData(3,2500))
#摄像头捕捉部分
cap = cv2.VideoCapture(0)
for i in range(50):
    ret, frame = cap.read()

# 写入相关信息初始化，通过这部分可以产生保存路径和标记好的文件名
#设置步长
step = 100
pauseTime = 5
#-------------------------left-------------------------------------
#使舵机位置初始化
# serialFd.write(sendData(0,2500))
# serialFd.write(sendData(1,500))
# serialFd.write(sendData(2,500))
# serialFd.write(sendData(3,2500))

# for MotorA in range(2500,1850,-step):
#     MotorB = 500
#     serialFd.write(sendData(0,MotorA))
#     serialFd.write(sendData(1,500))
#     serialFd.write(sendData(2,500))
#     serialFd.write(sendData(3,2500))
#     time.sleep(pauseTime)
#     for MotorC in range(500,1150,step):
#         MotorD = 2500
#         filePath = "%d-%d-%d-%d.jpg" % (MotorA,MotorB,MotorC,MotorD)
#         serialFd.write(sendData(2,MotorC))
#         time.sleep(pauseTime)
#         for i in range(48):
#             ret, frame = cap.read()
#         cv2.imwrite(filePath,frame)
#         # cv2.imshow("get_characteristic",frame)
#         print("%d-%d-%d-%d.jpg" % (MotorA,MotorB,MotorC,MotorD))
    
#     serialFd.write(sendData(2,500))
#     serialFd.write(sendData(3,2500))

#     for MotorD in range(2500-step,1850,-step):
#         MotorC = 500
#         filePath = "%d-%d-%d-%d.jpg" % (MotorA,MotorB,MotorC,MotorD)
#         serialFd.write(sendData(3,MotorD))
#         time.sleep(pauseTime)
#         for i in range(48):
#             ret, frame = cap.read()
#         cv2.imwrite(filePath,frame)
#         # cv2.imshow("get_characteristic",frame)
#         print("%d-%d-%d-%d.jpg" % (MotorA,MotorB,MotorC,MotorD))

#-------------------------right-------------------------------------
#使舵机位置初始化
serialFd.write(sendData(0,2500))
serialFd.write(sendData(1,500))
serialFd.write(sendData(2,500))
serialFd.write(sendData(3,2500))

for MotorB in range(500+step,1150,step):
    MotorA = 2500
    serialFd.write(sendData(0,2500))
    serialFd.write(sendData(1,MotorB))
    serialFd.write(sendData(2,500))
    serialFd.write(sendData(3,2500))
    time.sleep(pauseTime)
    for MotorC in range(500,1150,step):
        MotorD = 2500
        filePath = "%d-%d-%d-%d.jpg" % (MotorA,MotorB,MotorC,MotorD)
        serialFd.write(sendData(2,MotorC))
        time.sleep(pauseTime)
        for i in range(48):
            ret, frame = cap.read()
        cv2.imwrite(filePath,frame)
        # cv2.imshow("get_characteristic",frame)
        print("%d-%d-%d-%d.jpg" % (MotorA,MotorB,MotorC,MotorD))
    
    serialFd.write(sendData(2,500))
    serialFd.write(sendData(3,2500))

    for MotorD in range(2500-step,1850,-step):
        MotorC = 500
        filePath = "%d-%d-%d-%d.jpg" % (MotorA,MotorB,MotorC,MotorD)
        serialFd.write(sendData(3,MotorD))
        time.sleep(pauseTime)
        for i in range(48):
            ret, frame = cap.read()
        cv2.imwrite(filePath,frame)
        # cv2.imshow("get_characteristic",frame)
        print("%d-%d-%d-%d.jpg" % (MotorA,MotorB,MotorC,MotorD))

#使舵机位置初始化
serialFd.write(sendData(0,2500))
serialFd.write(sendData(1,500))
serialFd.write(sendData(2,500))
serialFd.write(sendData(3,2500))


cap.release()
cv2.destroyAllWindows()
print('done')