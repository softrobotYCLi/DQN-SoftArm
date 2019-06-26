import numpy as np
from numpy import array
from sklearn.preprocessing import MinMaxScaler
import xlrd
import keras
import matplotlib.pyplot as plt
import os
import math 
import tensorflow as tf
#cv
import cv2
#serial
import serial
import time
import serial.tools.list_ports
import xlwt
class softArm:
    def __init__(self):
        # parameters
        self.name = 'softArm'
        self.enable_actions = (0, 1, 2, 3)
        self.pos_acc = 10
        # self.minMax10 = MinMaxScaler()
        # a = np.array([2500,1450,1875,500])
        # b = a.reshape(2,-1)
        # self.minMax10.fit_transform(b)
        # variables
        # isreal = True 真实电机控制
        self.isreal = True
        if self.isreal == True:
            self.serial_init()
            self.cv_init()
        self.file_pos = open('pos_cur.txt','a')
        self.file_tar = open('pos_target.txt','a') 
            

        self.load_model()
        self.reset()

    def load_model(self):
        workbook = xlrd.open_workbook('Excel_Workbook.xls')
        sheet1 = workbook.sheet_by_index(0)
        
        self.row_num = len(sheet1.col_values(0))
        self.input_data = np.zeros([self.row_num,4])
        self.input_data[:,0] = array(sheet1.col_values(0))
        self.input_data[:,1] = array(sheet1.col_values(1))
        self.input_data[:,2] = array(sheet1.col_values(2))
        self.input_data[:,3] = array(sheet1.col_values(3))

        self.minMax0 = MinMaxScaler()
        self.input_data1 = self.minMax0.fit_transform(self.input_data)
        self.output_data = np.zeros([self.row_num,2])
        self.output_data[:,0] = array(sheet1.col_values(4))
        self.output_data[:,1] = array(sheet1.col_values(10))
        

        self.minMax2 = MinMaxScaler()
        self.output_data1 = self.minMax2.fit_transform(self.output_data)

        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(64,input_shape=(4,),activation='tanh'))
        self.model.add(keras.layers.Dense(64, activation='tanh'))
        self.model.add(keras.layers.Dense(64, activation='tanh'))
        self.model.add(keras.layers.Dense(64, activation='tanh'))
        self.model.add(keras.layers.Dense(64, activation='tanh'))
        self.model.add(keras.layers.Dense(64, activation='tanh'))
        self.model.add(keras.layers.Dense(2, activation='tanh'))
        opt =keras.optimizers.Adam(lr=0.001)
        self.model.compile(loss='mse',optimizer=opt, metrics=['accuracy'])
        # self.model.load_weights('123.h')
        # self.model.fit(self.input_data1,self.output_data1,epochs=10000,verbose=2)
        # self.model.save_weights("123.h")
        self.model.load_weights('304.h')
    def draw(self):
        def set_target_from_mouse(event):
            try:
                if event.ydata>-400 and event.ydata<-200:
                    self.target[0][0] = event.xdata
                    self.target[0][1] = -event.ydata
                    #-----------------------
                    # self.target[0][0] = 549
                    # self.target[0][1] = 228

            except Exception:
                pass
            

        plt.gcf().canvas.mpl_connect('motion_notify_event', set_target_from_mouse)
        
        aa = self.minMax2.inverse_transform(self.current_pos)
        # plt.plot(aa[0][0:6],-aa[0][6:12],'-')
        # plt.hold(True)
        plt.plot(aa[0][0],-aa[0][1],'o')
        plt.hold(True)
        plt.plot(self.target[0][0],-self.target[0][1],'*')

        plt.xlim(0,600)
        plt.ylim(-350,0)
        plt.title(str(self.motorA_pos)+' '+str(self.motorB_pos)+' '+str(self.motorC_pos)+' '+str(self.motorD_pos)+' '+str(self.reward))
        plt.pause(0.001)
        plt.hold(False)
    def update(self, action):
        """
        action:
            0: 上节 move left
            1: 上节 move right
            2: 下节 move left
            3: 下节 move right
            
        """
        # update player position
        self.over_boundary = 0
        
        if action == self.enable_actions[0]:
            #A move left
            if self.motorA_pos <= 1900:
                self.over_boundary = 1
            elif self.motorB_pos == 500:
                if self.motorA_pos - self.step <=1900:#判断可不可以执行
                    self.motorA_pos = 1900
                else:
                    self.motorA_pos -= self.step#25
            else:
                if self.motorB_pos - self.step <=500:#判断可不可以执行
                    self.motorB_pos = 500
                else:
                    self.motorB_pos -= self.step

        elif action == self.enable_actions[1]:
            #A move right
            if self.motorB_pos >= 1100:
                self.over_boundary = 1
            elif self.motorA_pos >= 2500:
                #判断可不可以执行
                if self.motorB_pos + self.step >=1100:
                    self.motorB_pos = 1100
                else:
                    self.motorB_pos += self.step
            else:
                #判断可不可以执行
                if self.motorA_pos + self.step >=2500:
                    self.motorA_pos = 2500
                else:
                    self.motorA_pos += self.step

        elif action == self.enable_actions[2]:
            #B move left
            if self.motorC_pos >= 1100:
                self.over_boundary = 1
            elif  self.motorD_pos >= 2500:
                #判断可不可以执行
                if self.motorC_pos + self.step >=1100:
                    self.motorC_pos = 1100
                else:
                    self.motorC_pos += self.step
            else:
                #判断可不可以执行
                if self.motorD_pos + self.step >=2500:
                    self.motorD_pos = 2500
                else:
                    self.motorD_pos += self.step
        elif action == self.enable_actions[3]:
            #B move right
            if self.motorD_pos <= 1900:
                self.over_boundary = 1
            elif  self.motorC_pos <= 500:
                #判断可不可以执行
                if self.motorD_pos - self.step <=1900:
                    self.motorD_pos = 1900
                else:
                    self.motorD_pos -= self.step
            else:
                #判断可不可以执行
                if self.motorC_pos - self.step <=500:
                    self.motorC_pos = 500
                else:
                    self.motorC_pos -= self.step
        else:
            # do nothing
            print('env:action error')
        if self.motorA_pos!=2500 and self.motorB_pos!=500:
            print('position error')
        if self.motorC_pos!=500 and self.motorD_pos!=2500:
            print('position error')
        # collision detection
        self.reward = 0
        #-----------------real or virtual---------------
        if self.isreal == False:
            q = array([[self.motorA_pos,self.motorB_pos,self.motorC_pos,self.motorD_pos]])
            input_ = self.minMax0.transform(q)
            self.current_pos = self.model.predict(input_)
            A_target= self.target
            B_current_pos= self.minMax2.inverse_transform(self.current_pos) 
        else:
            self.get_from_cv()
            self.output_data[0][0] = self.cv_yuanxin[0]
            self.output_data[0][1] = self.cv_yuanxin[1]
            self.current_pos = self.minMax2.transform([self.output_data[0]])
            A_target= self.target
            B_current_pos= self.minMax2.inverse_transform(self.current_pos) 
        #-----------------save target to file------------
        self.file_pos.write(str(B_current_pos[0][0])+' '+str(-B_current_pos[0][1])+'\n')
        self.file_tar.write(str(A_target[0][0])+' '+str(-A_target[0][1])+'\n')

        #-----------------last pos method---------------
        c_last_pos= self.minMax2.inverse_transform(self.last_pos) 
        #--------------------------------------------------
        # self.distance = 0
        # self.cur_distance = 0
        self.last_distance = 0
        self.terminal_dis = 0
        
            # self.distance += abs(A[0][current_point] - B[0][current_point])# 奖励为距离的绝对值
            #-----------------last pos method---------------------
        self.terminal_dis = abs(A_target[0][0]-B_current_pos[0][0]) + abs(A_target[0][1]-B_current_pos[0][1])
        self.last_distance = abs(A_target[0][0]-c_last_pos[0][0]) + abs(A_target[0][1]-c_last_pos[0][1])

        self.distance = self.last_distance - self.terminal_dis
        self.last_pos = self.current_pos
        #-----------------------------------------------------------
        
        self.run_step += 1
        self.terminal = False
        
        if self.terminal_dis < self.pos_acc:#10
            # catch
            self.terminal = True
            self.reward = 2
        else:
            if self.over_boundary == 1:
                self.reward = -5
            elif self.distance > 0:
                self.reward = 1
            else:
                self.reward = -1

        if self.run_step > 300:
            self.terminal = True
    def observe(self):
        # self.draw()
        envir = []
        temp_target = self.minMax2.transform(self.target)
        target = np.array([temp_target[0][0],temp_target[0][1]])
        current_pos = np.array([self.current_pos[0][0],self.current_pos[0][1]])
        envir.extend(target.flatten())
        envir.extend(current_pos.flatten())
        envir = np.array(envir)
        return envir, self.reward, self.terminal
    def execute_action(self, action):
        #虚拟环境
        self.update(action)
        #真实环境
        if self.isreal == True:
            self.update_rotor_pos()

    def reset(self):
        # 随机起点模式reset player position
        # start_pos = np.random.randint(0,self.row_num)
        # self.motorA_pos = self.input_data[start_pos,0]
        # self.motorB_pos = self.input_data[start_pos,1]
        # self.motorC_pos = self.input_data[start_pos,2]
        # self.motorD_pos = self.input_data[start_pos,3]

        # 固定起点模式reset player position
        self.motorA_pos = 2500
        self.motorB_pos = 500
        self.motorC_pos = 500
        self.motorD_pos = 2500
        

        # reset taget position
        end_pos = np.random.randint(0,self.row_num)
        c = self.input_data[end_pos,0]
        d = self.input_data[end_pos,1]
        e = self.input_data[end_pos,2]
        f = self.input_data[end_pos,3]
        q = array([[c,d,e,f]])
        input_ = self.minMax0.transform(q)
        self.target = self.minMax2.inverse_transform(self.model.predict(input_))
        
        q = array([[self.motorA_pos,self.motorB_pos,self.motorC_pos,self.motorD_pos]])
        input_ = self.minMax0.transform(q)
        self.current_pos = self.model.predict(input_)
        self.last_pos = self.current_pos
        # reset other variables
        self.reward = 0
        self.run_step = 0
        self.terminal = False
        self.distance = 0
        self.step = 10#步长
        if self.isreal == True:
            self.reset_rotor_place()
            

    # function to get_pic from cv
    def get_from_cv(self):
        def cv_set_target(event,x,y,flags,param):
            if event==cv2.EVENT_MOUSEMOVE:
                self.target[0][0] = x
                self.target[0][1] = y
                # self.target[0][0] = 549
                # self.target[0][1] = 228
                print(str(x)+' '+str(y))
            if event == cv2.EVENT_MOUSEHWHEEL:
                self.target[0][0] = x
                self.target[0][1] = y
                self.target[0][0] = 549
                self.target[0][1] = 228
                print(str(x)+' '+str(y))

        for i in range(2):
            self.ret, self.frame = self.cap.read()
        # get a frame
        self.ret, self.frame = self.cap.read()
        # show a frame
        self.frame = self.frame[60:420,:630]
        # cv2.imshow("aaa",frame)

        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
        # cv2.imshow("aaa",gray)
        # 2-mode
        self.ret1, self.binary  = cv2.threshold(self.gray, 23, 255, cv2.THRESH_BINARY)
        # cv2.imshow("aaa",binary)
        
        # 开运算
        self.opened_pic = cv2.morphologyEx(self.binary,cv2.MORPH_OPEN,self.kernel)
        # cv2.imshow("aaa",opened_pic)
        #寻找边界
        self.contours, self.boundary,self.ret2 = cv2.findContours(self.opened_pic,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
        cv2.drawContours(self.opened_pic,self.boundary,-1,(0,0,255),3)
        # cv2.imshow("bbb",opened_pic)
        j = 1
        while j < (len(self.boundary)) and len(self.boundary) > 6:
            if (self.boundary[j].size < 10 or self.boundary[j].size > 60):
                self.boundary.pop(j)
            else:
                j += 1

        #计算坐标
        x_min = np.zeros((len(self.boundary),1))
        x_max = np.zeros((len(self.boundary),1))
        y_max = np.zeros((len(self.boundary),1))
        y_min = np.zeros((len(self.boundary),1))
        color = (0,255,0)
        for i in range(len(self.boundary)):
            x_min[i] = self.boundary[i][0][0][0]
            x_max[i] = self.boundary[i][0][0][0]
            y_min[i] = self.boundary[i][0][0][1]
            y_max[i] = self.boundary[i][0][0][1]
            for j in range(len(self.boundary[i])):
                if self.boundary[i][j][0][0] > x_max[i]:
                    x_max[i] = self.boundary[i][j][0][0]
                if self.boundary[i][j][0][0] < x_min[i]:
                    x_min[i] = self.boundary[i][j][0][0]
                if self.boundary[i][j][0][1] > y_max[i]:
                    y_max[i] = self.boundary[i][j][0][1]
                if self.boundary[i][j][0][1] < y_min[i]:
                    y_min[i] = self.boundary[i][j][0][1]
        if len(self.boundary)!= 7:
            print('other things detect! move it!')
            return        
        for i in range(len(self.boundary)-1,0,-1):
            
            yuanxin_x = int((x_max[i] - x_min[i])/2+x_min[i])
            yuanxin_y = int((y_max[i] - y_min[i])/2+y_min[i])

            self.cv_yuanxin = (yuanxin_x,yuanxin_y)
            cv2.circle(self.frame,self.cv_yuanxin,7,color)
        cv2.circle(self.frame,self.cv_yuanxin,7,(255,0,0))
        cv2.circle(self.frame,(self.target[0][0],self.target[0][1]),12,(0,0,255))
        #jiaozhun----------------------------------------------
        # aa = self.minMax2.inverse_transform(self.current_pos)
        # cv2.circle(self.frame,(aa[0][0],aa[0][1]),12,(255,0,0))
        
        #-----------------------------------------------------
        cv2.waitKey(5)
        cv2.namedWindow('get_characteristic')
        cv2.setMouseCallback('get_characteristic',cv_set_target)
        cv2.imshow("get_characteristic",self.frame)
    def cv_init(self):
        self.cap = cv2.VideoCapture(0)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    #control arm from serial
    def sendData(self,motor,pos):
        # return control byte
        # the 'motor' has only two choice : '0' or '1'
        # the range of 'pos' is 500 - 2500, 500 means 0 degree, and 2500 means 90(?i m not sure) degree. 
        #对输入进行限制
        pos = int(pos)
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
    def serial_init(self):
        plist = list(serial.tools.list_ports.comports())
        if len(plist) <= 0:
            print("没有发现端口!")
        else:
            plist_0 = list(plist[0])
            serialName = plist_0[0]
            self.serialFd = serial.Serial(serialName, 9600, stopbits=1, parity=serial.PARITY_SPACE,timeout=60)
        if self.serialFd.isOpen():
            pass
        else:
            self.serialFd.open()
            print("可用端口名>>>", self.serialFd.name)
    def reset_rotor_place(self):
        #写入舵机速度
        self.serialFd.write(bytes.fromhex('ff 01 00 04 00')) 
        self.serialFd.write(bytes.fromhex('ff 01 01 04 00'))
        self.serialFd.write(bytes.fromhex('ff 01 02 04 00'))
        self.serialFd.write(bytes.fromhex('ff 01 03 04 00'))
        #使舵机位置初始化
        self.motorA_pos = 2500
        self.motorB_pos = 500
        self.motorC_pos = 500
        self.motorD_pos = 2500
        self.serialFd.write(self.sendData(0,2500))
        self.serialFd.write(self.sendData(1,500))
        self.serialFd.write(self.sendData(2,500))
        self.serialFd.write(self.sendData(3,2500))
        time.sleep(5)
    def update_rotor_pos(self):
        self.serialFd.write(self.sendData(0,self.motorA_pos))
        self.serialFd.write(self.sendData(1,self.motorB_pos))
        self.serialFd.write(self.sendData(2,self.motorC_pos))
        self.serialFd.write(self.sendData(3,self.motorD_pos))
        # time.sleep(1)
#jiaozhun----------------------------------------------
# if __name__ == "__main__":
#     a = softArm()
#     while(1):
#         a.get_from_cv()
