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

class softArm:
    def __init__(self):
        # parameters
        self.name = 'softArm'
        self.enable_actions = (0, 1, 2, 3)
        self.action_bound = [[1900,2500],[500,1100],[500,1100],[1900,2500]]
        self.load_model()
        self.reset()
    def load_model(self):
        workbook = xlrd.open_workbook('Excel_Workbook220.xls')
        sheet1 = workbook.sheet_by_index(0)
        
        self.row_num = len(sheet1.col_values(0))
        self.input_data = np.zeros([self.row_num,4])
        self.input_data[:,0] = array(sheet1.col_values(0))
        self.input_data[:,1] = array(sheet1.col_values(1))
        self.input_data[:,2] = array(sheet1.col_values(2))
        self.input_data[:,3] = array(sheet1.col_values(3))

        self.minMax0 = MinMaxScaler()
        self.input_data1 = self.minMax0.fit_transform(self.input_data)
        output_data = np.zeros([self.row_num,12])
        output_data[:,0] = array(sheet1.col_values(4))
        output_data[:,1] = array(sheet1.col_values(5))
        output_data[:,2] = array(sheet1.col_values(6))
        output_data[:,3] = array(sheet1.col_values(7))
        output_data[:,4] = array(sheet1.col_values(8))
        output_data[:,5] = array(sheet1.col_values(9))
        output_data[:,6] = array(sheet1.col_values(10))
        output_data[:,7] = array(sheet1.col_values(11))
        output_data[:,8] = array(sheet1.col_values(12))
        output_data[:,9] = array(sheet1.col_values(13))
        output_data[:,10] = array(sheet1.col_values(14))
        output_data[:,11] = array(sheet1.col_values(15))

        self.minMax2 = MinMaxScaler()
        self.output_data1 = self.minMax2.fit_transform(output_data)

        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(64,input_shape=(4,),activation='tanh'))
        self.model.add(keras.layers.Dense(64, activation='tanh'))
        self.model.add(keras.layers.Dense(64, activation='tanh'))
        self.model.add(keras.layers.Dense(64, activation='tanh'))
        self.model.add(keras.layers.Dense(64, activation='tanh'))
        self.model.add(keras.layers.Dense(64, activation='tanh'))
        self.model.add(keras.layers.Dense(12, activation='tanh'))
        opt =keras.optimizers.Adam(lr=0.001)
        self.model.compile(loss='mse',optimizer=opt, metrics=['accuracy'])
        # self.model.load_weights('123.h')
        # self.model.fit(self.input_data1,self.output_data1,epochs=10000,verbose=2)
        # self.model.save_weights("123.h")
        self.model.load_weights('220.h')
    def draw(self):
        aa = self.minMax2.inverse_transform(self.current_pos)       
        plt.plot(aa[0][0:6],-aa[0][6:12],'-')
        plt.hold(True)
        plt.plot(aa[0][0],-aa[0][6],'o')
        plt.hold(True)
        aa = self.minMax2.inverse_transform(self.target)
        plt.plot(aa[0][0],-aa[0][6],'*')
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
            print('?')
        if self.motorA_pos!=2500 and self.motorB_pos!=500:
            print('position error')
        if self.motorC_pos!=500 and self.motorD_pos!=2500:
            print('position error')
        # collision detection
        self.reward = 0
        q = array([[self.motorA_pos,self.motorB_pos,self.motorC_pos,self.motorD_pos]])
        input_ = self.minMax0.transform(q)
        self.current_pos = self.model.predict(input_)
        A_target= self.minMax2.inverse_transform(self.target) 
        B_current_pos= self.minMax2.inverse_transform(self.current_pos) 

        #-----------------last pos method---------------
        c_last_pos= self.minMax2.inverse_transform(self.last_pos) 
        #--------------------------------------------------
        # self.distance = 0
        # self.cur_distance = 0
        self.last_distance = 0
        self.terminal_dis = 0
        
            # self.distance += abs(A[0][current_point] - B[0][current_point])# 奖励为距离的绝对值
            #-----------------last pos method---------------------
        self.terminal_dis = abs(A_target[0][0]-B_current_pos[0][0]) + abs(A_target[0][6]-B_current_pos[0][6])
        
        self.last_distance = abs(A_target[0][0]-c_last_pos[0][0]) + abs(A_target[0][6]-c_last_pos[0][6])

        # self.distance = self.last_distance - self.terminal_dis
        self.last_pos = self.current_pos
        #-----------------------------------------------------------
        
        self.run_step += 1
        self.terminal = False
        
        if self.terminal_dis < 15:#15
            # catch
            self.terminal = True
            self.reward = 2
        else:
            if self.over_boundary == 1:
                self.reward = -1
            elif self.terminal_dis < 20:
                self.reward = 1.2 - 0.005*self.terminal_dis
                self.step = 5
            elif self.terminal_dis < 50:
                self.reward = 1 - 0.005*self.terminal_dis
                self.step = 10
            elif self.terminal_dis < 90:
                self.reward = 0.5 - 0.002*self.terminal_dis
                self.step = 25
            elif self.terminal_dis < 150:
                self.reward = 0.1  - 0.001*self.terminal_dis
                self.step = 50
            else:
                self.reward =  - 0.001*self.terminal_dis
                self.step = 75

        if self.run_step > 150:
            self.terminal = True
    def observe(self):
        # self.draw()
        envir = []
        target = np.array([self.target[0][0],self.target[0][6]])
        current_pos = np.array([self.current_pos[0][0],self.current_pos[0][6]])
        envir.extend(target.flatten())
        envir.extend(current_pos.flatten())
        envir = np.array(envir)
        return envir, self.reward, self.terminal
    def execute_action(self, action):
        self.update(action)
    def reset(self):
        # reset player position
        start_pos = np.random.randint(0,self.row_num)
        self.motorA_pos = self.input_data[start_pos,0]
        self.motorB_pos = self.input_data[start_pos,1]
        self.motorC_pos = self.input_data[start_pos,2]
        self.motorD_pos = self.input_data[start_pos,3]
        end_pos = np.random.randint(0,self.row_num)
        # reset taget position
        c = self.input_data[end_pos,0]
        d = self.input_data[end_pos,1]
        e = self.input_data[end_pos,2]
        f = self.input_data[end_pos,3]
        q = array([[c,d,e,f]])
        input_ = self.minMax0.transform(q)
        self.target = self.model.predict(input_)
        
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
    #function to get_pic from cv
    def get_from_cv(self):
        cap = cv2.VideoCapture(0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        for i in range(50):
                ret, frame = cap.read()
        while(1):
            # get a frame
            
            ret, frame = cap.read()
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
                yuanxin = (yuanxin_x,yuanxin_y)
                cv2.circle(frame,yuanxin,10,color)
            cv2.waitKey(50)
            cv2.imshow("get_characteristic",frame)

        cap.release()

