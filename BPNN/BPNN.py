# -*- coding:utf-8 -*-
"""
auther:Greepex
代码用于学习交流，若有大错，还请不吝赐教。若是用于其他用途，请在开头声明本段文字。
谢谢
"""
import numpy as np
import sys
import pandas
reload(sys)
sys.setdefaultencoding("utf-8")

class BPNN(object):
    out_num = 10
    hid_num = 9
    lrate = 0.5
    def __init__(self):
        return
    def __init__(self, input, output, out_num = 10, hid_num =9,lrate = 0.5):
        self.out_num = out_num
        self.hid_num = hid_num
        self.lrate = lrate
        self.in_num = input.shape[1] #获得输入的列数
        self.loop_num=input.shape[0] #获得迭代次数,训练样本的大小
        self.w1 = 0.2*np.random.random((self.in_num,self.hid_num))-0.1
        self.w2 = 0.2*np.random.random((self.hid_num,self.out_num))-0.1
        self.hid_offset=np.mat(np.zeros(self.hid_num))
        self.out_offset=np.mat(np.zeros(self.out_num))
        for i in range(0, self.loop_num):
            label = np.zeros(self.out_num)
            label[int(output[i, 0])] = 1#理想的输出
            print label
            # print np.argmax(label)
            #前向神经网络
            hid_value = np.dot(input[i],self.w1)+self.hid_offset #隐含值
            hid_act = self.act_fuction(hid_value,False)#隐含层激活值
            out_value = np.dot(hid_act, self.w2) +self.out_offset#输出层值
            out_act = self.act_fuction(out_value,False)#输出层激活
            #反向传播过程
            e = label-out_act
            out_delta = np.multiply(e,self.act_fuction(out_act,True)) #输出层delta计算
 #           print self.w2.shape, out_delta.shape
            #隐含层delta计算
            hid_delta = np.multiply(np.dot(self.w2, out_delta.T), self.act_fuction(hid_act,True))
            for k in range(0,self.hid_num):
                for j in range(0, self.out_num):
                    self.w2[k, j] += self.lrate * out_delta[0, j] * hid_act[0, k]
                for j in range(0, self.in_num):
                    self.w1[j,k] += self.lrate*hid_delta[0,k]*input[i,j]
            self.out_offset += self.lrate*out_delta[0] #更新偏置
            self.hid_offset +=self.lrate*hid_delta[0]
        pass
    def act_fuction(self, x, flag):
        if flag == True:
            return np.multiply(x, 1-x)
        return 1 / (1 + np.exp(-x))
    def predicate(self, x):
        result = []
        for i in x:
            hid_value = np.dot(i, self.w1)+self.hid_offset #隐含值
            hid_act = self.act_fuction(hid_value,False)#隐含层激活值
            out_value = np.dot(hid_act, self.w2) +self.out_offset#输出层值
            out_act = self.act_fuction(out_value,False)#输出层激活
            result.append(np.argmax(out_act))
            pass
        return result
    def exportTrain(self):
        with open("BPNN.config","wb") as f:
            in_hid_out_lirate_offset = str(self.in_num)+','+str(self.hid_num)+','+str(self.out_num)+','+str(self.lrate)
            f.write(in_hid_out_lirate_offset)
            pass
        save = pandas.DataFrame(self.hid_offset)
        save.to_csv('hid_offset', index=False, sep=',')
        save = pandas.DataFrame(self.out_offset)
        save.to_csv('out_offset', index=False, sep=',')
        save = pandas.DataFrame(self.w1)
        save.to_csv('w1', index=False, sep=',')
        save = pandas.DataFrame(self.w2)
        save.to_csv('w2', index=False, sep=',')
        pass
    def importTrain(self):
        with open("BPNN.config","rb") as f:
            string = f.read().split(',')
            self.in_num = int(string[0])
            self.hid_num=int(string[1])
            self.out_num=int(string[2])
            self.lrate = float(string[3])
            pass
        self.hid_offset = np.mat(pandas.read_csv("hid_offset", sep=',', header=None))
        self.hid_offset = np.delete(self.hid_offset, 0, 0)
        self.out_offset = np.mat(pandas.read_csv("out_offset", sep=',', header=None))
        self.out_offset = np.delete(self.out_offset, 0, 0)
        self.w1 = np.mat(pandas.read_csv("w1", sep=',', header=None))
        self.w1 = np.delete(self.w1, 0, 0)
        self.w2 = np.mat(pandas.read_csv("w2", sep=',', header=None))
        self.w2 = np.delete(self.w2, 0, 0)
def main():
    x=BPNN(np.mat([[1,2,3,0,0,0,0,0,0,0,0],[3,4,5,0,0,0,0,0,0,0,0]]),np.mat([8,5]).T)
    print x.predicate(np.mat([[1,2,3,0,0,0,0,0,0,0,0],[3,4,5,0,0,0,0,0,0,0,0]]))
    x.exportTrain()
    x.importTrain()
