#-*-coding:utf-8-*-
'''
在train_x和train_y中直接导入训练集
在test_x和test_y中导入测试集
注意:x.shape=[None,224,224,3], y.shape=[None,1000]
根据具体要求改变输出神经元的个数
'''
import tensorflow as tf
import cv2
import numpy as np
import os

def conv2d(x, W, b, strides=1):  
        # Conv2D wrapper, with bias and relu activation  
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')  
    x = tf.nn.bias_add(x, b)  
    return tf.nn.relu(x)  
def maxpool2d(x, k=2):  
        # MaxPool2D wrapper  
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')  
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

x = tf.placeholder(tf.float32, [None, 224,224,3])
y_ = tf.placeholder(tf.float32,[None, 1000])#
x=tf.reshape(x, shape=[-1,224,224,3])
conv1_1 = conv2d(x, weight_variable([3,3,3,64]),bias_variable([64]),1)
conv1_2 = conv2d(conv1_1,weight_variable([3,3,64,64]),bias_variable([64]),1)
pool1 = maxpool2d(conv1_2, k = 2)

conv2_1 = conv2d(pool1, weight_variable([3,3,64,128]),bias_variable([128]),1)
conv2_2 = conv2d(conv2_1,weight_variable([3,3,128,128]),bias_variable([128]),1)
pool2 = maxpool2d(conv2_2, k = 2)

conv3_1 = conv2d(pool2, weight_variable([3,3,128,256]),bias_variable([256]),1)
conv3_2 = conv2d(conv3_1,weight_variable([3,3,256,256]),bias_variable([256]),1)
conv3_3 = conv2d(conv3_2,weight_variable([3,3,256,256]),bias_variable([256]),1)
pool3 = maxpool2d(conv3_3, k = 2)

conv4_1 = conv2d(pool3, weight_variable([3,3,256,512]),bias_variable([512]),1)
conv4_2 = conv2d(conv4_1,weight_variable([3,3,512,512]),bias_variable([512]),1)
conv4_3 = conv2d(conv4_2,weight_variable([3,3,512,512]),bias_variable([512]),1)
pool4 = maxpool2d(conv4_3, k = 2)

conv5_1 = conv2d(pool4, weight_variable([3,3,512,512]),bias_variable([512]),1)
conv5_2 = conv2d(conv5_1,weight_variable([3,3,512,512]),bias_variable([512]),1)
conv5_3 = conv2d(conv5_2,weight_variable([3,3,512,512]),bias_variable([512]),1)
pool5 = maxpool2d(conv5_3, k = 2)

h_pool2_flat = tf.reshape(pool5, [-1, 7*7*512])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weight_variable([7 * 7 * 512, 4096]) + bias_variable([4096])))

#防止过拟合的dropout，keep_prob为起作用的神经元概率
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, weight_variable([4096, 4096]) + bias_variable([4096])))
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
#全连接的输出层
#计算输出结果
y_conv = tf.matmul(h_fc2_drop, weight_variable([4096, 1000])) +bias_variable([1000])

#交叉熵用于损失函数，指定学习速率
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#tf.argmax可以给出tensor对象在某一维度上数据的最大值所在位置
#tf.equal可以检测预测值是否和真实值匹配，返回一个bool数组，eg:[True, False, True, True]
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#将bool数组变为[1,0,1,1]，计算出平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        #10个一组，迭代1000次
        haha_x, haha_y = getbetch(50)
        print('step %d start'%i)
        sess.run(train_step, feed_dict={x: train_x, y_:train_y,keep_prob: 0.9})
        saver.save(sess,'Model/model.ckpt')
        ha_x, ha_y = getbetch(50)
        print('step %d'%i,sess.run(accuracy,feed_dict={x:test_x,y_:test_y,keep_prob: 1.0}),'loss:',sess.run(cross_entropy,feed_dict={x:test_x,y_:test_y,keep_prob: 1.0}))
