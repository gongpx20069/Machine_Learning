#-*-coding:utf-8-*-
import tensorflow as tf
import cv2
import numpy as np
import os
def conv2d(x, W, b, strides=1,is_train=True):  
        # Conv2D wrapper, with bias and relu activation  
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')  
    x = tf.nn.bias_add(x, b)  
    x = tf.layers.batch_normalization(x,training=is_train)
    return tf.nn.relu(x)  

def dw_conv2d(x, W, b, strides=1,is_train=True):  
        # Conv2D wrapper, with bias and relu activation  
    x = tf.nn.depthwise_conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')  
    x = tf.nn.bias_add(x, b)  
    x = tf.layers.batch_normalization(x,training=is_train)
    return tf.nn.relu(x)  

def avgpool2d(x, k,s):  
        # MaxPool2D wrapper  
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME')  
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
    
x = tf.placeholder(tf.float32, [None, 224,224,3])
y_ = tf.placeholder(tf.float32,[None, 1000])#
is_train=tf.placeholder(tf.bool)
x=tf.reshape(x, shape=[-1,224,224,3])
conv1 = conv2d(x, weight_variable([3,3,3,32]),bias_variable([32]),2,is_train)

conv2 = dw_conv2d(conv1, weight_variable([3,3,32,1]), bias_variable([32]),1,is_train)

conv3=conv2d(conv2, weight_variable([1,1,32,64]),bias_variable([64]),1,is_train)

conv4 = dw_conv2d(conv3, weight_variable([3,3,64,1]),bias_variable([64]),2,is_train)

conv5 = conv2d(conv4, weight_variable([1,1,64,128]),bias_variable([128]),1,is_train)

conv6 = dw_conv2d(conv5, weight_variable([3,3,128,1]),bias_variable([128]),1,is_train)

conv7=conv2d(conv6, weight_variable([1,1,128,128]),bias_variable([128]),1,is_train)

conv8=dw_conv2d(conv7, weight_variable([3,3,128,1]),bias_variable([128]),2,is_train)

conv9=conv2d(conv8, weight_variable([1,1,128,256]),bias_variable([256]),1,is_train)

conv10=dw_conv2d(conv9, weight_variable([3,3,256,1]),bias_variable([256]),2,is_train)

conv11=conv2d(conv10, weight_variable([1,1,256,512]),bias_variable([512]),1,is_train)

for i in range(0, 5):
    conv12 = dw_conv2d(conv11, weight_variable([3,3,512,1]),bias_variable([512]),1,is_train)
    conv13=conv2d(conv10, weight_variable([1,1,256,512]),bias_variable([512]),1,is_train)
    conv11=conv13

conv14 = dw_conv2d(conv13, weight_variable([3,3,512,1]),bias_variable([512]),2,is_train)
conv15=conv2d(conv14, weight_variable([1,1,512,1024]),bias_variable([1024]),1,is_train)

conv16 = dw_conv2d(conv15, weight_variable([3,3,1024,1]),bias_variable([1024]),2,is_train)
conv17=conv2d(conv16, weight_variable([1,1,1024,1024]),bias_variable([1024]),1,is_train)

pool1 = avgpool2d(conv17, k = 7,s=7)

h_pool1_flat = tf.reshape(pool1, [-1, 1024])

y_conv = tf.matmul(h_pool1_flat, weight_variable([1024, 1000])) +bias_variable([1000])

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
        #10个一组，迭代100次
        sess.run(train_step, feed_dict={x: train_x, y_:train_y,is_train:True})
        saver.save(sess,'Model/model.ckpt')
        print('step %d,[accuracy, loss]'%i,sess.run([accuracy,cross_entropy],feed_dict={x:test_x,y_:test_y,is_train:False}))
