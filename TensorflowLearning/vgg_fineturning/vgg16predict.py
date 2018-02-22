#-*-coding:utf-8-*-
import tensorflow as tf
import cv2
import numpy as np
import os
import sys
import random
import vgg16

#dirpath = 'birds'
dirpath = sys.argv[1]
imgpath=sys.argv[2]
print("图片路径为：",imgpath)
index0 = 0
classis={}

for filename in os.listdir(dirpath):
    filepath = os.path.join(dirpath,filename)
    classis[index0]=filename
    index0+=1
#print(classis)

def getimg(imgpath):
    img = cv2.imread(imgpath)
    img = cv2.resize(img, (224,224),interpolation=cv2.INTER_CUBIC)
    img = [img]
    return np.array(img).astype(np.float32)

vgg=vgg16.Vgg16()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

x = tf.placeholder(tf.float32, [None, 224,224,3])

vgg.build(x)
pool5=vgg.pool5

h_pool2_flat = tf.reshape(pool5, [-1, 7*7*512])
#h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weight_variable([7 * 7 * 512, 4096]) + bias_variable([4096])))

#防止过拟合的dropout，keep_prob为起作用的神经元概率
keep_prob = tf.placeholder(tf.float32)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, weight_variable([4096, 4096]) + bias_variable([4096])))
h_fc2_drop = tf.nn.dropout(h_pool2_flat, keep_prob)
#全连接的输出层
#计算输出结果
y_conv = tf.matmul(h_fc2_drop, weight_variable([7 * 7 * 512, index0])) +bias_variable([index0])

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,'Model/model.ckpt')
    result=sess.run(y_conv, feed_dict={x:getimg(imgpath),keep_prob:1})
    print("最终结果为",classis[np.argmax(result[0])])
    

