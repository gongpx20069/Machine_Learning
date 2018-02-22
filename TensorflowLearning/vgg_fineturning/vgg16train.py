#-*-coding:utf-8-*-
import tensorflow as tf
import cv2
import numpy as np
import os
tf.logging.set_verbosity(tf.logging.INFO)

import random
#dirpath = 'birds'
dirpath=sys.argv[1]
index0 = 0
classis={}
imgs=[]
labels=[]
for filename in os.listdir(dirpath):
    filepath = os.path.join(dirpath,filename)
    classis[index0]=filename
    for img_name in os.listdir(filepath):
        img_path = os.path.join(filepath,img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224,224),interpolation=cv2.INTER_CUBIC)
        imgs.append(img)
        labels.append(index0)
    index0+=1
print(classis)
def getbetch(num):
    temp1 = []
    temp2=[]
    index=np.random.randint(len(labels),size=num).tolist()
    for i in index:
        temp = np.zeros(index0).tolist()
        temp[labels[i]]=1
        temp1.append(imgs[i])
        temp2.append(temp)
    temp1=np.array(temp1).astype(np.float32)
    temp2=np.array(temp2).astype(np.float32)
    return temp1,temp2
print(getbetch(10))
print('ok')
import vgg16
vgg=vgg16.Vgg16()
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
y_ = tf.placeholder(tf.float32,[None, index0])#

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

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

log_path='tf_writer'
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ha_x, ha_y = getbetch(25)
    saver.restore(sess,'Model/model.ckpt')
    for j in range(200):
        print('epoch%d/200'%(j+1))
        for i in range(50):
            haha_x, haha_y = getbetch(10)
            sess.run(train_step, feed_dict={x: haha_x, y_:haha_y,keep_prob:0.9})
            print('step %d,[accuracy, loss]'%i,sess.run([accuracy,cross_entropy], feed_dict={x: ha_x, y_:ha_y,keep_prob:1}))
			
        saver.save(sess,'Model/model.ckpt')

