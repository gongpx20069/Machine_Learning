#-*-coding:utf-8-*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])#使用占位符，占用输入层
y_ = tf.placeholder(tf.float32,[None, 10])#使用占位符，占用输出层
#定义y=wx+b的计算函数,权重初始化
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#定义卷积层，strides为步长,padding保证输入和输出是同一个大小
#池化层的ksize为大小，strides为步长，padding='same'可以保证在尾部填充0，而不丢弃信息
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='
                      ')
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
#一个卷积层通常接一个maxpooling，卷积在每个5x5的patch中算出32个特征。卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。
#而对于每一个输出通道都有一个对应的偏置量b。
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
#将图像reshape，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数（灰度图像、二值图像、RGB）
x_image =tf.reshape(x, [-1,28,28,1])
#第一层卷积层输出relu，以及第一层池化层
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#第二层卷积和池化的定义
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
#第二层卷积和池化
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#池化计算公式：
#全连接层初始化，这时候图片尺寸为7×7,通道数64,加入1024全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
#全连接层计算
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#防止过拟合的dropout，keep_prob为起作用的神经元概率
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#全连接的输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
#计算输出结果
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#交叉熵用于损失函数，指定学习速率
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#tf.argmax可以给出tensor对象在某一维度上数据的最大值所在位置
#tf.equal可以检测预测值是否和真实值匹配，返回一个bool数组，eg:[True, False, True, True]
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#将bool数组变为[1,0,1,1]，计算出平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        #50个一组，迭代1000次
        batch = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1],keep_prob: 0.9})
        print 'step %d'%i,sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob: 1.0})
                      
                      
                      
'''
 The TensorFlow Convolution example gives an overview about the difference between SAME and VALID :

    For the SAME padding, the output height and width are computed as:

    out_height = ceil(float(in_height) / float(strides[1]))

    out_width = ceil(float(in_width) / float(strides[2]))

And

    For the VALID padding, the output height and width are computed as:

    out_height = ceil(float(in_height - filter_height + 1) / float(strides1))

    out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))

'''
