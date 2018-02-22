VGG_fineturning（vgg迁移学习，利用预训练好的vgg卷积层训练小数据或快速训练）
=====================================================================

预先准备：
----------

1, python

2, tensorflow

3, vgg16.npy文件[官方地址](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM), [百度云下载](https://pan.baidu.com/s/1gg9jLw) 百度云密码：umce

4, 将vgg16.npy文件放入vgg_finturning目录下

5, 将想要训练的文件放到vgg_finturning目录下，子目录为分类文件，文件名为分类名。eg:birds

代码说明：
-----------

1, 训练代码vgg16train.py

···
python3 vgg16train.py [文件名eg:birds]
···

2, 测试代码vgg16predict.py

···
python3 vgg16predict.py [文件名eg:birds] [图片路径eg:birds/owl/owl001.jpg]
···
