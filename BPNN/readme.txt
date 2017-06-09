BPNN.py中包含四个可操作函数。
1> temp = BPNN(input, output, out_num = 10, hid_num =9,lrate = 0.5)
input:训练集合（每一行为一个训练样本，行数为迭代次数）
output:输出集合（为单列矩阵）
out_num:输出神经元个数(通常与理想输出的取值范围有关)
hid_num:隐含层神经元个数，默认为9
lrate:学习速度，默认为0.5
2>predicate(x)
x:测试样本的矩阵，格式与训练集合相同
3>exportTrain()
将训练的神经网络配置文件导出，会在本地生成（BPNN.config, w1, w2, hid_offset, out_offset）五个配置文件，供下次使用导入
4>importTrain()
导入存储的神经网络配置文件。
