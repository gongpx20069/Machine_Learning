# 1. Machine_Learning

这是我学习机器学习的一个路程，也是我自己编写的机器学习包。将Tensorflow进行一定的封装处理，使得工作更为便利。同时在编写代码的过程中，我们可以从原理上理解机器学习算法的具体执行流程，理解神经网络的自然传播过程。

## 1.1 文件中目前有算法包:

1.BPNN（BP神经网络）;(BPNN是一个比较简单的神经网络，但我对其中反向传播过程中的一些数学公式没有理解透彻，算法应该还可以优化。使用前请阅读BPNN中的readme,BPNN是之前写的算法包，现在看来有很多疏漏，不能在实际生产中使用，只是学习的一个过程)

2.TensorflowLearning(Tensorflow实例CNN详解); 复现的VGG-16和moblilenetV1神经网络。(VGG16和MobilnetV1网络复现很完全，可以直接使用，但没有fineturn的话，数据集和训练时间都一定要大且长)；有vgg16的fineturn，其中的vgg16模型可以单独使用；

3.Estimate(评估算法包)；目前里面有评估图像清晰度的算法

4.TensorTool；主要有一些简单的拼装函数用于整体网络的搭建，比如ops.py中有神经网络的层级结构;

5.LearnTorch：包含了一些复现的pyTorch论文代码，详细解析看下文；

## 1.2 一些重要包说明

### 1.2.1 TensorTool

主要有一些简单的拼装函数用于整体网络的搭建，比如ops.py中有神经网络的层级结构;Opt.py中有关于优化器的接口；

### 1.2.2 LearnTorch

包含了一些使用pytorch复现的论文，目前包含的论文有：

[《Event-based High Dynamic Range Image and Very High Frame Rate Video Generation using Conditional Generative Adversarial Networks》](https://github.com/gongpx20069/Machine_Learning/tree/master/LearnTorch/Event2HDR)

# 2. 强烈推荐

1, [vgg_fineturning](https://github.com/gongpx20069/Machine_Learning/tree/master/TensorflowLearning/vgg_fineturning) (可以直接使用，训练成本低，网络收敛快，能达到很高的准确度)
