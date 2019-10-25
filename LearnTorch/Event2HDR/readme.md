## 1. 关于论文说明

《Event-based High Dynamic Range Image and Very High Frame Rate Video Generation using Conditional Generative Adversarial Networks》是2018年发表在CVPR上的一篇论文，其核心在于使用条件生成对抗网络，以Event Camera采集的Events为输入，以HDR图像为输出。
整体网络基本是参照Pix2Pix的U-net生成模型和PatchGAN判别模型，损失函数也基本与Pix2Pix一致。

## 2. 关于数据集

目前该数据集还未开源，但我们联系论文作者王霖，非常感谢王博士的支持，我们得到了一部分数据集。
但我们不打算将其开源，根据王博士的回信，在年底该数据集会放在他们实验室官网。

![输入图像和输出图像](https://github.com/gongpx20069/Machine_Learning/blob/master/LearnTorch/Event2HDR/image/4779.png "InputOutput")

## 3. 关于训练好的模型

[模型下载地址](https://github.com/gongpx20069/Machine_Learning/releases/tag/Event2HDR)
