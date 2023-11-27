# 8-Bit Approximations for Parallelism in Deep Learning

## Motivation

创建实用的深度学习产品通常需要跨处理器和计算机并行化，使得深度学习在大型数据集上可行，但通信带宽的瓶颈使得通过并行性很难获得良好的加速。

## Contribution

作者提出了一种8bit近似算法，通过将32bit梯度和非线性激活压缩到8bit近似，更好地利用可用带宽。

这种近似不会影响深度学习在Mnist、Cifar10以及ImageNet上的表现。