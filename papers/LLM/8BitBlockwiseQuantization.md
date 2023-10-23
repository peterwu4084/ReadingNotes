# 8-Bit Optimizers via Block-wise Quantization

## Motivation

训练大模型需要大量的内存，而如何减少优化器的内存占用还没有有效的解决方案。已有工作者在16位优化器上进行尝试，但8位优化器仍没有成功的工作。8位优化器存在如下三个挑战：

1. 量化准确率：维持高精度，必须使用非线性量化，减少出现频率高的小幅度值和罕见的大幅度值。

2. 计算效率：引入非线性量化导致训练速度下降。

3. 大模型稳定性：为了维持稳定的训练，量化方法不仅需要有较低的平均量化误差，同时最差情况下也要有较好的表现。

 

## Contribution



SGD with momentum and Adam limit the max size fo models.

Develop the first optimizers using 8-bit statistics while maintaining the performance levels of 32-bit.

Block-wise dynamic quantization:

1. split input tensors into blocks and performs quantization on each block indenpently

2. combine with two novel methods for stable:

   - dynamic quantization
   - stable embedding layer

hard points

1. quantization acc: critical non-linear quantization

2. compute efficiency: non-linear is difficult to be fast

3. large-scale stability: not only good mean error but also excellent worse case performance

## Background

### Stateful Optimizers

![optimizer](./assets/blockwiseQ_optimizer.png)

For 32-bit states, Momentum and Adam consume 4 and 8 bytes per parameter.

### Non-Linear Quantization

### Dynamic Tree Quantization

![dynamic tree quantization](./assets/blockwiseQ_dynamic_tree_q.png)

## 8-Bit Optimizers

1. dequantize 8-bit optimizer states to 32-bit, element-by-element in registers

2. perform update

3. quantize the states back to 8-bit for storage

### Blockwise Quantization

reduce the cost of computing normalization and improve quantization precision by isolating outliers.

