# 8-Bit Optimizers via Block-wise Quantization

## Motivation

训练大模型需要大量的内存，其中优化器占用了很大的比例，而如何减少优化器的内存占用还没有有效的解决方案。已有工作者在16位优化器上进行尝试，但8位优化器仍没有成功的工作。同时，8位优化器存在如下三个挑战：

1. 量化准确率：维持高精度，必须使用非线性量化，减少出现频率高的小幅度值和罕见的大幅度值。

2. 计算效率：引入非线性量化导致训练速度下降。

3. 大模型稳定性：为了维持稳定的训练，量化方法不仅需要有较低的平均量化误差，同时最差情况下也要有较好的表现。

## Contribution

- 提出基于块的量化方法：将张量分块，并对每块进行单独量化。该方法存在以下优点：

  - 因为异常值被隔离在特定的块中，减弱异常值的影响，提升性能和稳定性。

  - 不同的块可以并行量化，提升速度。

- 结合两个新的方法提升稳定性：

  - 动态量化（dynamic quantization），基于动态树量化针对无符号数据的一种拓展。

  - 稳定嵌入层，标准嵌入层的拓展，通过归一化高度非均匀分布以支持更好的量化，避免极端梯度。

- 该8位优化器和32位优化器具有相同的表现，内存占用更少。

## Background

### Stateful Optimizers

![optimizer](./assets/blockwiseQ_optimizer.png)

对于32位优化器，Momentum和Adam每个参数分别消耗4和8个字节

### Non-Linear Quantization

### Dynamic Tree Quantization

![dynamic tree quantization](./assets/blockwiseQ_dynamic_tree_q.png)

## 8-Bit Optimizers

1. dequantize 8-bit optimizer states to 32-bit, element-by-element in registers

2. perform update

3. quantize the states back to 8-bit for storage

### Blockwise Quantization

reduce the cost of computing normalization and improve quantization precision by isolating outliers.

