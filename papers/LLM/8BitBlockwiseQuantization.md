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

量化 （或int8,float16,float32）是一个映射，将k位整数映射为一个实数，$ Q^{map}: [0, 2^k-1] \rightarrow R $.

将一个数据类型 $Q^{map}_1$ （如fp32）量化为另一个数据类型 $Q^{map}_2$（如int8），需要以下三个步骤：

1. 将 $Q^{map}_1$ 的值域通过正则化常数 $N$ 归一化到 $Q^{map}_2$的值域范围；

2. 对于每一个 $Q^{map}_1(i) / N$，找到 $Q^{map}_2$ 的值域中最接近的值 $Q^{map}_2(j_i) $.

3. 反量化则是将 $Q^{map}_2(j_i)$ 乘以 $N$。

### Dynamic Tree Quantization

动态数量化是一种量化方法，可以同时对大幅度值和小幅度值产生较小的量化误差。不同于int8、fp16、fp32等数据类型，动态树量化使用的数据类型具有动态的指数位和小数位，如下图所示。该数据由4部分组成：

1. 符号位，数据的第一位。

2. 指数位，由符号位后连续的0表示，连续 $i$ 个 $0$ 表示指数为为 $1e-i$.

3. 分隔位，指数位后第一个 $1$表示，用于分隔指数位和小数位。

4. 小数位，分隔位后续均为小数位，采用线性量化，假设有 $j$ 位，小数位从小到大$[0, 1, ..., 2^j-1]$对应了小数 $[0, 1 / (2^j-1), ..., 1]$.

所以其能表示的范围为 $[-1, 1]$.

![dynamic tree quantization](./assets/blockwiseQ_dynamic_tree_q.png)

## 8-Bit Optimizers

1. dequantize 8-bit optimizer states to 32-bit, element-by-element in registers

2. perform update

3. quantize the states back to 8-bit for storage

### Blockwise Quantization

reduce the cost of computing normalization and improve quantization precision by isolating outliers.

