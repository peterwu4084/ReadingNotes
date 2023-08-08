# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

## Motivation
Attention计算需要大量时间和内存，时间和内存复杂度与序列长度呈二次关系。近似attention方法通过牺牲模型表型以减少计算复杂度从而解决上述问题，但是通常无法减少真正的执行时间。这些方法仅考虑了FLOP却忽略了内存访问的开销。而attention的速度主要受内存IO限制，而不是计算限制。

Flash attention正是考虑不同内存的读写速率设计的attention算法，具有更快的执行速度和更小的内存占用。

## Contribution
* 提出flash attention算法，具有更快的计算速度和内存占用，因而可以实现更快的模型训练和推理以及更大输入序列长度，进而提升模型的表现；
* 基于flash attention改进的稀疏注意力，可以进一步提升速度和节省内存；

## Background
* GPU Memory Hierarchy

    GPU内存结构包含不同形式的内存，这些内存具有不同的大小和速度，如图1左图所示。
    ![fig1](./assets/flashattention_fig1.png)

* Execution Model

    GPU中执行一个操作需要大量线程，需要将输入从HBM加载到寄存器和SRAM，执行计算，再将输出写入HBM

* Naive Attention Implementation

    给定输入序列 $Q,K,V\in R^{N\times d}$，其中$N$为序列长度，$d$为特征长度，attention的输出 $O\in R^{N \times d}$ 通过下式计算：
    
    $$S=QK^T\in R^{N\times N}$$
    
    $$P=softmax(S)\in R^{N\times N}$$
    
    $$O=PV\in R^{N\times d}$$

    标准attention在GPU上的实现如下所示：
    ![alg0](./assets/flashattention_alg0.png)

## Flash Attention

Flash attention的核心在于将 $Q,K,V$ 分割成小块，并将其从较慢的HBM加载到较快的SRAM，通过增量和迭代的方式完成输出$O$的计算。

* Tiling

    对于向量 $x\in R^B$的softmax，可以用过以下方式计算：
    $$ m(x):=\max x_i$$ 
    $$ f(x):=[e^{x_1-m(x), ... , e^{x_B-m(x)}}] $$
    $$ l(x):=\sum_i f(x)_i $$
    $$ softmax(x):=\frac{f(x)}{l(x)} $$

    对于向量 $x^{(1)},x^{(2)} \in R^B$，拼接向量 $x=[x^{(1)}, x^{(2)}]$的softmax可以以以下迭代的方式计算：

    