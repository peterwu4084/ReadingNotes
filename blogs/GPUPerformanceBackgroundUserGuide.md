# [GPU Performance Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#undefined)

## Abstract

本指南提供了有关GPU结构的背景知识，操作是如何执行的，以及深度学习操作的常见限制。

## 1.Overview

在推理特定层或神经网络如何有效地利用给定GPU时，理解GPU执行的基础知识是有帮助的。

本指南介绍:

- GPU的基本结构 ([GPU Architecture Fundamentals](#2gpu-architecture-fundamentals))

- 操作是如何划分并并行执行的 ([GPU Execution Model](#3gpu-execution-model))

- 如何估计算术强度的性能限制 ([Understanding Performance](#4understanding-performance))

- 深度学习操作的松散分类以及每种操作的性能限制 ([DNN Operation Categories](#5dnn-operation-categories))

## 2.GPU Architecture Fundamentals

GPU是一种高度并行的处理器架构，由处理单元和内存层次结构组成。在高层次上，nvidia GPU由许多 Stream MultiProcessor (SM)、片上L2缓存和高带宽DRAM组成。算术和其他指令由SM执行;数据和代码通过L2缓存从DRAM访问。例如，NVIDIA A100 GPU包含108个SMs, 40MB L2缓存，可提供2039gb /s的带宽的80GB HBM2内存。

![gpu arch](./assets/gpu_arch.png)

每个SM都有自己的指令调度程序和各种指令执行管道。乘-加是现代神经网络中最常见的操作，作为全连接层和卷积层的构建块，两者都可以被视为向量点积的集合。下表显示了在NVIDIA最新的GPU架构上，单个SM对于不同数据类型的每个时钟的乘法加操作。每个乘法加操作包括两个操作，因此可以将表中的吞吐量乘以2来获得每个时钟的FLOP计数。

## 3.GPU Execution Model

## 4.Understanding Performance

## 5.DNN Operation Categories

## 6.Summary
