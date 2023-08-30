# Ray Core导览: 远程任务

## 介绍

Ray允许任意Python函数在单独的Python工作线程上异步执行。这些异步Ray函数称为“Task”。可以通过cpu、gpu和自定义资源来指定任务的资源需求。集群调度器使用这些资源请求在集群中分发任务，以便并行执行。

![ray_tasks_actors_immutable_objects](./assets/ray_tasks_actors_immutable_objects.png)

![ray_tasks](./assets/ray_tasks.png)

## 学习目标

在本教程中，你将学习:

- 远程任务并行模式

- 作为分布式任务的无状态远程函数

- 串行与并行执行

- 理解Ray task的概念

- 简单的API将现有的Python函数转换为Ray远程任务

- 通过示例分别比较串行与分布式Python函数和Ray任务

## 任务并行模式

Ray通过 `@ray.remote` 修饰函数，使其成为无状态任务，在集群中的Ray节点的工作器上调度。

它们将在集群上的何处执行(以及在哪个节点上由哪个工作进程执行)，您不必担心其细节。一切都已安排好了，所有的工作都由Ray完成。您只需将现有的Python函数转换为分布式无状态Ray任务：就这么简单!

### 串行与并行执行

作为常规Python函数的串行任务以顺序的方式执行，如下图所示。如果我启动十个任务，它们将一个接一个地在单个worker上运行。

![timeline_of_sequential_tasks](./assets/timeline_of_sequential_tasks.png)

与串行执行相比，Ray任务是并行执行的，调度在不同的工作器上。Raylet将根据调度策略调度这些任务。

![sample_timeline_of_parallel_tasks](./assets/sample_timeline_of_parallel_tasks.png)

让我们对比一些任务的串行运行和并行运行。为了说明，我们将使用以下任务:

- 生成斐波那契数列

- 用蒙特卡罗方法计算 $\pi$

- 转换和处理大型高分辨率图像

- 使用Ray Task进行批推理

但首先，让我们了解一些基本概念: 原始Python函数和修饰后的函数之间存在一些关键区别:

- 调用: 使用 `func_name()` 调用常规版本，而使用 `func_name.remote()` 调用远程Ray版本。所有Ray远程执行方法都是这个模式。

- 执行方式和返回值: Python 常规版本的函数同步执行并返回结果，而Ray任务 `func_name.remote()` 立即返回 `ObjectRef`，然后在远程工作进程的后台执行任务。通过在 `ObjectRef` 上调用 `ray.get(ObjectRef)` 来获得结果，这是一个阻塞函数。

让我们在本地机器上启动一个Ray集群。

``` python
import os
import time
import logging
import math
import random

from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tqdm
import ray

if ray.is_initialized:
    ray.shutdown()
ray.init(logging_level=logging.ERROR)
```
