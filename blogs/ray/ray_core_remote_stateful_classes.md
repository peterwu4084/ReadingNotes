# A Guided Tour of Ray Core: Remote Stateful Classes

## 概述

Actor将Ray API从函数(任务)扩展到类。Actor本质上是一个有状态的工作器(或服务)。当实例化一个新的actor时，将创建一个新的工作器或使用一个现有的工作器。Actor的方法被安排在特定的工作器上，并且可以访问和改变该工作器的状态。与task一样，actor也支持CPU、GPU和自定义资源需求。

## 学习目标

在本教程中，我们将讨论Ray actor:

- Ray actor的工作原理

- 如何编写一个有状态的Ray actor

- Ray actor如何被编写成一个有状态的分布式服务

远程类(就像远程任务一样)使用 `@rayremote` 装饰Python类。

Ray actor模式是强大的。它们允许你获取一个Python类并将其实例化为一个有状态的微服务，可以从其他actor和task甚至其他Python应用程序中查询。Actor可以作为参数传递给其他任务和actor。

当您实例化一个远程actor时，一个单独的工作进程被附加到一个工作进程上，并成为该工作节点上的一个actor进程——所有这些都是为了运行在该actor上调用的方法。其他Ray task和actor可以在该进程上调用它的方法，如果需要，可以改变它的内部状态。如果需要，也可以手动终止actor。

## 远程类作为有状态actor

### 例1：actor的方法跟踪

问题：我们想要跟踪在不同actor中谁调用了 特定方法。这可能是遥测数据的用例，我们希望跟踪正在使用的actor及其各自调用的方法。或者哪些Actor服务的方法最常被访问或使用。

让我们使用这个actor来跟踪Ray actor方法的方法调用。每个Ray actor实例将跟踪它的方法被调用了多少次。定义一个基类ActorCls，并定义两个子类ActorClsOne和ActorClsTwo。

``` python
import logging
import time
import os
import math
import random
import tqdm

from typing import Dict, Tuple, List
from random import randint

import numpy as np
import ray

if ray.is_initialized:
    ray.shutdown()
ray.init(logging_level=logging.ERROR)


class ActorCls:
    def __init__(self, name: str):
        self.name = name
        self.method_calls = {"method": 0}
        
    def method(self, **args) -> None:
        pass
    
    def get_all_method_calls(self) -> Tuple[str, Dict[str, int]]:
        return self.get_name(), self.method_calls
        
    def get_name(self) -> str:
        return self.name


@ray.remote
class ActorClsOne(ActorCls):
    def __init__(self, name: str):
        super().__init__(name)

    def method(self, **args) -> None:
        time.sleep(args['timeout'])
        self.method_calls['method'] += 1


@ray.remote
class ActorClsTwo(ActorCls):
    def __init__(self, name: str):
        super().__init__(name)

    def method(self, **args) -> None:
        time.sleep(args['timeout'])
        self.method_calls['method'] += 1


# 随机调用
# 使用class_name.remote(args)创建一个actor实例
# 要调用actor的方法，只需使用actor_instance.method_name.remote(args)
actor_one = ActorClsOne.remote("ActorClsOne")
actor_two = ActorClsTwo.remote("ActorClsTwo")

CALLERS_NAMES = ["ActorClsOne", "ActorClsTwo"]
CALLERS_CLS_DICT = {
    "ActorClsOne": actor_one,
    "ActorClsTwo": actor_two
}

# 随机调用每个Actor的方法，同时保持本地跟踪以进行验证
count_dict = {"ActorClsOne": 0, "ActorClsTwo": 0}
for _ in range(len(CALLERS_NAMES)):
    for _ in range(15):
        name = random.choice(CALLERS_NAMES)
        count_dict[name] += 1
        CALLERS_CLS_DICT[name].method.remote(timeout=1, store='mongo_db') if name == 'ActorClsOne' else \
        CALLERS_CLS_DICT[name].method.remote(timeout=1.5, store='delta')
    print(f'State of counts in this execution: {count_dict}')
    time.sleep(0.5)

print(ray.get([CALLERS_CLS_DICT[name].get_all_method_calls.remote() for name in CALLERS_NAMES]))

ray.shutdown()
```

在上面的例子中，我们看到了如何使用actor来跟踪调用它的方法的次数。如果您有兴趣获取作为服务部署的actor的使用，那么这可能是遥测数据的一个有用示例。

### 示例2:使用Actor跟踪进度

问题：在我们的第一个教程中，我们探索了如何仅使用任务来近似 $\pi$ 的值。在这个例子中，我们通过定义一个Ray actor来扩展它，这个actor可以被我们的Ray采样任务调用来更新进度。采样Ray任务向Ray actor发送消息(通过方法调用)以更新进度。

让我们定义一个做以下事情的Ray actor:

- 跟踪每个任务id及其完成的任务

- 可以被采样任务调用(或向其发送消息)来更新进度

``` python
@ray.remote
class ProgressPIActor:
    def __init__(self, total_num_samples: int):
        # 所有任务的所有样本总数
        self.total_num_samples = total_num_samples
        # 跟踪每个任务id的字典
        self.num_samples_completed_per_task = {}
        
    def report_progress(self, task_id: int, num_samples_completed: int) -> None:
        # 更新任务id完成的更新示例
        self.num_samples_completed_per_task[task_id] = num_samples_completed
        
    def get_progress(self) -> float:
        return sum(self.num_samples_completed_per_task.values()) / self.total_num_samples
```

在之前的任务教程中，我们定义了一个Ray任务，它将采样到 `num_samples` 并返回圆圈内的样本数量。 `frequency_report` 是我们希望在进度actor中更新当前 `task_ids` 进度的值。

``` python
@ray.remote
def sampling_task(num_samples: int, task_id: int,
                  progress_actor: ray.actor.ActorHandle,
                  frequency_report: int = 1_000_000) -> int:
    num_inside = 0
    for i in range(num_samples):
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)
        if math.hypot(x, y) <= 1:
            num_inside += 1
            
        if (i + 1) % frequency_report == 0:
            progress_actor.report_progress.remote(task_id, i + 1)
    progress_actor.report_progress.remote(task_id, num_samples)
    return num_inside
```

定义一些可调的实验参数：

- `NUM_SAMPLING_TASKS` 您可以根据集群上的cpu进行扩展

- `NUM_SAMPLES_PER_TASK` 您可以增加或减少每个任务的样本数量，以实验它如何影响 $\pi$ 的准确性

- `SAMPLE_REPORT_FREQUENCY` 在采样Ray任务中达到该数字后报告进度

``` python
NUM_SAMPLING_TASKS = os.cpu_count()
NUM_SAMPLES_PER_TASK = 10_000_000
TOTAL_NUM_SAMPLES = NUM_SAMPLING_TASKS * NUM_SAMPLES_PER_TASK
SAMPLE_REPORT_FREQUENCY = 1_000_000

progress_actor = ProgressPIActor.remote(TOTAL_NUM_SAMPLES)

# 并行执行采样任务
time.sleep(1)
results = [
    sampling_task.remote(NUM_SAMPLES_PER_TASK, i, progress_actor, frequency_report=SAMPLE_REPORT_FREQUENCY)
    for i in range(NUM_SAMPLING_TASKS)
]

# 调用进程actor
while True:
    progress = ray.get(progress_actor.get_progress.remote())
    print(f"Progress: {int(progress * 100)}%")
    if progress == 1:
        break
    time.sleep(1)

# 计算pi
total_num_inside = sum(ray.get(results))
pi = (total_num_inside / TOTAL_NUM_SAMPLES) * 4
print(f"Estimated value of pi: {pi}")

ray.shutdown()
```

Ray Actors是有状态的，它们的方法可以被调用来传递消息或改变类的内部状态。Actor被安排在一个专用的Ray节点的工作进程中。因此，所有actor的方法都在特定的工作进程上执行。

我们演示了如何使用actor来跟踪某些Ray任务的进度；在我们的例子中，我们跟踪射线任务的进度近似$ $\pi$ 的值。
