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