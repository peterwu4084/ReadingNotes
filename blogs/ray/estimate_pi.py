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


NUM_SAMPLING_TASKS = os.cpu_count()
NUM_SAMPLES_PER_TASK = 10_000_000
TOTAL_NUM_SAMPLES = NUM_SAMPLING_TASKS * NUM_SAMPLES_PER_TASK

def sampling_task(num_samples: int, task_id: int, verbose=True) -> int:
    num_inside = 0
    for i in range(num_samples):
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)
        if math.hypot(x, y) <= 1:
            num_inside += 1
    if verbose:
        print(f"Task id: {task_id} | Samples in the circle: {num_inside}")
    return num_inside


def run_serial(sample_size) -> List[int]:
    results = [sampling_task(sample_size, i) for i in range(NUM_SAMPLING_TASKS)]
    return results


@ray.remote
def sample_task_distributed(sample_size, i) -> object:
    return sampling_task(sample_size, i)

def run_distributed(sample_size) -> List[int]:
    # 在一个推导式列表中启动Ray远程任务，每个任务立即返回一个未来的ObjectRef
    # 使用ray.get获取计算值；这将阻塞直到ObjectRef被解析或它的值被具体化。
    results = ray.get([sample_task_distributed.remote(sample_size, i) for i in range(NUM_SAMPLING_TASKS)])
    return results


def calculate_pi(results: List[int]) -> float:
    return 4 * sum(results) / TOTAL_NUM_SAMPLES

# 串行计算π
start = time.time()
results = run_serial(NUM_SAMPLES_PER_TASK)
pi = calculate_pi(results)
end = time.time()
print(f"Estimated value of pi is: {pi:5f}")
print(f"Serial execution time: {end - start:5f}")

# 分布式计算π
start = time.time()
results = run_distributed(NUM_SAMPLES_PER_TASK)
pi = calculate_pi(results)
end = time.time()
print(f"Estimated value of pi is: {pi:5f}")
print(f"Distributed execution time: {end - start:5f}")