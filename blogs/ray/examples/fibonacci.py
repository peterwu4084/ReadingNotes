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


SEQUENCE_SIZE = 100000

# 本地执行的函数
def generate_fibonacci(sequence_size):
    fibonacci = []
    for i in range(0, sequence_size):
        if i < 2:
            fibonacci.append(i)
            continue
        fibonacci.append(fibonacci[i - 1] + fibonacci[i - 2])
    return len(fibonacci)

# 用于远程Ray task的函数
@ray.remote
def generate_fibonacci_distributed(sequence_size):
    return generate_fibonacci(sequence_size)

# 获取内核的数量
print(os.cpu_count())

# 单个进程中的普通Python
def run_local(sequence_size):
    results = [generate_fibonacci(sequence_size) for _ in range(os.cpu_count())]

# 分布在Ray集群上
def run_remote(sequence_size):
    results = ray.get([generate_fibonacci_distributed.remote(sequence_size) for _ in range(os.cpu_count())])
    return results

start = time.time()
run_local(SEQUENCE_SIZE)
end = time.time()

print(f"Local: {end - start}")

start = time.time()
run_remote(SEQUENCE_SIZE)
end = time.time()
print(f"Remote: {end - start}")