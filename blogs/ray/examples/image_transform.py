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
import tasks_helper_utils as t_utils

if ray.is_initialized:
    ray.shutdown()
ray.init(logging_level=logging.ERROR)

DATA_DIR = Path(os.getcwd() + "/task_images")
BATCHES = [10, 20, 30, 40, 50]
SERIAL_BATCH_TIMES = []
DISTRIBUTED_BATCH_TIMES = []

# 定义一个Ray task来转换，增强和执行一些计算密集型的任务
@ray.remote
def augment_image_distributed(image_ref: object, fetch_image) -> List[object]:
    return t_utils.transform_image(image_ref, fetch_image)

# 定义一个函数，在单个节点、单个核心上串行地运行这些转换任务
def run_serially(img_list_refs: List) -> List[Tuple[int, float]]:
    transform_results = [t_utils.transform_image(image_ref, fetch_image=True) for image_ref in tqdm.tqdm(img_list_refs)]
    return transform_results

# 定义函数以分布式地运行这些转换任务
def run_distributed(img_list_refs: List[object]) -> List[Tuple[int, float]]:
    return ray.get([augment_image_distributed.remote(image_ref, False) for image_ref in tqdm.tqdm(img_list_refs)])

# 检查文件夹是否存在。如果是，忽略下载。
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
    print(f"downloading images...")
    for url in tqdm.tqdm(t_utils.URLS):
        t_utils.download_images(url, DATA_DIR)

# 获取整个图像列表      
image_list = list(DATA_DIR.glob("*.jpg"))

# 将所有图像放入对象存储中。因为Ray任务可能是分布
# 在不同的机器上，工作线程上可能没有DATA_DIR。然而,
# 将它们放入Ray分布式对象器中，可以在Ray worker上
# 访问任何调度远程任务
image_list_refs = [t_utils.insert_into_object_store(image) for image in image_list]

for idx in BATCHES:
    # 使用索引获取N个指向图像的url
    image_batch_list_refs = image_list_refs[:idx]
    print(f"\nRunning {len(image_batch_list_refs)} tasks serially ...")
    
    # 串行运行
    start = time.perf_counter()
    serial_results = run_serially(image_batch_list_refs)
    end = time.perf_counter()
    elapsed = end - start
    
    # 以元组的形式跟踪批处理、执行时间
    SERIAL_BATCH_TIMES.append((idx, round(elapsed, 2)))
    print(f"Serial transformation/computations of {len(image_batch_list_refs)} images: {elapsed:.2f}) sec")

# 迭代批次，为处理中的每个图像启动Ray task
for idx in BATCHES:
    image_batch_list_refs = image_list_refs[:idx]
    print(f"\nRunning {len(image_batch_list_refs)} tasks distributed...")

    # 依次运行每一个
    start = time.perf_counter()
    distributed_results = run_distributed(image_batch_list_refs)
    end = time.perf_counter()
    elapsed = end - start

    # 以元组的形式跟踪批处理和执行时间
    DISTRIBUTED_BATCH_TIMES.append((idx, round(elapsed, 2)))
    print(f"Distributed transformation/computations of {len(image_batch_list_refs)} images: {elapsed:.2f}) sec")

# 打印每一项的时间，并绘制它们以便比较
print(f"Serial times & batches     : {SERIAL_BATCH_TIMES}")
print(f"Distributed times & batches: {DISTRIBUTED_BATCH_TIMES}")
t_utils.plot_times(BATCHES, SERIAL_BATCH_TIMES, DISTRIBUTED_BATCH_TIMES)

ray.shutdown()