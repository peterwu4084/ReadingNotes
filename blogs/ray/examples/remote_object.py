import logging
import random

from typing import Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import ray


def create_rand_tensor(size: Tuple[int, int]) -> torch.tensor:
    return torch.randn(size=(size), dtype=torch.float)

@ray.remote
def transform_rand_tensor(tensor: torch.tensor) -> torch.tensor:
    return torch.mul(tensor, random.randint(2, 10))

torch.manual_seed(42)

tensor_list_obj_ref = [ray.put(create_rand_tensor(((i+1)*25, 50))) for i in range(100)]
print(tensor_list_obj_ref[:2], len(tensor_list_obj_ref))

# 因为我们得到了一个ObjectRefIDs列表，从中索引到张量的第一个值
val = ray.get(tensor_list_obj_ref[0])
print(val.size(), val)

# 或者，您可以获取多个对象引用的所有值。
results = ray.get(tensor_list_obj_ref)
print(results[:1], results[:1][0].size())

transformed_object_list = [transform_rand_tensor.remote(t_obj_ref) for t_obj_ref in tensor_list_obj_ref]
print(transformed_object_list[:2])

# 获取所有变换后的张量
transformed_tensor_values = ray.get(transformed_object_list)
print(transformed_tensor_values[:2])

ray.shutdown()