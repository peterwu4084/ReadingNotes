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