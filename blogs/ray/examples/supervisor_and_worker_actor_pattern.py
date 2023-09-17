import logging
import time
import ray
import random
from random import randint
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pprint import pprint
from operator import itemgetter

import ray
from ray.util.actor_pool import ActorPool

from model_helper_utils import RFRActor
from model_helper_utils import DTActor
from model_helper_utils import XGBoostActor
from model_helper_utils import RANDOM_FOREST_CONFIGS, DECISION_TREE_CONFIGS, XGBOOST_CONFIGS


if ray.is_initialized:
    ray.shutdown()
ray.init(logging_level=logging.ERROR)


class ModelFactory:
    MODEL_TYPES = ['random_forest', 'decision_tree', 'xgboost']
    
    @staticmethod
    def create_model(model_name: str) -> ray.actor.ActorHandle:
        if model_name == 'random_forest':
            configs = RANDOM_FOREST_CONFIGS
            return RFRActor.remote(configs)
        elif model_name == 'decision_tree':
            configs = DECISION_TREE_CONFIGS
            return DTActor.remote(configs)
        elif model_name == 'xgboost':
            configs = XGBOOST_CONFIGS
            return XGBoostActor.remote(configs)
        else:
            raise ValueError(f"{model_name} is not a valid model type")
        

@ray.remote
class Supervisor:
    def __init__(self):
        self.worker_models = [ModelFactory.create_model(name) for name in ModelFactory.MODEL_TYPES]
        
    def work(self):
        results = [worker_model.train_and_evaluate_model.remote() for worker_model in self.worker_models]
        return ray.get(results)

supervisor = Supervisor.remote()
results = supervisor.work.remote()
values = ray.get(results)

while True:
    states = []
    for value in values:
        states.append(value['state'])
    result = all('DONE' == e for e in states)
    if result:
        break

sorted_by_mse = sorted(values, key=itemgetter('mse'))
print(f"\nResults from three training models sorted by MSE ascending order:")
pprint(sorted_by_mse)