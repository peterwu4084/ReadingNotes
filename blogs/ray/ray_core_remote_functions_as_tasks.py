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