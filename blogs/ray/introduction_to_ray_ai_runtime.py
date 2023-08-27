import ray

if ray.is_initialized:
    ray.shutdown()

ray.init()

dataset = ray.data.read_parquet(
    "s3://anyscale-training-data/intro-to-ray-air/nyc_taxt_2021.parquet"
)

import pdb; pdb.set_trace()
print()