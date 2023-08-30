import ray

from ray import serve, tune
from ray.air.config import ScalingConfig
from ray.data.preprocessors import MinMaxScaler
from ray.serve import PredictorDeployment
from ray.serve.http_adapters import pandas_read_json
from ray.train.batch_predictor import BatchPredictor
from ray.train.xgboost import XGBoostTrainer, XGBoostPredictor
from ray.tune.tuner import Tuner, TuneConfig


if ray.is_initialized:
    ray.shutdown()

ray.init()

# Read Parquet file to Ray Dataset.
dataset = ray.data.read_parquet(
    "s3://anyscale-training-data/intro-to-ray-air/nyc_taxt_2021.parquet"
)

# Split data into training and validation subsets.
train_dataset, valid_dataset = dataset.train_test_split(test_size=0.3)

# Split datasets into blocks for parallel preprocessing.
# `num_blocks` should be lower than number of cores in the cluster.
train_dataset = train_dataset.repartition(num_blocks=5)
valid_dataset = valid_dataset.repartition(num_blocks=5)

# Define a preprocessor to normalize the columns by their range.
preprocessor = MinMaxScaler(columns=["trip_distance", "trip_duration"])

trainer = XGBoostTrainer(
    label_column="is_big_tip",
    num_boost_round=50,
    scaling_config=ScalingConfig(
        num_workers=5,
        use_gpu=False,
    ),
    params={
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
        "tree_method": "approx",
    },
    datasets={"train": train_dataset, "valid": valid_dataset},
    preprocessor=preprocessor,
)

# Invoke training. 
# The resulting object grants access to metrics, checkpoints, and errors.
result = trainer.fit()

# Define a search space of hyperparameters.
param_space = {
    "params": {
        "eta": tune.uniform(0.2, 0.4), # learning rate
        "max_depth": tune.randint(1, 6), # default=6; higher value means more complex tree
        "min_child_weight": tune.uniform(0.8, 1.0), # min sum of weights of all data in a child
    }
}

tuner = Tuner(
    trainer,
    param_space=param_space,
    tune_config=TuneConfig(num_samples=3, metric="train-logloss", mode="min"),
)

# Execute tuning on `num_samples` of trials.
# You can query the ResultGrid for metrics, results, and checkpoints by trial.
result_grid = tuner.fit()

best_result = result_grid.get_best_result()
print(f"Best result: \n {best_result} \n")
print(f"Training accuracy: {1 - best_result.metrics['train-error']:.4f}")
print(f"Validation accuracy: {1 - best_result.metrics['valid-error']:.4f}")


test_dataset = ray.data.read_parquet(
    "s3://anyscale-training-data/intro-to-ray-air/nyc_taxi_2022.parquet"
).drop_columns("is_big_tip")

test_dataset = test_dataset.repartition(num_blocks=5)

# Obtain the best checkpointed result from the tuning step.
best_result = result_grid.get_best_result()

# Create a BatchPredictor from the best result and specify a Predictor class.
batch_predictor = BatchPredictor.from_checkpoint(
    checkpoint=best_result.checkpoint, predictor_cls=XGBoostPredictor
)

# Run batch inference.
# Prediction scales across heterogeneous hardware if specified in the ScalingConfig in the Trainer.
predicted_probabilities = batch_predictor.predict(test_dataset)


# Deploy the best checkpoint as a live endpoint using PredictorDeployment.
serve.run(
    PredictorDeployment.options(
        name="XGBoostService", num_replicas=2, route_prefix="/rayair"
    ).bind(XGBoostPredictor, best_result.checkpoint, http_adapter=pandas_read_json)
)

# import requests

# sample_input = test_dataset.take(1)
# sample_input = dict(sample_input[0])


# # Send a request through HTTP.
# output = requests.post("http://localhost:8000/rayair", json=[sample_input]).json()
# print(output)