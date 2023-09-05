# Preliminaries 
import time
from operator import itemgetter

import pandas as pd
import ray
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Initial Ray runtime
if ray.is_initialized():
    ray.shutdown()

ray.init()

# Prepare dataset
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=201
)

# Put data in the object store
X_train_ref = ray.put(X_train)
X_test_ref = ray.put(X_test)
y_train_ref = ray.put(y_train)
y_test_ref = ray.put(y_test)

# Set number of models to train
NUM_MODELS = 20

# Implement function to train and score model
@ray.remote
def train_and_score_model(
    train_set: pd.DataFrame,
    test_set: pd.DataFrame,
    train_labels: pd.Series,
    test_labels: pd.Series,
    n_estimators: int,
) -> tuple[int, float]:
    start_time = time.time() # measure wall time for single model training

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=201)
    model.fit(train_set, train_labels)
    y_pred = model.predict(test_set)
    score = mean_squared_error(test_labels, y_pred)

    time_delta = time.time() - start_time

    print(f"n_estimators={n_estimators}, mse={score:.4f}, took: {time_delta:.2f} seconds")

    return n_estimators, score

# Implement function that runs sequential model training
def run_sequential(n_models: int) -> list[tuple[int, float]]:
    results_ref = [
        train_and_score_model.remote(
            train_set=X_train,
            test_set=X_test,
            train_labels=y_train,
            test_labels=y_test,
            n_estimators=8 + 4 * j,
        )
        for j in range(n_models)
    ]
    return ray.get(results_ref)

# Run sequential model training
mse_scores = run_sequential(n_models=NUM_MODELS)

# Analyze results
best = min(mse_scores, key=itemgetter(1))
print(f"Best model: mse={best[1]:.4f}, n_estimators={best[0]}")

# Shutdown Ray runtime
ray.shutdown()