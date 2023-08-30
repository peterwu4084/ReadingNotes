# [Introduction to Ray AI Runtime (AIR)](https://github.com/ray-project/ray-educational-materials/blob/main/Introductory_modules/Introduction_to_Ray_AI_Runtime.ipynb)

## 学习目标

- 了解组成Ray AIR的高级ML库

  - Data

  - Train

  - Tune

  - Serve

  - RLlib (未深入介绍)

- 使用Ray AIR作为统一的工具包来编写端到端的ML应用程序。

- 扩展这里提供的迷你示例，使用Ray AIR在Python中扩展单个工作负载。

- 确定Ray AIR试图解决的问题和挑战。

## 你会做什么

通过动手练习，您将从示例ML工作流的每个阶段练习关键概念。

| ML工作流阶段 | Ray AIR 关键概念 |
| --- | --- |
| **数据加载和预处理** | 用于加载和转换数据的 `Preprocessor` |
| **模型训练** | 支持ML框架（Keras，Torch等）的 `Trainer` |
| **超参数调优** | 用于调整超参数的 `Tuner` |
| **批量推理** | 用于加载模型并批量推理的 `BatchPredictor`  |
| **模型服务** | 用于在线推理的 `PredictorDeployment` |

## Ray AI Runtime (AIR) 概述

**Ray AI Runtime (AIR)是一个开源的、基于python的、特定领域的库，它为机器学习工程师、数据科学家和研究人员提供了一个可扩展的、统一的机器学习应用程序工具包。**

Ray AIR建立在Ray Core之上，继承了Core提供的所有性能和可扩展性优势，同时为机器学习提供了方便的抽象层。Ray AIR的python优先原生库允许机器学习从业者分发个人工作负载、端到端应用程序，并在统一框架中构建自定义用例。

### 使用Ray AIR的机器学习工作流

Ray AIR封装了5个原生Ray库，它们可以扩展ML工作流的特定阶段。此外，Ray AIR将一个不断增长的生态系统与流行的机器学习框架集成在一起，以创建一个通用的开发接口。

Ray AIR支持端到端机器学习开发，并提供多种选项，可与MLOps生态系统中的其他工具和库集成。

![Ray AIR](./assets/ray_air.png)

1. Ray Data: 跨训练、调优和预测的可伸缩的、与框架无关的数据加载和转换。

2. Ray Train: 具有容错的分布式多节点和多核模型训练，集成了流行的机器学习训练库。

3. Ray Tune: 拓展超参数调优以优化模型性能。

4. Ray Serve: 为在线推理部署一个模型或模型集合。

5. Ray RLlib: 与其他Ray AIR库集成扩展强化学习工作负载。

## 使用Ray AI Runtime的端到端工作流程

为了说明Ray AIR的功能，您将实现一个端到端机器学习应用程序，该应用程序使用纽约市出租车数据预测小费。每一节将介绍相关的Ray AIR库或组件，并通过代码示例演示其功能。

对于这个分类任务，您将应用一个简单的XGBoost模型到2021年6月的[纽约市出租车和豪华轿车委员会的旅行记录数据](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)。这个数据集包含超过200万个黄色出租车的样本，目标是预测一次旅行的小费是否会超过20%。

该数据的特征如下：

- **passenger_count**: 浮点数(整数)，代表乘客人数。

- **trip_distance**: 浮点数，表示以英里为单位的行程距离。

- **fare_amount**: 浮动数，代表总价，包括税、小费、费用等。

- **trip_duration**: 整数，表示经过的秒数。

- **hour**: 整数，表示行程开始的小时，范围为0到23。

- **day_of_week**: 整数，表示星期几，范围为0到6。

- **is_big_tip**: 布尔值，表示小费是否超过20%。

## Ray Data

首先，您需要加载出租车数据，并将原始输入转换为经过清理的特征，以便传递给XGBoost模型。

Ray AIR封装Ray Data，在训练、调优和推理期间提供分布式数据摄取和转换。

### Ray Datasets介绍

在PyArrow的支持下，Ray Datasets并行化了数据的加载和转换，并提供了一种跨射线库和应用程序传递数据引用的标准方式。数据集不打算取代更通用的数据处理系统。相反，它充当了从ETL管道输出到Ray中的分布式应用程序和库的最后一步。其关键特征如下：

- 灵活：数据集与各种文件格式、数据源和分布式框架兼容。它们可以与像Dask这样的库无缝地工作，并且可以在Ray task和actor之间传递而无需复制数据。

- ML工作负载的性能：数据集提供了重要的功能，如加速器支持、流水线和全局随机洗牌，可以加速机器学习训练和推理工作。它们还支持基本的分布式数据转换，如映射、筛选、排序、分组和重分配。

- 一致的预处理：`Preprocessor` 捕获并存储用于将输入转换为特征的转换。它在训练、调优、批处理预测期间应用，并用于保持整个管道的预处理一致。

- 基于Ray Core构建：数据集继承了Ray Core的数百个节点的可扩展性、高效的内存使用、对象溢出和故障处理。因为数据集只是对象引用的列表，所以它们可以在任务和参与者之间传递，而不需要复制数据，这对于使数据密集型应用程序和库具有可伸缩性至关重要。

记住这个通用结构后，接下来您将看到如何将其应用于提示预测任务。

### 初始化Ray

``` python
import ray

if ray.is_initialized:
    ray.shutdown()

ray.init()
```

启动一个Ray集群，以便Ray可以利用所有可用的内核作为工作线程。

- 检查 `ray.is_initialized` 确保从一个新的集群开始

- 使用 `Ray .init()` 初始化Ray上下文

### 创建Ray Datasets

``` python
# Read Parquet file to Ray Dataset
dataset = ray.data.read_parquet(
    "s3://anyscale-training-data/intro-to-ray-air/nyc_taxt_2021.parquet"
)

# Split data into training and validation subsets
train_dataset, valid_dataset = dataset.train_test_split(test_size=0.3)

# Split datasets into blocks for parallel preprocessing
# `num_blocks` should be lower than number of cores in the cluster
train_dataset = train_dataset.repartition(num_blocks=5)
valid_dataset = valid_dataset.repartition(num_blocks=5)
```

### 数据集预处理

为了将原始数据转换为特征，您将定义一个预处理器 `Preprocesser` 。Ray AIR的 `Preprocessor` 持续的捕获和转换数据:

- 在训练期间

  `Preprocessor` 会被传递给 `Trainer`，拟合并转换输入数据。

- 在调优期间

  每一个 `Trial`会创建自己的 `Preprocessor` 副本。拟合和转换的逻辑在每一个 `Trial`中发生一次。

- 在检查点期间

  如果 `Preprocessor` 被传递到 `Trainer` 中，它会被保存到 `Checkpoint` 中。

- 在推理期间

  如果 `Checkpoint` 包含一个 `Preprocessor`, 那么它将在推理前对批输入调用 `transform_batch`。

``` python
from ray.data.preprocessors import MinMaxScaler

# 定义一个预处理器，按列的范围对列进行归一化
preprocessor = MinMaxScaler(columns=['trip_distance', 'trip_duration'])
```

### 小结

#### 关键概念

- `Dataset`

  在Ray AIR中加载和转换数据的标准方式。在AIR中，`Dataset` 被广泛用于数据加载和转换。它们是连接ETL管道输出到Ray中的分布式应用程序和库的最后一步。

- `Preprocessor`

  预处理器是将输入数据转换为特征的原语。它们对数据集进行操作，使其可扩展并与各种数据源和数据框架库兼容。

  预处理器在管道的各个阶段持续存在:

  - 在训练期间拟合和转换数据

  - 存在于超参数调优的每个 `Trial`

  - 存在于检查点中

  - 用于推理的输入批次

AIR附带了一系列内置预处理器，您还可以使用简单的模板定义自己的预处理器。

## Ray Train

### Ray Train介绍

#### 训练中的常见挑战

ML从业者在训练模型中往往会遇到一些常见的问题，这些问题促使他们考虑分布式解决方案:

1. 训练时间太长，不切合实际。

2. 数据太大，一台机器装不下。

3. 按顺序训练多个模型并不能有效地利用资源。

4. 模型本身太大，一台机器装不下。

Ray Train通过分布式多节点训练来提高性能，从而解决了这些问题。

#### 与Ray生态系统集成

Ray Train的 `Trainer` 与Ray生态系统的其余部分集成得很好:

- Ray Data

  - 通过 `Dataset` 和 `Preprocessor` 实现可扩展的数据加载和预处理。

- Ray Tune

  - 与 `Tuner` 实现分布式超参数调优。

- Ray AIR Predictor

  - 在推理过程中作为模型训练的检查点。

- 流行的ML训练框架

  - PyTorch

  - Tensorflow

  - Horovod

  - XGBoost

  - HuggingFace Transformers

  - Scikit-Learn

  - 其他

#### 有用的特性

- 提前停止的回调

- 检查点

- 集成实验跟踪工具，如Tensorboard、Weights & Biases和MLFlow

- 模型的导出机制

在下一节，我们将定义并拟合一个XGBoost训练器来拟合纽约市出租车数据。

### 定义AIR `Trainer`

Ray AIR提供了各种内置训练器(PyTorch, Tensorflow, HuggingFace等)。在下面的示例中，您将使用Ray `XGBoostTrainer`，它提供了对XGBoost模型的支持。

``` python
from ray.air.config import ScalingConfig
from ray.train.xgboost import XGBoostTrainer

trainer = XGBoostTrainer(
  label_column='is_big_tip',
  num_boost_round=50,
  scaling_config=ScalingConfig(
    num_workers=5,
    use_gpu=False
  ),
  params={
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'error'],
    'tree_method': 'approx'
  },
  datasets={'train': train_dataset, 'valid': valid_dataset},
  preprocessor=preprocessor,
)
```

要构造一个 `Trainer`，您需要提供三个基本组件:

- `ScalingConfig`，指定多少并行训练工作器和什么类型的资源(cpu / gpu);支持跨异构硬件的无缝扩展。

- 训练集和验证集的字典。

- 用于转换 `Dataset` 的 `Preprocessor`。

您还可以选择添加 `resume_from_checkpoint`，它允许您在运行中断时从保存的检查点继续训练。

### 拟合训练器

``` python
# 调用训练
# 结果对象可以访问训练度量、检查点和错误。
result = trainer.fit()
```

### 小结

#### 关键概念

- `Checkpoint`

  定期存储模型的完整状态，使部分训练好的模型可用，并可用于从中间点恢复训练，而不是从头开始;还允许保存最佳模型，以便以后进行批推理。

- `Trainer`

  `Trainer` 是第三训练框架(如XGBoost、Pytorch和Tensorflow)的包装类。它们的构建是为了帮助与Ray Actor(用于分发)、Dataset和Tune集成。

## Ray Tune

现在您已经训练了一个基线XGBoost模型，您可以尝试通过运行超参数调优实验来提高性能。

### Ray Tune介绍

超参数优化(HPO)是为机器学习模型选择最优超参数的过程。与模型学习的权重相反，超参数是您设置以影响训练的参数。

设置和执行HPO在计算资源和运行时方面可能非常昂贵，其中包括以下几个复杂性:

- 巨大的搜索空间：您的模型可以有几个超参数，每个都有不同的数据类型、范围和可能的相关性。从高维空间中采样是很困难的。

- 搜索算法：策略性地选择超参数需要测试复杂的搜索算法以获得良好的结果。

- 漫长的运行时间：即使分布调优，训练复杂模型本身也需要很长时间才能完成每次运行，因此最好在管道中的每个阶段都有效率。

- 资源分配：在每次试验期间，必须有足够的计算资源可用，以免由于调度不匹配而减慢搜索速度。

- 用户体验：面向开发人员的观察工具，如早期停止错误的运行、保存中间结果、从检查点重新启动或暂停/恢复运行，使HPO变得更容易。

Ray Tune是一个分布式HPO库，它解决了上述所有问题，为运行试验提供了一个简化的接口，并与HyperOpt和Optuna等流行框架集成在一起。

在下一节中，您将了解如何应用这些步骤来优化上一节中创建的基线XGBoost模型。

### 使用AIR `Tuner` 进行超参数搜索

顺便说一句，Ray Tune将为超参数调优提供一个默认的检查点系统。对于特别大的模型，最好设置一个定义检查点策略的 `CheckpointConfig`。特别是，您可以切换 `num_to_keep` 以避免将任何徒劳的试验保存到磁盘上。

``` python
from ray import tune
from ray.tune.tuner import Tuner, TuneConfig

# 定义一个超参数搜索空间。
param_space = {
  "params": {
    "eta": tune.uniform(0.2, 0.4), # 学习率
    "max_depth": tune.randint(1, 6), # 默认6，值越高意味着树越复杂
    "min_child_weight": tune.qrandint(0.8, 1.0), # 子节点中所有数据的最小权重之和
  }
}

tuner = Tuner(
  trainer,
  param_space=param_space,
  tune_config=TuneConfig(
    num_samples=3,
    metric='train-logloss',
    mode='min'
  )
)
```

要设置AIR `Tuner`，您必须指定:

- `Trainer`: 训练器，支持内置在每个Trainer的ScalingConfig中的异构硬件

- `param_space`: 希望调优的一组超参数。

- `TuneConfig`: 设置实验的数量、指标以及是最小化还是最大化。

- `search_algorithm`: 优化参数搜索(可选)。

- `scheduler`: 早停搜索并加速实验(可选)。

### 执行超参数搜索并分析结果

``` python
result_grid = tuner.fit()
```

### 小结

#### 关键概念

- `Tuner`

  提供与AIR `Trainer` 一起工作以执行分布式超参数调优的接口。定义一组希望在搜索空间中调优的超参数，指定搜索算法，然后 `Tuner` 在 `ResultGrid` 中返回结果，该 `ResultGrid` 包含每个试验的指标、结果和检查点。

## Ray AIR Predictors

`Ray AIR Predictors` 从训练或调优期间生成的检查点加载模型，以执行分布式推理。
`BatchPredictor` 是一个用于大规模批处理推理的实用程序，它包含几个组件:

1. `Checkpoint`: 从训练或调优中保存的模型。

2. `Preprocessor`: 先前定义的用于转换输入数据的预处理器，可以重新应用于预测(可选)。

3. `Predictor`: 从检查点加载模型以执行推理。

### 使用AIR `BatchPredictor` 进行批预测

之前，您已经在2021年6月的数据上训练和调整了XGBoost模型。现在，您将从调优中获得最佳检查点，并对2022年6月的出租车小费数据执行离线或批量推理。

``` python
from ray.train.batch_predictor import BatchPredictor
from ray.train.xgboost import XGBoostPredictor

test_dataset = ray.data.read_parquet(
  "s3://anyscale-training-data/intro-to-ray-air/nyc_taxi_2022.parquet"
).drop_columns("is_big_tip")

test_dataset = test_dataset.repartition(num_blocks=5)
```

### 从HPO的最佳试验中创建 `BatchPredictor`

``` python
# 从调优步骤中获得最佳检查点结果。
best_result = result_grid.get_best_result()

# 从最佳结果创建BatchPredictor并指定一个Predictor类。
batch_predictor = BatchPredictor.from_checkpoint(
  checkpoint=best_result.checkpoint, predictor=XGBoostPredictor
)

# 执行推理。
# 如果在训练器的ScalingConfig中指定，预测可以跨异构硬件扩展。
predicted_probabilitites = batch_predictor.predict(
  test_dataset
)
```

### 小结

#### 关键概念

- `BatchPredictor`

  从检查点加载最佳模型，对大规模推理或在线推理执行批推理。

## Ray Serve

最后，您可能希望将这个出租车小费预测应用程序提供给最终用户，希望它具有较低的延迟，从而最大限度地为工作中的司机提供帮助。这带来了挑战，因为机器学习模型是计算密集型的，理想情况下，这个模型不会孤立地服务，而是与业务逻辑甚至其他ML模型相邻。

### Ray Serve介绍

Ray Serve是一个可扩展的计算层，用于服务机器学习模型，它支持服务单个模型或创建复合模型管道，您可以在其中独立部署、更新和扩展单个组件。

Serve没有绑定到特定的机器学习库，而是将模型视为普通的Python代码。

此外，它允许您灵活地将普通Python业务逻辑与机器学习模型结合起来。这使得完全端到端构建在线推理服务成为可能:

- 验证用户输入。

- 查询数据库。

- 跨多个ML模型可扩展地执行推理。

- 在处理单个推理请求的过程中组合、过滤和验证输出。

### 使用 `PredictorDeployment` 进行在线推理

#### 从检查点部署XGBoost模型

``` python
from ray import serve
from ray.serve import PredictorDeployment
from ray.serve.http_adapters import pandas_read_json

# 使用PredictorDeployment将最佳检查点部署为实时端点。
serve.run(
  PredictorDeployment.options(
    name='XGBoostService', num_replicas=2, route_prefix='/rayair'
  ).bind(XGBoostPredictor, best_result.checkpoint, http_adapter=pandas_read_json)
)
```

#### 发送一些测试数据

``` python
import requests

sample_input = test_dataset.take(1)
sample_input = dict(sample_input[0])

# 发送http请求。
output = requests.post(
  'http://localhost:8000/rayair', json=[sample_input]
).json()
print(output)
```

### 关闭Ray

``` python
# 断开工作器的连接，并终止由ray.init()启动的进程。
ray.shutdown()
```

### 小结

#### 关键概念

- `Deployments`

  一组被管理的Ray actor，它们可以被一起处理，并在它们之间平衡负载请求。

## 总结

现在，您已经创建了一个Ray Dataset，预处理了一些特征，用XGBoost构建了一个模型，在超参数空间中搜索最佳配置，从检查点加载最佳模型以执行批处理推理，并将该模型用于在线推理。

通过这个端到端示例，您探索了如何使用Ray AIR来分发整个ML管道。

### 关键概念

- `Dataset`

  在Ray AIR中加载和转换数据的标准方式。在AIR中，数据集被广泛用于数据加载和转换。它们是连接ETL管道输出到Ray中的分布式应用程序和库的最后一步。

- `Preprocessor`

  预处理器是将输入数据转换为特征的基础元素。它们对数据集进行操作，使其可扩展并与各种数据源和数据框架库兼容。

- `Checkpoint`

  定期存储模型的完整状态，使部分训练好的模型可用，并可用于从中间点恢复训练，而不是从头开始;还允许保存最佳模型，以便以后进行批推理。

- `Trainer`

  Trainer是第三方培训框架(如XGBoost、Pytorch和Tensorflow)的包装类。它们的构建是为了帮助与Ray actor(用于分发)、Ray Dataset和Ray Tune集成。

- `Tuner`

  提供与AIR Trainer一起工作以执行分布式超参数调优的接口。定义一组希望在搜索空间中调优的超参数，指定搜索算法，然后Tuner在ResultGrid中返回结果，该ResultGrid包含每个试验的指标、结果和检查点。

- `BatchPredictor`

  从检查点加载最佳模型，对大规模推理或在线推理执行批推理。

- `Deployments`

  一组被管理的Ray actor，它们可以被一起处理，并在它们之间平衡负载请求。