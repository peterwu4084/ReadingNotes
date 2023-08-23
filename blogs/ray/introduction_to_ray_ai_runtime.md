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

在PyArrow的支持下，Ray Datasets并行化了数据的加载和转换，并提供了一种跨射线库和应用程序传递数据引用的标准方式。数据集不打算取代更通用的数据处理系统。相反，它充当了从ETL管道输出到Ray中的分布式应用程序和库的最后桥梁。其关键特征如下：

- 灵活：数据集与各种文件格式、数据源和分布式框架兼容。它们可以与像Dask这样的库无缝地工作，并且可以在Ray task和actor之间传递而无需复制数据。

- ML工作负载的性能：数据集提供了重要的功能，如加速器支持、流水线和全局随机洗牌，可以加速机器学习训练和推理工作。它们还支持基本的分布式数据转换，如映射、筛选、排序、分组和重分配。

- 一致的预处理：`Preprocessor` 捕获并存储用于将输入转换为特征的转换。它在训练、调优、批处理预测期间应用，并用于保持整个管道的预处理一致。

- 基于Ray Core构建：数据集继承了Ray Core的数百个节点的可扩展性、高效的内存使用、对象溢出和故障处理。因为数据集只是对象引用的列表，所以它们可以在任务和参与者之间传递，而不需要复制数据，这对于使数据密集型应用程序和库具有可伸缩性至关重要。

记住这个通用结构后，接下来您将看到如何将其应用于提示预测任务。

#### Start Ray runtime

``` python
import ray

if ray.is_initialied:
    ray.shutdown()

ray.init()
```

启动一个Ray集群，以便Ray可以利用所有可用的内核作为工作线程。

- 检查 `ray.is_initialized` 确保从一个新的集群开始

- 使用 `Ray .init()` 初始化Ray上下文

#### 创建Ray Datasets

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

#### 