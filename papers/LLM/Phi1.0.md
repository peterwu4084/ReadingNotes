# Textbooks Are All You Need

## Contribution

一个新的代码大语言模型“phi-1”：

1. 参数量为 1.3B；

2. 训练使用“教科书质量”的网络数据（6B tokens）和 GPT3.5 生成的数据（1B tokens）；

3. HumanEval 和 MBPP 上分别取得 50.6% 和 55.5% 的 pass@1 准确率

## Training details and the importance of high-quality data

现有网络上的代码数据的缺陷：

- 样本依赖于其他外部的模块或文件，使得代码难以理解；

- 样本未包含有意义的逻辑计算，如设置参数、常量的代码；

- 样本中包含复杂的逻辑，并且文档不足；

- 各种主题的代码分布不均衡；

通过使用高质量的数据，可以在更小的模型和计算量下取得更优的结果。phi-1.0 的训练集包括：

1. 一个基于语言模型过滤的代码数据集，包含 Stack、 StackOverflow的 6B token 数据；

2. 一个GPT3.5生成的教科书数据集，包含小于 1B token 数据；

3. 一个小的生成的 Python 练习数据集，包含约 180M token 数据；

一、二数据集用于模型的预训练，获得基础模型 phi-1-base，三数据集用于微调模型，获得 phi-1 模型。

### 基于语言模型过滤代码数据集

通过GPT4过滤 Python 代码数据，使用的提示词为 “determine its educational value for a student whose goal is to learn basic coding concepts”。

GPT4标注的数据被用于训练一个随机森林分类器，预测一个样本的质量。

使用未过滤的数据训练350M的模型，训练96k步，在HumanEval上的表现基本饱和，为12.19%；而使用过滤数据，仅36k步就可以达到17.68%的表现。

### 合成教科书数据

合成数据的难点之一在于如何确保数据的多样性和不重复。而大模型生成的回复倾向于生成最常见的回复。为了使模型生成的数据更丰富，作者从预设的词典中随机选取词汇，并要求模型生成的回复包含这些词汇。同时对回复的主题进行了限制。

### 合成 Python 练习数据

该数据同样通过 GPT3.5 生成，其目的是对齐模型以基于自然语言指令完成代码补全的任务。生成的数据进行了清洗，确保没有和评测数据 HumanEval 重复。

