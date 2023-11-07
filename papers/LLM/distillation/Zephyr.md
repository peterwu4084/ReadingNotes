# Zephyr: Direct Distillation of LM Alignment

## Motivation

蒸馏是提升模型在下游任务表现的有效方法。蒸馏监督微调（dSFT）可以提升下游任务的准确率，但是用户发现微调的模型存在与人类意图不一致的问题，导致模型无法正确回答自然的提示词。

而PPO等强化学习对齐人类意图的方法，需要大量的人类标注和采样。

## Contribution

提出了一种基于蒸馏的方法解决对齐问题。主要步骤包括：

1. 使用dSFT训练模型；
2. 使用AI feedback数据，dDPO (distilled direct preference optimization)微调模型；

微调的模型在MT-Bench和AlpacaEval上均表现良好：优于开源7b模型，和LLAMA2 70b模型相媲美。