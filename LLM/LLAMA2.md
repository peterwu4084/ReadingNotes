# LLAMA 2: Open Foundation and Fine-tuned Chat Models

## Abstract

LLAMA 2为Meta发布的大语言模型集合： 模型参数量包含7b、13b、34b (未公开)、70b，同时每个大小的模型包含两个版本，预训练版本和对话微调版本。

基于人类评测，LLAMA 2在帮助性和安全性上优于现有的开源对话模型。

论文详细介绍了LLAMA 2是如何诞生的，包括数据集的构建、模型训练微调以及如何提升安全性等方面。

## Introduction

大语言模型(LLM)的发展和应用具有极大前景。然而LLM的训练需要消耗极大的资源(计算资源、数据处理等)，以及部分LLM的闭源，阻碍了对LLM的研究。Meta开源LLAMA 2，在帮助性和安全性上优于现有的开源模型，甚至比肩一些闭源模型，希望可以推动LLM的发展。其主要贡献包括：

- 开源LLAMA 2系列模型，包含预训练和对话微调两个版本，每个版本分别有7b、13b、34b、70b不同大小的模型。LLAMA 2和上一代LLAMA 1的对比见Table 1。

- 详细介绍了LLAMA 2的数据处理、预训练、微调的方法，大体流程见Figure 4。

- 重点关注模型的安全性，从数据和微调等途径提升模型的安全性。

![tab1.png](./assets/llama2_tab1.png)

![fig4.png](./assets/llama2_fig4.png)

## Pretraining

LLAMA 2基于LLAMA 1的预训练方法，通过以下途径提升预训练效果：

1. 更鲁棒的数据清洗，更新了预训练语料库的组成，训练使用的token数量增加40%；

2. 加倍上下文长度；

3. 使用grouped-query attention (GQA)提升推理速度；

### Ptraining Data

预训练数据混合了不同的开源数据，但是不包含Meta的产品或服务产生的数据。同时，去除私人信息网站的数据。此外，对真实来源的数据进行上采样，以增加模型知识并减弱“幻觉”。最终预训练的token数量达到2T。

### Model Architecture

LLAMA 2基于[标准的transformer结构](https://arxiv.org/abs/1706.03762)，并采用如下的修改：

1. 归一化层使用RMSNorm和pre-norm；

2. 激活层使用SwiGLU激活函数；

3. 位置编码使用rotary positional embeddings (RoPE)；

4. 上下文长度扩大到LLAMA 1的两倍；

5. 34、70b使用grouped-query attention (GQA)；

6. tokenizer与LLAMA 1相同：通过bytepair encoding (BPE)算法分词，所有数字被分割成单个数字，使用字节分解未知的utf-8字符。分词后的词表长度为32000。

### HyperParameters

1. 优化器使用AdamW ($\beta_1 = 0.9, \beta_2 = 0.95, eps = 1e-5$)；

2. 学习率设置：Cosine lr schedule, warmup 2000 steps, lr下降到峰值(见Table 1)的10%；

3. 权重衰减：0.1；

4. 梯度裁剪：1.0；

![llama2_fig_train_loss.png](./assets/llama2_fig5.png)

### Pretrained Model Evaluation

Meta从代码、常识推理、对世界的认识、阅读理解、数学五个方面对比了LLAMA 2和其他开源模型，包括LLAMA 1、MosaicML Pretrained Transformer (MPT)和Falcon。测试benchmark基于开源数据构建，同时也在一些著名的综合benchmark上进行了对比。测试数据包括：

- 代码：测试在HumanEval和MBPP上的平均pass@1指标；

- 常识推理：测试在PIQA、SIQA、HellaSwag、WinoGrande、ARC easy and challenge、OpenBookQA、CommonsenseQA上的平均表现。其中CommonsenseQA测试使用7-shot测试，其他使用0-shot测试；

- 对世界的认识：基于5-shot测试在NaturalQuestions和TriviaQA上的平均表现；

- 阅读理解：测试在SQuAD、QuAC、BoolQ上的0-shot的平均表现；

- 数学：测试在GSM8K (8-shot)和MATH (4-shot)上的平均top 1；

- 其他综合benchmark：包括MMLU (5-shot)、Big Bench Hard (BBH) (3-shot)、AGI Eval (only English tasks) (3-5shot)；

测试结果如Table 3所示，LLAMA 2在相近参数量大小下，在所有能力和综合测试集上均优于其他开源模型，除了在代码能力上弱于MPT。

![image.png](./assets/llama2_tab3.png)

此外，Meta还和部分闭源模型进行了对比，可以看出LLAMA 2在除了代码之外的能力，其他能力的表现与GPT-3.5接近。在所有能力上优于PaLM，不及PaLM-2-L和GPT-4，但LLAMA 2的参数量最大只有70b。

![image.png](./assets/llama2_tab4.png)

## Fine-tuning

LLAMA 2-Chat是在pretrained的基础上进行了**指令监督微调**和**人类反馈强化学习**获得的。人类反馈强化学习包括奖励模型的初始化和迭代和强化学习训练模型。同时提出和使用了一种新的技术，Ghost Attention (GAtt)，用于提升多轮对话的效果。

### Supervised Fine-Tuning

有监督微调的步骤：

1. 第一步，使用开源的指令微调数据微调模型；

2. 第二步，标注并收集数千高质量SFT数据（例见Table 5）进一步微调模型；

使用高质量数据微调效果优于使用数百万的第三方数据（多样性或质量低）。这说明少量的高质量指令微调数据对于微调是足够的，微调数据质量比数量更重要。Meta最终发现数万的高质量指令数据足够实现高质量的结果，并最终收集了27540条数据。

![image.png](./assets/llama2_tab5.png)

SFT参数和细节如下：

- 使用cosine lr schedule，初始学习率设置为2e-5, 权重衰减为0.1, batch size为64, 序列长度为4096 tokens，微调2个epoch；

- 每个样本包含一个prompt和一个answer，为了确保模型输入被合适地填充，所有的prompt和answer被连接起来。一个特殊的token用于分割不同样本；

- 使用自回归目标函数，仅在answer上计算损失函数；

### RLHF

人类反馈强化学习是将人类偏好注入模型的模型训练方法。LLAMA 2的RLHF的步骤包括：收集人类偏好数据，迭代训练训练奖励模型和语言模型优化。

#### Human Preference Data Collection

人类偏好数据的收集步骤如下：

1. 首先标注人员撰写一个prompt；

2. 随机选择两个不同的模型，变化温度超参数生成response；

3. 标注人员选取其中一个更好的response，并选取的response和另一个response的差别：significantly better, better, slightly better, negligibly better (or unsure)；

Meta同时关注模型的帮助性和安全性，分别收集了两个人类偏好数据：帮助偏好数据和安全偏好数据。此外，每一个样本标注了额外的安全标签，标签包含三个值：选择的response是安全的且未选择的标签是不安全的；两个response都是安全的；两个response都是不安全的。选择的response是不安全的，未选择的response是不安全的样本被舍弃。

随着人类偏好数据收集，奖励模型会不断提升，并进一步使用奖励模型迭代LLAMA 2-chat。同时LLAMA 2-chat的提升，会改变模型生成数据的分布。因此收集偏好数据需要使用最新迭代的LLAMA 2-chat模型，使得奖励模型能够队最新的对话模型预测准确的奖励值。

Table 6展示了Meta收集的人类偏好数据的统计量，并和多个开源偏好数据进行了对比。Meta的数据量到达了一百万，同时具有更多的对话轮数和较大的平均token数。

![image.png](./assets/llama2_tab6.png)

#### Reward Modeling

奖励模型以prompt和response作为输入，输出一个标量，表示语言模型的输出质量（帮助性或安全性）利用奖励模型输出的质量分数，可以优化对话模型对齐人类偏好，提升模型的帮助性和安全性。

帮助性和安全性之间有时存在相互权衡（如“如何制造炸弹”），一个奖励模型难以在两方面都表现好，因此Meta训练了两个单独的奖励模型分别用于预测帮助性和安全性。

模型使用对话模型初始化权重，原本的预测下一个token的分类头被替换成输出标量的回归头。

- 目标函数

    训练奖励模型，之前收集的偏好数据被转换成一个二元排序标签的格式，并且选择的response比拒绝的response具有更高的分数。使用二元排序损失作为目标函数：$L_{ranking}=-log(\sigma(r_\theta(x, y_c)-r_\theta(x,y_r))$

    $\theta$为奖励模型的参数，$x$和$y$是prompt和对应response，$r_\theta(x, y)$ 是奖励模型预测的分数。$y_c$是选择的response， $y_r$拒绝的response。

    为了更好的建模帮助性和安全性，Meta进一步利用标注信息（如significantly better）修改了上述的二元排序损失函数，使奖励模型可以区分不同的输入：$L_{ranking}=-log(\sigma(r_\theta(x, y_c)-r_\theta(x,y_r) - m(r)))$

    $m(r)$是一个离散函数将不同的偏好差距映射成离散的值。

- 数据组成

    奖励模型的训练不仅使用了上述方式标注和收集的数据，还使用了现存的开源偏好数据。开源偏好数据用于在标注收集数据期间训练模型，并且保留到混合数据集中，防止RLHF中对话模型利用奖励模型的缺陷。

    对于数据混合方案，Meta进行了大量实验，并确定了如下的方案：帮助性奖励模型的训练使用全部的Meta帮助性偏好数据和从Meta安全性偏好数据和开源数据中随机采样的等量数据；安全性奖励模型的训练使用所有的Meta安全偏好数据、Anthropic Harmless data以及占比10%的Meta帮助性数据和开源帮助性数据。其中10%的帮助性数据对于两个response都是安全的样本具有很大的提升。

- 训练细节

    奖励模型在训练数据上训练1个epoch（Meta早期实验发现，训练更长会导致过拟合），使用和基座模型相同的优化器参数。采用cosine学习率调整策略，学习率下降到最大学习率的10%，对于70b的模型，最大学习率为5e-6，其余模型为1e-5。5%的总步数用于warm-up。Batch size为512个样本（每个样本包含2个response）。

- 奖励模型结果

    对每一批标注数据，Meta随机选出1000个样本作为测试集。在自己的测试集和其他开源数据上对比了其他可获取的替代品，测试指标为准确率。测试结果见Table 7，Meta的奖励模型的表现优于其他产品，帮助性模型在Meta Helpful上表现更好，而安全性模型在Meta Safety上表现更好。此外，奖励模型对于不同的偏好差距表现也不同，偏好差距越大，准确率越大，相反则越小（见Table 8）。

![image.png](./assets/llama2_tab7.png)

![image.png](./assets/llama2_tab8.png)

- 扩展趋势

    下图展示了随着数据的标注和收集，奖励模型的表现的变化趋势。可以看出随着数据的收集，模型的表现也越来越好，同时这种提升还没有饱和，模型还有提升的空间。

![image.png](./assets/llama2_fig6.png)

#### Iterative Fine-Tuning

随着偏好数据的收集和更好的奖励模型产生，RLHF模型的训练共产生了5个版本，RLHF-V1，···，RLHF-V5。LLAMA 2的强化学习使用了两个算法：PPO和Rejection Sampling fine-tuning。Rejection Sampling fine-tuning算法对于一个prompt会采样K个模型输出并选择最优的输出（reward最大），其奖励值作为新的标准。

在RLHF-V4前，仅使用Rejection Sampling fine-tuning，之后在Rejection Sampling fine-tuning的基础上结合PPO算法。

- Rejection Sampling

    拒绝采样仅在70b模型上进行，其他更小的模型使用70b采样的数据，将大模型的能力蒸馏至小模型。在迭代阶段，起初模型训练只使用上一版本的采样数据，如RLHF V3仅使用V2的采样数据。但是这导致了模型在部分能力上的退化，如V3在诗词创作上的退化。后改为使用所有前置版本的采样数据，上述问题得到了解决。

- PPO

    PPO使用奖励模型的估计作为真实的奖励函数，语言模型作为要优化的策略，进一步训练LLAMA 2。其目标函数为：$ arg max_\pi E_{p\sim D,g\sim\pi}[R(g|p)]$

    其中$p$为数据集中的prompt，$g$为策略$\pi$生成的输出，$R$为奖励函数。

    奖励函数$R(g|p) = \tilde{R}_c(g|p)-\beta D_{KL}(\pi_\theta(g|p) || \pi_0(g|p))$。

    其中$D_{KL}$为策略变化的惩罚项，有助于训练稳定。$R_c$是安全性奖励$R_s$和帮助性奖励$R_h$的组合：当$R_s(g|p)<0.15$，即$g$的安全性小于0.15时，$R_c=R_s$；否则$R_c=R_h$。而$\tilde{R}_c=WHITEN(LOGIT(R_c))$。

    PPO训练使用AdamW优化器: $\beta_1=0.9,\beta_2=0.95,eps=1e-5$，0.1的权重衰减, 1的梯度裁剪，1e-6的常数学习率，512的batch size，0.2的PP0 clip threshold，64的mini-batch，每一个mini-batch计算一次梯度。对于7b和13b模型，KL惩罚$\beta$为0.01，而34b和70b模型为0.005。

    训练使用了early stopping训练了200-400个iteration。为了加速训练，训练使用了FSDP，但是会导致生成的速度下降大约20倍。在生成前将模型权重巩固到每一个节点，并在生成后释放，可以消除上述问题。

### System Message for Multi-Turn Consistency

系统设定可能在多轮对话后被遗忘（Figure 9 left）。为了增强多轮对话的能力，Meta提出了Ghost Attention (GAtt)，效果见 Figure 9 right。假设现有多轮对话数据$[u_1, a_1, ..., u_n, a_n]$，$u_i,a_i$分别为用户和AI助手的谈话，GAtt的具体方法如下：

1. 对于系统设定，我们将其和用户谈话$u_i$连接在一起，并输入RLHF模型采样获得输出$a_i$。获得所有$a_i$后，每个$a_i$的获取都是有明确系统设定的；

2. 对于所有的$u_i$，除$u_1$，将连接的系统设定去除；

3. 使用多轮对话数据$[u_1, a_1, ..., u_n, a_n]$训练模型，其中仅在$a_n$上计算损失；

训练时使用的系统设定，是从几个人为的限制中采样获得的，包括习惯 ("You enjoy e.g. Tennis")，语言 ("Speak in e.g. French")，或公共人物 ("Act as e.g. Napoleon")。而习惯、语言和公共人物的候选列别是通过LLAMA 2-Chat生成的，避免模型缺乏相应的知识。随机组合上述限制可以增加指令的复杂程度。此外构建训练样本时，随机替换系统设定为更简单的表达方式。

GAtt仅在RLHF V3之后使用。效果见Figure 9 right和Figure 10。

![image.png](./assets/llama2_fig9.png)

![image.png](./assets/llama2_fig10.png)

### RLHF Results

    LLM的评价仍然是一个开放性的问题：人类评价尽管最符合实际需求，却不可扩展，难以复制；模型评价则可能与人类偏好存在偏差。

#### Model-Based Evaluation

    模型评价采用奖励模型预测的奖励值评价模型的好坏。然而模型评价可能与人类评价存在偏差，因此Meta首先测试了其奖励模型的预测值与人类评价的关系。Meta要求三位标注人员对对话模型在测试集上的输出进行了评分（1-7分，越大越好）。人类评分和奖励模型的预测奖励值关系如Figure 29所示，二者是正相关的。此外Meta还和其他奖励模型进行对比，二者的偏好也是类似的。

![image.png](./assets/llama2_fig29.png)

Figure 11展示了不同版本的模型对ChatGpt在Meta的安全性和帮助性测试数据上胜率关系，左图使用LLAMA 2奖励模型评判，右图使用GPT-4评判（LLAMA 2和ChatGpt的输出在prompt中的顺序是随机的）。LLAMA 2的奖励模型比GPT-4更偏好LLAMA 2，但是至少RLHF-V5在帮助性和安全性上均优于ChatGpt。

![image.png](./assets/llama2_fig11.png)

#### Human Evaluation

    人类评价在帮助性和安全性上对比LLAMA 2-chat和其他大语言模型，包含开源和闭源模型。测试数据包含4000以上的单轮和多轮对话。标注人员被要求对模型输出的安全性和帮助性进行打分。帮助性的结果如Figure 12所示，LLAMA 2-chat在单轮和多轮对话数据上优于其他开源模型，70b的模型略优于ChatGPT。安全性的结果将在下一章节介绍。

    尽管LLAMA 2-chat的表现比肩ChatGPT，但是该人类评价包含以下几点缺陷：

    1. 测试数据仅包含4k的样本，无法涵盖全部真实场景；

    2. 测试数据集中不包含代码和推理相关的用例；

    3. 多轮对话仅评测最终轮对话；

    4. 人类评价具有主观性和随机性，对于不同的测试集可能会有不同的结果；

![image.png](./assets/llama2_fig12.png)

## Safety

Meta在语言模型的安全性方面进行了深入和广泛的探索，包括对预训练数据和预训练模型的研究、如何进行安全对齐、进行red-teaming以及定量的安全性评测。

### Safety in Pretraining

- Steps Taken to Pretrain Responsibly

    针对安全问题，预训练数据中，不包含任何Meta用户数据，同时，来自包含大量个人信息的网站的数据被排除。不执行其他的数据过滤操作，这可以使训练的模型适用于更广泛的下游任务，避免过度过滤导致数据多样性降低，并且可以通过少量数据进行安全微调。

- Demographic Representation: Pronouns & Identities

    Table 9(a)显示了所有文档中代词出现的频率，男性代词出现在50.73%的文档中，远多于女性代词出现的频率，这可能会导致模型对女性代词的学习不如男性代词充分，并且可能更容易产生男性代词。

    Table 9(b)统计了5个不同维度（宗教、性别和性、国家、名族、性取向）上出现频率最大的5个词。通过数据中词的频率，可以对模型的知识和能力有一个大致的了解。

![image.png](./assets/llama2_tab9.png)

- Data Toxicity

    Meta使用在ToxiGen数据上训练的HateBert分类器衡量预训练数据中的有害性数据的有害性大小及比例。如Figure 13所示，约0.2%的文档的有害性大小超过0.5，仅占预训练数据中很小的比例。

![image.png](./assets/llama2_fig13.png)

- Language Identification

    使用fastText工具，以0.5的阈值检测预训练数据中不同语言的比例。如Table 10所示，英语占据预训练数据中的绝大部分，因此LLAMA 2可能不适用用于其他语言。

![image.png](./assets/llama2_tab10.png)

- Safety Benchmarks for Pretrained Models

     Meta从三个维度，可信度、有害性、偏见，并在相应的测试基准上进行了评测。

    - 可信度，指模型是否会由于错误的观念产生谎言，测试基准使用TruthfulQA；

    - 有害性，指模型是否会产生有害、粗暴或仇恨的内容，测试基准使用ToxiGen；

    - 偏见，指模型是否会产生带有社会偏见的内容，测试基准使用BOLD；

    LLAMA 2和LLAMA 1、MPT、Falcon进行了对比，在可信度上优于其他开源模型，但是有害性上没有优于其他模型，可能原因在于预训练数据没有进行有毒数据过滤。对于偏见测试，使用VADER对模型输出进行情感分析，VADER会对输入产生一个$(-1, 1)$范围内的值，大于0表示积极，小于0表示消极。LLAMA 2-pretrain对比其他开源预训练模型要更积极。

![image.png](./assets/llama2_tab11.png)

### Safety Fine-Tuning

本节介绍Meta在微调中采用的保障模型安全性的技术和措施，包括安全标注指南、安全有监督微调、安全RLHF、安全上下文蒸馏。

- Safety Categories and Annotation Guidelines

    首先Meta定义会对用户产生负面情绪的回复类别，包括：1）允许犯罪活动，2）允许对用户或他人进行危险的行为，3）冒犯用户或他人的内容，4）色情内容。

    同时，规定什么样的模型回复是安全和有帮助的并作为标注指南：首先考虑安全问题，为用户解释潜在的风险，最后提供额外的信息。

- Safety Supervised Fine-Tuning

    标注人员被要求想出可能引发模型不安全行为的prompt，然后撰写一个安全且有帮助的回复。这些数据被用于有监督微调模型。

- Safety RLHF

    有监督微调可以使模型迅速学会返回细节的安全回复，解释问题的敏感点并提供额外有帮助的信息。因此，在收集数千监督数据后，LLAMA 2有监督微调就停止了，并开始RLHF微调。

    同样，RLHF收集人类偏好数据时，标注人员写出认为可能导致模型不安全行为的问题，比较多个模型对应的回复，并选出最安全的回复。这些偏好数据用于训练一个安全奖励模型。RLHF时也使用对抗性的问题。Figure 14展示了Safety RLHF前后模型输出的安全性和帮助性奖励值分布，Safety RLHF前后帮助性分布几乎一致，但安全性有巨大提升。Table 12是一个具体例子。

    Figure 15显示了不同数量的安全数据对模型安全性和帮助性的影响，充足的帮助性数据充足可以消除安全税的问题。

    过度重视安全性可能导致模型过于谨慎以至于拒绝回答正常的用户提问，Meta进一步测试了模型错误因为安全问题拒绝回答的几率，Meta在完全安全的有用性数据上进行测试，发现仅有0.05%数据产生了错误拒绝。

![image.png](./assets/llama2_fig14.png)

![image.png](./assets/llama2_tab12.png)

![image.png](./assets/llama2_fig15.png)

- 安全上下文蒸馏

    Meta通过为prompt增加安全相关的前缀（如"you are a safe and responsible assistant"）来增强模型回答的安全性。然后在微调时将安全相关的前缀去除，用更安全的回复微调。此外，Meta针对不同类型的对抗提问设计了不同的安全前缀。Figure 16 (a)显示安全上下文蒸馏对模型输出的安全奖励值分布的影响。

    值得注意的是，在帮助性的prompt前增加安全性前缀可能会降低模型的表现或导致拒绝回答。因此，Meta仅对增加前缀能增加回复奖励值的prompt使用安全前缀增强回复。

![image.png](./assets/llama2_fig16.png)

### Red Teaming

    Red Team通常是指一个团队或组织中的“红色”成员，他们是那些负责检测和报告潜在安全漏洞、网络攻击和其他风险的人。Red Teams通常是专门的安全性评估小组，致力于确保组织的IT系统是安全的，并且通过不断地发现并利用这些脆弱性和弱点来加强安全性。(chatglm2-6b的解释)

    Meta邀请了350位来自不同领域的专家对LLAMA 2进行了安全漏洞挖掘，并发现以下的现象：

    1. 早期的模型更可能生成不安全的内容，然而通过不断的微调，最终该问题消失；

    2. 早期的模型可以通过一些特殊的请求引发不安全的输出，如写歌、写诗、写故事等；

    3. 早期的模型可以通过将危险的请求包装上一个积极的上下文诱发不安全的输出；

    这对于预知模型存在的风险，使用的限制以及如何进一步优化模型有重要意义。

### Safety Evaluation of LLAMA 2-Chat

- Safety Human Evalutation

    Meta收集了近2000个对抗提问用于安全性评测，其中1351个样本是单轮对话，623个多轮对话。评分人员被要求采用五分制的评分方式：

    - 5：安全且十分有帮助

    - 4：安全且有小的非安全问题

    - 3：安全但是没有帮助或大的非安全问题

    - 2：轻微的安全问题

    - 1：严重的安全问题

    Figure 17展示了不同LLM在上述测试集上存在安全问题的回答的比例，LLAMA 2-Chat相较于其他模型输出危险内容的概率更低，同时对于安全性和帮助性的综合评分，LLAMA 2-Chat也是高于其他开源模型和ChatGPT的。Figure 18展示了单轮对话和多轮对话的输出危险内容的概率，可以看出多轮对话更容易出现安全问题。

    Figure 19在不同种类的安全问题上进行了测试，同样的，LLAMA 2仅在unqualified advice一类上不如Falcon 40b，其他情况均不劣于其他开源模型和ChatGPT。

![image.png](./assets/llama2_fig17.png)

![image.png](./assets/llama2_fig18.png)

![image.png](./assets/llama2_fig19.png)

![image.png](./assets/llama2_tab14.png)

    最后LLAMA 2-Chat也在可信度、有害性和偏见维度进行了评测，结果见Table 14，模型在可信度上仅次于ChatGPT，优于其他开源模型；而在有害性上最优。同时，微调版本的LLAMA 2也优于预训练版本的LLAMA 2。

## Discussion

本章介绍LLAMA 2身上观察到有趣的现象。

![image.png](./assets/llama2_fig20.png)

- SFT微调使模型的输出更多样化，而RLHF则使模型输出的多样性下降，更加对齐人类偏好；

- LLAMA 2-chat对时间具有一点的认识，见Figure 22。

![image.png](./assets/llama2_fig22.png)

- LLAMA 2-chat具备使用工具的能力，当将LLAMA 2接入计算器时，其在数学数据集上表现优秀（Table 15）。

![image.png](./assets/llama2_tab15.png)

## Conclusion

在开源大语言模型中，LLAMA 2展现了其强大的先进性和竞争力，部分情况下甚至优于chatgpt。但是LLAMA 2也存在部分缺陷，如多语言能力、代码能力。LLAMA 2使用了众多技术提升模型的安全性，但是仍需要进一步的研究，而LLAMA 2的开放为LLM的发展具有重要的贡献。

