<!--
 * @Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
 * @Date: 2026-05-28 09:11:19
 * @LastEditors: Mingxin Zhang
 * @LastEditTime: 2026-05-28 14:37:36
 * Copyright (c) 2026 by Mingxin Zhang, All Rights Reserved. 
-->
# Transformer + NeuPSL最小实现

该项目是一个从 https://github.com/linqs/neupsl-ijcai23 重构而来的PyTorch最小实现。该项目呈现了一个意图预测的demo，通过Transformer接收当前状态与历史动作序列，输出预测动作的概率软标签，并且与PSL基于规则推理的结果进行融合，得到预测的动作类别。其不依赖原项目，可独立运行，同时在项目结构上重新做了调整以适应新的任务目标。该项目包含Transformer的预训练以及NeuPSL的联合微调。关于项目的详细信息请参考README.md以及docs中的项目地图和handoff。不要对目前的项目结构进行大改，按照现在的目录将MNIST的demo修改至在babyai数据集上做的行为预测。

## 数据集准备

该项目使用minigrid library中的babyai数据集。首先，基于项目Transformer接收历史动作序列，输出预测动作类别概率软标签的要求，请你规划应该如何组织数据集。同时，由于PSL需要做规则约束，请你考虑如何从数据集中提取规则并存储/维护规则。规则库的格式与存放可以参考目前项目MNIST的规则与位置。

为了适配NeuPSL适应的低数据场景，请准备约200-500条动作序列（可以不定长）与50-100条规则用于Transformer的预训练。同时再准备100条左右的动作序列用作预训练时的验证集。然后，准备50条动作序列与10条对应于这50条动作序列但不同于预训练数据集的规则，将新增规则与预训练的规则一同作为新的规则库，用作NeuPSL微调时的训练集。同样准备100条左右的动作序列用作微调验证集。注意，请你保证预训练时的训练集与验证集既要不重叠，又要属于同一分布；NeuPSL的数据集也是一样。但是预训练数据集与NeuPSL微调使用的数据集在分布上应有区别，以便观察微调的效果。最后，准备1000条动作序列用作推理，推理数据集的分布应与微调时一致。

请你在data目录中拆分并重组generator脚本，构建rule_extraction.py和sequence_extraction.py，分别用于规则的提取与动作序列的提取。然后在scripts中的create_data中调用这两个脚本，产生用于预训练/微调/推理的数据集。首先，先根据你先前给出的建议构建用于预训练/微调的规则库（推理使用和微调时相同的规则库），可以提取下面两类规则：

1. 环境动力学规则
这是最可靠的一类，因为 MiniGrid/BabyAI 的世界是程序化环境，动作含义固定。比如：
WallAhead(t) -> not Action(t, forward)
DoorAhead(t) & Closed(t, door) -> Action(t, toggle) is plausible
LockedDoorAhead(t) & not HasMatchingKey(t) -> not Action(t, toggle/open)
ObjectHere(t, obj) & MissionPickup(obj) -> Action(t, pickup)
CarryingTarget(t) & MissionPickup(obj) -> Action(t, done)
这些规则不是统计相关性，而是环境规则/动作前置条件，最适合 PSL。
2. 任务语义规则
BabyAI 的 mission 是由 grammar 生成的，所以很多指令可以解析成结构化目标：
"go to the red ball"
"pick up the blue key"
"open the yellow door"
"put the green box next to the red ball"
可以抽成 predicates：

MissionGoTo(obj)
MissionPickup(obj)
MissionOpen(door)
TargetColor(obj, red)
TargetType(obj, ball)
然后写规则：

MissionGoTo(obj) & AtAgentNear(obj, t) -> Action(t, done)
MissionPickup(obj) & AtSameCell(obj, t) & EmptyHand(t) -> Action(t, pickup)
MissionOpen(door) & DoorAhead(door, t) & Closed(door, t) -> Action(t, toggle)
这类规则比“从轨迹频率中学出来的规则”更干净，因为它来自任务定义。

在根据这两种方案提取规则后，请你参考现在用于MNIST的规则存放在哪里，用什么格式存储；用相同的流程来维护提取出的预训练规则库/在预训练规则库上加入新增规则用于微调和推理的规则库。

修改notebook中的data exploration以展示提取出的动作序列与规则格式；使用T-SNE展示预训练/微调/推理三个数据集的分布差异。

## 模型结构调整

请你根据babyai数据集中的动作空间来设计Transformer的输出。由于babyai数据集中动作空间较小，因此视作简单分类问题设计输出即可。

现在的输入是自然语言构成的动作序列，请你设计合适的embedding方式以让Transformer接受序列输入。

可以不考虑预训练的时间与资源开销；但是微调过程需要控制在5-10 min。在当前数据规模下应该是一个充裕的时间设置。

## 项目结构调整

尽量按照目前MNIST demo的项目结构组织。清除目录中所有与MNIST有关的内容以及旧API，重新适配新的任务。删除experiments目录中全部内容，留作重构demo后对新任务做性能调优的实验。三个notebook的功能可以保持不变。审阅整个项目，做完全调整与适配，使项目完全移至babyai，并清除旧api防止冲突。

## 需求修订v1

data目录中不要保留generator，所有与动作序列相关的操作全部移至sequence_extration，同理规则提取完全在rule_extraction进行。提取数据的整体组织可以在create_data中进行，请你给出建议是继续保持其在scripts中还是移动至data。

data/experiment_babyai/中数据集目录只提及规模即可，不需要size_0050-valid_0100-newrules_0010/这样连规则也提及。规则可以分至各个数据集，虽然会造成重复但是便于检索、增删与维护：

data/experiment_babyai/
├── pretrain/
│   ├── size_0500-valid_0100/
│   └── rules/
├── neupsl-train/
│   ├── size_0050-valid_0100/
│   └── rules/
├── inference/
│   ├── size_1000/
│   └── rules/
└── action-vocab.json

调整命名，models/transformer.py 应从 MNISTTransformerClassifier 改成 BabyAI Predictor，例如：

BabyAIPredictor

不需要的api或参数如overlap等应移除。

data exploration如果可能的话应对场景进行可视化展示，如环境可视化、动作序列可视化等。

## 需求修订v2

可以拆分episode为step-level next-action prediction，但每条序列的长度最好>=3，以保证有足够的序列特征。另外不要让情景陷入类似RNN的递归预测，充分发挥Transformer相对传统RNN的优势。

你提到“按照修订版 AGENTS，每个 split 都可以有自己的 rules/”，应为rules应该跟着数据集走，因为规则是从历史行为序列中产生的，因此也应与数据集互相对应。不同的分割可能包含有不同的行为序列，规则也应有所不同。你对“预训练规则库可以包含通用规则，NeuPSL 新增规则可以引入新任务语义”的理解是准确的。

使用babyaibot来生成行为轨迹，先考虑简单场景，即你建议的：

1. BabyAI-GoToObj-v0
2. BabyAI-Pickup-v0 或 BabyAI-PickupLoc-v0
3. BabyAI-OpenDoor-v0
4. BabyAI-GoToSeq-v0 / SynthSeq

请在.neupsl_env中配置所需环境，并更新环境要求到requirements.txt。

tokenization一开始先按照给每个token一个id来做：

```
mission: "go to the red ball"

token_id:   go   to   the  red  ball
type_id:    M    M    M    M    M

observation: target red ball ahead

token_id:   target red ball ahead
type_id:    OBS    OBS OBS  OBS

history: forward left

token_id:   forward left
type_id:    HIST    HIST

tokens:
[CLS] pick up red ball [SEP] agent facing east front ball red empty hand target ahead [SEP] forward left [SEP]

types:
SPECIAL MISSION MISSION MISSION MISSION SPECIAL
OBS OBS OBS OBS OBS OBS OBS OBS OBS OBS SPECIAL
HISTORY HISTORY SPECIAL
```

embedding成 `x = token_embedding(token_ids) + type_embedding(type_ids) + position_embedding(pos_ids)`

让模型能够学习同一个token的概念，不会因为情景不同就对同一个词产生不同理解。

在流程跑通后，考虑升级为不同层级的token有各自单独的encoder作为不同通道：

```python
mission_encoder(mission_tokens)
state_encoder(observation_tokens)
history_encoder(history_action_tokens)

z = concat([mission_repr, state_repr, history_repr])
logits = classifier(z)
```

或者用 cross-attention 来融合各通道embedding。

## 需求修订v3

数据集规模按episode确定，训练的时候将episode拆分为segment用于训练模型。

过滤短轨迹以保证序列长度要求与时间序列特征，不进行自回归的递归预测。

可以按你给出的方案

```
pretrain:
  BabyAI-GoToObj-v0
  BabyAI-Pickup-v0

neupsl-train + inference:
  BabyAI-PickupLoc-v0
  BabyAI-OpenDoor-v0
```

来进行第一步的数据集划分。

observation token可以先在以下范围：

```
agent facing <dir>
front <empty|wall|door|key|ball|box>
front color <color>          # 如果有颜色
front state <open|closed|locked> # 如果是 door
hand <empty|key|ball|box>
hand color <color>           # 如果 carrying
target <type> <color>
target relation <ahead|left|right|behind|unknown>
target distance <near|mid|far|unknown>
mission kind <goto|pickup|open>
```

跑通流程之后再完全展开为完整grid。

非法规则可以用你提到的方案 B：

rule_extraction 先生成派生 facts：

```
InvalidAction(S, forward)
PlausibleAction(S, pickup)
```

PSL 规则：

```
InvalidAction(S,A) -> ~Action(S,A)
PlausibleAction(S,A) -> Action(S,A)
```

Action 与 NeuralAction 要分开，因为这是两个部分，代表神经输出和规则输出，之后还要进行融合，分开表示方便后续验证。

NeuPSL微调阶段先按照原始NeuPSL框架进行，如果效果不佳或调整太弱再考虑加入显式的Ground truth约束。

Transformer可以按照以下配置搭建：

```
max_seq_len = 96
embedding_dim = 64
num_layers = 2
num_heads = 4
dropout = 0.1
pooling = CLS token
classifier = Linear/ReLU/Linear
```

entity-data-map按照你给的方案，分别存储

```
entity-data-map.txt
203    1 14 15 16 17 2 40 ... 0    3

entity-type-map.txt
203    0 1 1 1 1 0 2 ... 0
```

以保证可扩展性。根据新的data map调整系统的输入输出。

Transformer的输入保留可选参数 `mission_encoding = "surface" | "structured"`，提供可选项，用参数决定模型mission字段的输入是直接的mission string还是通过instrs获得的结构化输入。
