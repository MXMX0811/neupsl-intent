# neupsl-intent

`neupsl-intent` is a minimal Transformer + NeuPSL demo for BabyAI-style action
prediction.

The task is:

```text
mission + symbolic observation + expert action history -> BabyAIPredictor -> next-action probabilities
next-action probabilities + PSL rules -> fused action prediction
```

The first implementation uses the standard MiniGrid/BabyAI action space:

```text
left, right, forward, pickup, drop, toggle, done
```

## Data

Generate the BabyAI-format data:

```bash
python scripts/create_data.py \
  --pretrain-size 500 \
  --pretrain-valid-size 100 \
  --train-size 50 \
  --valid-size 100 \
  --inference-size 1000
```

The canonical generated layout is:

```text
data/experiment_babyai/
├── action-vocab.json
├── token-vocab.json
├── type-vocab.json
├── pretrain/
│   ├── rules/
│   └── size_0500-valid_0100/
├── neupsl-train/
│   ├── rules/
│   └── size_0050-valid_0100/
└── inference/
    ├── rules/
    └── size_1000/
```

Each split writes token ids, type ids, action labels, PSL targets/truth, and
derived rule observations such as `InvalidAction(step, action)` and
`PlausibleAction(step, action)`.

`--generator-backend auto` uses `minigrid.utils.baby_ai_bot.BabyAIBot` when
MiniGrid is installed and otherwise falls back to a symbolic generator with the
same file schema.

## Model

`models.transformer.BabyAIPredictor` embeds:

```python
token_embedding(token_ids) + type_embedding(type_ids) + position_embedding(pos_ids)
```

It uses CLS pooling and predicts a 7-way action distribution.

Pretrain:

```bash
python scripts/pretrain_transformer.py --epochs 20
```

## NeuPSL

The BabyAI PSL template is:

```text
psl/rules/experiment__babyai.json
```

It separates neural and fused predicates:

```text
NeuralAction(Step, Action)
Action(Step, Action)
InvalidAction(Step, Action)
PlausibleAction(Step, Action)
```

Run NeuPSL:

```bash
python scripts/neupsl_train.py \
  --pretrained-path ckpt/pretrained-babyai-transformer.pt
```

Summarize results:

```bash
python scripts/evaluate.py
```
