# neupsl-intent Project Map

> Last Updated: 2026-05-28
> Project: Minimal Transformer + NeuPSL BabyAI Action Prediction

## Current State

The repository has been migrated away from MNIST Addition and now targets
BabyAI-style behavior prediction:

```text
mission + symbolic observation + expert action history -> BabyAIPredictor -> NeuralAction
NeuralAction + InvalidAction/PlausibleAction PSL facts -> Action
```

The first implementation keeps the original high-level layout while replacing
MNIST APIs, predicates, generated files, and scripts with BabyAI equivalents.

## Directory Structure

```text
neupsl-intent/
├── README.md
├── requirements.txt
├── setup.py
├── data/
│   ├── __init__.py
│   ├── loader.py
│   ├── sequence_extraction.py
│   ├── rule_extraction.py
│   └── experiment_babyai/       # generated
├── models/
│   ├── __init__.py
│   ├── transformer.py           # BabyAIPredictor
│   └── deeppsl.py               # BabyAIActionModel
├── psl/
│   ├── __init__.py
│   ├── runner.py
│   └── rules/
│       └── experiment__babyai.json
├── scripts/
│   ├── check_env.py
│   ├── create_data.py
│   ├── pretrain_transformer.py
│   ├── neupsl_train.py
│   └── evaluate.py
├── notebooks/
├── docs/
└── utils/
```

## Data Flow

`scripts/create_data.py` orchestrates generation through:

- `data.sequence_extraction`: builds step-level next-action examples.
- `data.rule_extraction`: derives rule facts and split-local rule libraries.
- `data.loader`: loads token/type/action arrays for training.

Generated files include:

- `sequence-data-{partition}.jsonl`
- `entity-data-map.txt`
- `entity-type-map.txt`
- `action-target-{partition}.txt`
- `action-truth-{partition}.txt`
- `invalid-action-{partition}.txt`
- `plausible-action-{partition}.txt`

## Model

`BabyAIPredictor` combines token, type, and position embeddings:

```python
token_embedding(token_ids) + type_embedding(type_ids) + position_embedding(pos_ids)
```

It uses a small Transformer encoder, CLS pooling, and a 7-way classifier over:

```text
left, right, forward, pickup, drop, toggle, done
```

## PSL

The BabyAI PSL template lives at `psl/rules/experiment__babyai.json` and uses:

- `NeuralAction(Step, Action)` as the DeepPredicate output.
- `Action(Step, Action)` as the PSL-fused prediction.
- `InvalidAction(Step, Action)` and `PlausibleAction(Step, Action)` as derived
  rule facts.

Core rules:

```text
NeuralAction(S,A) -> Action(S,A)
InvalidAction(S,A) -> ~Action(S,A)
PlausibleAction(S,A) -> Action(S,A)
Action(S,+A) = 1 .
```
