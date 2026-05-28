# mnist-add-minreal

`mnist-add-minreal` is a minimal, standalone PyTorch + NeuPSL implementation of
the MNIST Addition experiment. It keeps the original task idea from the IJCAI
2023 NeuPSL repository, but reorganizes the code into small Python modules that
are easy to inspect, replace, and call directly.

The project uses real MNIST images only. There is no synthetic or mocked image
data path.

## What This Project Does

The task is to predict the sum of two MNIST numbers while jointly learning:

- a PyTorch Transformer digit classifier,
- PSL rule weights,
- Transformer digit probabilities and PSL-fused sum predicates.

For `mnist-1`, each example is a pair of single digits:

```text
image_a + image_b -> sum in [0, 18]
```

The project currently keeps only the `mnist-1` task. This makes the data layout,
PSL predicates, and notebooks easier to inspect while still showing Transformer
probabilities, PSL gradients, DeepPredicate digit outputs, and fused `ImageSum` soft
vectors.

## Workflow

The implementation follows this pipeline:

1. Load real MNIST images with `torchvision.datasets.MNIST`.
2. Generate one staged, non-overlapping data layout:
   - supervised pretraining data,
   - NeuPSL training/validation data,
   - held-out inference data.
3. Pretrain the Transformer on the supervised pretraining split.
4. Build a PSL runtime config from a small JSON rule template.
5. Run NeuPSL through `pslpython.runtime`.
6. During learning, PSL calls the PyTorch `DeepModel` bridge:
   - `internal_predict()` returns Transformer digit probabilities to PSL.
   - PSL computes soft-rule violations and gradients.
   - `internal_fit()` backpropagates those gradients through the Transformer.
7. Save the trained Transformer checkpoint, step checkpoints, learned PSL rules, and runtime atoms
   logs for notebook-based inspection.

The final PSL runtime output contains atoms such as:

- `NeuralClassifier(image, digit)`: Transformer/DeepPredicate digit probabilities.
- `ImageSum(image_1, image_2, sum)`: PSL-fused sum soft vector.

## File Structure

```text
mnist-add-minreal/
├── README.md
├── requirements.txt
├── setup.py
├── data/
│   ├── loader.py              # real MNIST loading and normalization
│   ├── generator.py           # PSL-compatible single-digit addition data generation
│   ├── raw/                   # downloaded MNIST files, created on demand
│   └── experiment_mnist-*/    # generated PSL data, created on demand
├── models/
│   ├── transformer.py         # Transformer-based PyTorch classifier
│   └── deeppsl.py             # pslpython.deeppsl DeepModel bridge
├── psl/
│   ├── runner.py              # PSL config builder and runtime/CLI runner
│   └── rules/
│       └── experiment__mnist-1.json
├── scripts/
│   ├── check_env.py           # environment check entry point
│   ├── create_data.py         # data generation entry point
│   ├── pretrain_transformer.py # supervised Transformer pretraining entry point
│   ├── neupsl_train.py        # end-to-end NeuPSL joint fine-tuning entry point
│   └── evaluate.py            # result summarization and JSON/log metric parser
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_demo.ipynb
│   └── 03_inference_demo.ipynb
└── utils/
    ├── config.py              # DatasetConfig and canonical paths
    ├── io.py                  # JSON and PSL TSV helpers
    └── metrics.py             # accuracy helpers
```

## Core Python Interfaces

### Dataset Configuration

`utils.config.DatasetConfig` is the central configuration object used by data
generation, PSL config building, result paths, and notebooks.

```python
from utils.config import DatasetConfig, dataset_dir, inference_dir, neupsl_train_dir, pretrain_dir

config = DatasetConfig(
    name="mnist-1",
    pretrain_size=20000,
    pretrain_valid_size=5000,
    pretrain_overlap=0.4,
    train_size=1000,
    valid_size=1000,
    inference_size=5000,
    overlap=0.0,
    seed=None,
)

pretrain_path = pretrain_dir(config)
neupsl_train_path = neupsl_train_dir(config)
inference_path = inference_dir(config)
```

Important parameters:

| Parameter | Meaning |
|-----------|---------|
| `name` | `"mnist-1"` for single-digit addition |
| `pretrain_size` | Number of MNIST digit images assigned to supervised pretraining |
| `pretrain_valid_size` | Number of MNIST digit images assigned to supervised pretraining validation |
| `pretrain_overlap` | Fractional resampling overlap used only for pretraining train data |
| `train_size` | Number of MNIST digit images assigned to NeuPSL training before forming addition blocks |
| `valid_size` | Number of digit images assigned to the validation partition |
| `inference_size` | Number of digit images assigned to the held-out inference partition |
| `overlap` | Fractional resampling overlap within a generated partition; default demos use `0.0` |
| `seed` | Optional explicit seed; if omitted, a deterministic seed is derived from dataset settings |

Derived fields:

| Field | Meaning |
|-------|---------|
| `num_digits` | Automatically set to `1` |
| `class_size` | Digit class count, currently `10` |
| `max_number` | Largest represented number, `9` |
| `max_sum` | Largest possible addition result, `18` |

Path helpers:

```python
dataset_dir("mnist-1")
pretrain_dir(config)
neupsl_train_dir(config)
inference_dir(config)
```

These return the shared dataset directory and the concrete generated-data directory under
`data/`.

### Real MNIST Loading

`data.loader.load_mnist_arrays()` loads train and test MNIST from torchvision,
concatenates them, flattens each image to 784 features, and normalizes pixel
values to `[0, 1]`.

```python
from data.loader import load_mnist_arrays

features, labels = load_mnist_arrays(
    root=None,
    download=True,
)
```

Parameters:

| Parameter | Meaning |
|-----------|---------|
| `root` | Optional MNIST download/cache directory; defaults to `data/raw/` |
| `download` | Whether torchvision may download MNIST if missing |

Return values:

| Value | Shape / type |
|-------|--------------|
| `features` | `np.ndarray` with shape `[70000, 784]`, normalized and rounded |
| `labels` | `np.ndarray` with shape `[70000]`, integer digit labels |

### PSL Data Generation

`data.generator.generate_experiment_datasets()` writes the staged data layout
used by the scripts and notebooks. It shuffles MNIST once, then allocates
disjoint image ids to pretraining, NeuPSL training/validation, and inference.

```python
from data.generator import generate_experiment_datasets
from data.loader import load_mnist_arrays
from utils.config import DatasetConfig

features, labels = load_mnist_arrays()
config = DatasetConfig(name="mnist-1")
paths = generate_experiment_datasets(config, features, labels)

pretrain_path = paths["pretrain"]
neupsl_train_path = paths["neupsl_train"]
inference_path = paths["inference"]
```

Parameters:

| Parameter | Meaning |
|-----------|---------|
| `config` | `DatasetConfig` controlling task, staged partition sizes, overlap, and seed |
| `features` | Flattened MNIST feature matrix |
| `labels` | Digit labels aligned with `features` |
| `data_root` | Optional output root; defaults to `data/` |

Default generated directories:

```text
data/experiment_mnist-1/
├── pretrain/
│   └── size_20000-valid_5000-overlap_0.40/
├── neupsl-train/
│   └── size_1000-valid_1000-overlap_0.00/
└── inference/
    └── size_5000/
```

Main generated files:

| File | Meaning |
|------|---------|
| `entity-data-map.txt` | `image_id`, 784 features, true digit label |
| `image-target-{partition}.txt` | Target rows for `NeuralClassifier(image, digit)` |
| `image-sum-block-{partition}.txt` | Addition examples as image id blocks |
| `image-sum-target-{partition}.txt` | Target rows for possible sum values |
| `image-sum-truth-{partition}.txt` | One-hot truth rows for true sums |
| `number-sum.txt` | Symbolic digit addition facts |
| `possible-digits.txt` | Symbolic consistency facts between digits and sums |
| `config.json` | Dataset metadata, including `"data-source": "mnist"` |

Useful helper functions in `data.generator`:

| Function | Purpose |
|----------|---------|
| `digits_to_number(digits)` | Convert a digit sequence to an integer |
| `digits_to_sum(digits, num_digits)` | Compute left-number plus right-number from a block |
| `generate_split(config, labels, indexes)` | Convert image indexes into addition blocks and sum labels |
| `create_entity_data_map(features, labels, entities)` | Build rows for `entity-data-map.txt` |
| `create_image_data(config, entities)` | Build `NeuralClassifier` target rows |
| `create_image_sum_data(config, sum_entities, sum_labels)` | Build `ImageSum` targets and truth rows |
| `write_shared_data(config, out_dir)` | Write task-level symbolic facts |
| `write_partition_data(config, out_dir, features, labels, partition_indexes, overlaps=None)` | Write a concrete set of partitions |
| `generate_experiment_datasets(config, features, labels)` | Generate staged pretraining and NeuPSL/inference data |

### Transformer Model

`models.transformer.MNISTTransformerClassifier` is a small replaceable PyTorch
classifier. It uses two Transformer encoder layers with 64 hidden dimensions and
4 attention heads, pools the Transformer output to 256 features, then uses the
unchanged `256 -> 120 -> 84 -> num_classes` classifier head. Its `forward()`
method returns probabilities, while `logits()` returns raw logits for supervised
training utilities.

```python
from models.transformer import MNISTTransformerClassifier, save_checkpoint, load_checkpoint

model = MNISTTransformerClassifier(num_classes=10)
probs = model.predict(features[:8])
save_checkpoint("results/demo/model.pt", model)
restored = load_checkpoint("results/demo/model.pt")
```

Core methods and functions:

| Interface | Parameters | Return |
|-----------|------------|--------|
| `MNISTTransformerClassifier(num_classes=10)` | Number of output digit classes | PyTorch module |
| `forward(x)` | `[batch, 784]`, `[batch, 28, 28]`, or tensor image batch | Softmax probabilities |
| `logits(x)` | Same accepted input shapes | Raw class logits |
| `predict(x, device="cpu")` | Array-like image batch and device | `np.ndarray` probabilities |
| `save_checkpoint(path, model, metadata=None)` | Path, model, optional metadata | Writes `.pt` checkpoint |
| `load_checkpoint(path, map_location="cpu")` | Checkpoint path and device | Restored `MNISTTransformerClassifier` |

Supervised pretraining uses an explicit PyTorch training loop in
`scripts/pretrain_transformer.py`. NeuPSL training updates the Transformer
classifier through PSL gradients in `MNISTDeepPSL.internal_fit()`.

### DeepPSL Bridge

`models.deeppsl.MNISTDeepPSL` is the class loaded by PSL as a deep predicate
model. It also exposes the alias `MNISTAdditionModel`, which is the default
class name used in generated PSL configs.

```python
from models.deeppsl import MNISTDeepPSL, MNISTAdditionModel
```

Lifecycle methods called by PSL:

| Method | Purpose |
|--------|---------|
| `internal_init_model(application, options=None)` | Initialize/load the PyTorch Transformer classifier, optimizer, checkpoint settings, and device |
| `internal_predict(data, options=None)` | Return Transformer probabilities for PSL deep predicate atoms |
| `internal_fit(data, gradients, options=None)` | Backpropagate PSL-provided gradients through Transformer probabilities |
| `internal_eval(data, options=None)` | Return prediction metrics for inference/evaluation |
| `internal_save(options=None)` | Save the trained Transformer checkpoint |

Important options passed through the PSL config:

| Option | Meaning |
|--------|---------|
| `class-size` | Number of digit classes |
| `learning-rate` | Adam learning rate for Transformer updates |
| `batch-size` | Neural forward/backward batch size inside each full-graph NeuPSL step |
| `checkpoint-dir` | Directory for step checkpoints and automatic resume |
| `checkpoint-frequency` | Save a Transformer checkpoint every N NeuPSL gradient steps; use `0` to disable |
| `random-seed` | Optional NumPy/PyTorch seed for model initialization |
| `save-path` | Directory or `.pt` path used by `internal_save()` and inference loading |
| `pretrained-path` | Optional checkpoint used to initialize the Transformer before NeuPSL learning |
| `device` | Optional PyTorch device, default `cpu` |

### PSL Runtime Runner

`psl.runner.PSLRunner` builds a concrete PSL config and runs it through either
the Python runtime bridge or the PSL CLI jar.

```python
from pathlib import Path

from psl.runner import PSLRunner
from utils.config import DatasetConfig

config = DatasetConfig(name="mnist-1")
result_dir = Path("results/demo").resolve()

runner = PSLRunner()
config_path = runner.build_config(
    config,
    output_path=result_dir / "psl-config.json",
    learning_rate=1e-3,
    batch_size=256,
    gradient_steps=50,
    admm_iterations=5,
    checkpoint_dir=result_dir / "checkpoints",
    checkpoint_frequency=10,
    random_seed=16,
    save_path=result_dir / "saved-networks" / "nesy-trained-pt",
)

payload = runner.run_runtime(config_path, result_dir)
```

`PSLRunner` constructor:

| Parameter | Meaning |
|-----------|---------|
| `jar_path` | Optional local PSL CLI jar path |
| `psl_version` | PSL CLI version for jar download, default `2.4.0` |

`build_config()` parameters:

| Parameter | Meaning |
|-----------|---------|
| `config` | `DatasetConfig` for dataset and task shape |
| `data_root` | Generated data root, default `data/` |
| `output_path` | Where to write the concrete PSL config |
| `learning_rate` | Learning rate passed to the DeepModel |
| `batch_size` | Neural classifier batch size used inside each full-graph NeuPSL step |
| `gradient_steps` | PSL gradient descent steps; each can update Transformer parameters |
| `admm_iterations` | Optional quick-demo override for PSL ADMM iterations |
| `checkpoint_dir` | Directory for step checkpoints; defaults to `result_dir/checkpoints` |
| `checkpoint_frequency` | Save a Transformer checkpoint every N NeuPSL gradient steps; use `0` to disable |
| `log_level` | PSL runtime log level |
| `rule_path` | Optional replacement PSL rule/config JSON |
| `model_path` | Optional replacement DeepModel Python file |
| `model_class` | Class name inside `model_path`, default `MNISTAdditionModel` |
| `random_seed` | Optional DeepModel seed |
| `save_path` | Directory or `.pt` path for trained Transformer checkpoint |
| `pretrained_path` | Optional Transformer checkpoint used to initialize NeuPSL learning |

Runtime methods:

| Method | Purpose |
|--------|---------|
| `run_runtime(config_path, output_dir)` | Run `pslpython.runtime.run()`, write `runtime-output.json`, `learned-rules.json`, and logs |
| `run(config_path, output_dir, extra_options=None)` | Run the PSL CLI jar with Java |
| `ensure_jar()` | Download/return the PSL CLI jar for CLI backend |
| `run_psl_experiment(config_path, output_dir, extra_options=None)` | Convenience helper returning `(runner, completed_process)` |

The generated runtime config uses H2 instead of SQLite:

```text
runtime.db.type = H2
runtime.db.h2.inmemory = false
```

This avoids SQLite upsert/commit issues observed with deep predicates.

### Result Loading and Inference Inspection

After training, the main artifacts are written under:

```text
results/train_<train_size>_infer_<inference_size>/
```

Important artifacts:

| File | Meaning |
|------|---------|
| `model.pt` | Trained PyTorch Transformer checkpoint copied from DeepPSL save path |
| `runtime-output.json` | PSL runtime output, including fused atoms and evaluations |
| `learned-rules.json` | Learned PSL rule weights |
| `learned-rules.txt` | Human-readable learned-rule dump |
| `checkpoints/step_*.pt` | Transformer checkpoints saved every N NeuPSL gradient steps and used for resume |
| `psl-config.json` | Concrete config used for the run |

Python-side loading:

```python
from collections import defaultdict

from models.transformer import load_checkpoint
from utils.io import load_json

model = load_checkpoint(result_dir / "model.pt")
payload = load_json(result_dir / "runtime-output.json")
learned_rules = load_json(result_dir / "learned-rules.json")

neural_atoms = defaultdict(dict)
image_sum_atoms = defaultdict(dict)
for atom in payload["atoms"]:
    args = [int(value) for value in atom["arguments"]]
    if atom["predicate"] == "NEURALCLASSIFIER":
        image_id, digit = args
        neural_atoms[image_id][digit] = float(atom["value"])
    elif atom["predicate"] == "IMAGESUM":
        image_1, image_2, sum_value = args
        image_sum_atoms[(image_1, image_2)][sum_value] = float(atom["value"])
```

The inference notebook wraps this pattern and visualizes multiple held-out inference pairs as
subplot grids:

- two real MNIST images per addition example,
- Transformer digit probabilities and predictions,
- Transformer/DeepPredicate digit probabilities,
- PSL-fused `ImageSum` soft vectors and predicted sums.

### Utility Interfaces

`utils.io`:

| Function | Purpose |
|----------|---------|
| `ensure_dir(path)` | Create and return a directory path |
| `write_json(path, data, indent=2)` | Write JSON with parent directory creation |
| `load_json(path)` | Read JSON |
| `write_psl(path, rows)` | Write tab-separated PSL rows |
| `load_psl(path, dtype=str)` | Read tab-separated PSL rows |

`utils.metrics`:

| Function | Purpose |
|----------|---------|
| `one_hot(labels, num_classes)` | Convert integer labels to one-hot arrays |
| `categorical_accuracy(y_pred, y_true)` | Digit classification accuracy |
| `digit_pair_sum_accuracy(digit_probs, entity_ids, sum_truth_rows, num_digits)` | Sum accuracy derived from digit predictions |

`scripts.check_env.check_environment()`:

```python
from scripts.check_env import check_environment

status = check_environment()
```

Returns Python, Java, `pslpython`, and `pslpython.runtime` availability in a
dictionary. The command-line script is just a thin wrapper around this function.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration.ipynb` | Generate real-MNIST data, show image pairs, sums, PSL rows, and rule snippets |
| `02_training_demo.ipynb` | Read prepared NeuPSL data, run one NeuPSL training job, and show key-step Transformer vectors plus PSL gradient stats |
| `03_inference_demo.ipynb` | Load saved model/rules/runtime atoms; show held-out inference examples with Transformer, PSL digit soft labels, and `ImageSum` vectors |

The notebooks are intended for interpretation and visualization. The reusable
logic lives in the Python modules above.

## Replacing Models or Rules

To replace the neural model, implement a `pslpython.deeppsl.model.DeepModel`
class with the same lifecycle methods as `MNISTDeepPSL`, then pass its file and
class name through `PSLRunner.build_config(model_path=..., model_class=...)`.

To replace PSL rules, provide a compatible JSON rule/config template through
`PSLRunner.build_config(rule_path=...)`.

The replacement rule file must preserve the predicate names and arities expected
by the data generator, or the generator and config builder must be updated
together.

## Requirements

- Java available on `PATH`
- `pslpython == 2.4.0`
- `torch == 2.3.0`
- `torchvision == 0.18.0`
- `numpy`
- `matplotlib` and `jupyter` for notebooks

`pslpython` may print a socket shutdown traceback after a successful DeepPSL run
on some platforms. In the tested environment, learning, inference, runtime JSON,
learned rules, model checkpoint, and runtime output were still written correctly.

## CLI Commands

The project is designed around Python functions, but the `scripts/` files expose
thin command-line wrappers for full runs.

Check the environment:

```bash
cd mnist-add-minreal
/Users/zmx/.pyenv/versions/.neupsl_env/bin/python scripts/check_env.py
```

Generate real-MNIST PSL data:

```bash
/Users/zmx/.pyenv/versions/.neupsl_env/bin/python scripts/create_data.py \
  --dataset mnist-1 \
  --pretrain-size 20000 \
  --pretrain-valid-size 5000 \
  --pretrain-overlap 0.4 \
  --train-size 1000 \
  --valid-size 1000 \
  --inference-size 5000 \
  --overlap 0.0
```

Pretrain the Transformer classifier on the same train/valid data:

```bash
/Users/zmx/.pyenv/versions/.neupsl_env/bin/python scripts/pretrain_transformer.py \
  --dataset mnist-1 \
  --pretrain-size 20000 \
  --pretrain-valid-size 5000 \
  --pretrain-overlap 0.4 \
  --epochs 10 \
  --batch-size 256 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --grad-clip 1.0
```

Run end-to-end NeuPSL training initialized from the supervised checkpoint. The
runtime also performs the PSL inference phase and writes fused atoms for the
inference notebook:

```bash
/Users/zmx/.pyenv/versions/.neupsl_env/bin/python scripts/neupsl_train.py \
  --dataset mnist-1 \
  --train-size 1000 \
  --valid-size 1000 \
  --inference-size 5000 \
  --overlap 0.0 \
  --pretrained-path ckpt/pretrained-transformer.pt \
  --batch-size 256 \
  --gradient-steps 500 \
  --admm-iterations 100 \
  --checkpoint-frequency 10
```

Summarize and parse results:

```bash
/Users/zmx/.pyenv/versions/.neupsl_env/bin/python scripts/evaluate.py \
  --dataset mnist-1 \
  --train-size 1000 \
  --valid-size 1000 \
  --inference-size 5000 \
  --overlap 0.0 \
  --results-dir results/train_1000_infer_5000
```

Run the inference visualization notebook after training artifacts exist:

```bash
/Users/zmx/.pyenv/versions/.neupsl_env/bin/jupyter notebook notebooks/03_inference_demo.ipynb
```

See [docs/PROJECT_MAP.md](docs/PROJECT_MAP.md) for a compact maintenance map.
