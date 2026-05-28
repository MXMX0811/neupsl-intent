# mnist-add-minreal Project Map

> Last Updated: 2026-05-22
> Project: Minimal PyTorch + NeuPSL MNIST Addition

---

## Current State

`mnist-add-minreal` is a standalone minimal implementation of the MNIST Addition
NeuPSL experiment. It uses real MNIST data only and trains a PyTorch Transformer through
PSL-provided gradients via `pslpython.deeppsl.model.DeepModel`.

The module is function-first: Python modules contain the reusable logic, while
`scripts/` and notebooks are thin orchestration and visualization layers.

Default notebook configuration:

| Field | Value |
|-------|-------|
| Dataset | `mnist-1` |
| Train size | `10000` MNIST digit images |
| Validation size | `10000` MNIST digit images |
| Test size | `1000` MNIST digit images |
| Overlap | `0.0` |
| Training steps | `50` DeepPSL gradient steps |
| Step checkpoints | Every `10` DeepPSL gradient steps |
| Inference examples | Multiple test pairs, default `6` |

Important behavior:

- No synthetic/mock MNIST-like data remains in the project.
- Training is a single NeuPSL run. Intermediate model inspection uses the
  checkpoint files saved by that run, not extra training jobs.
- PSL runtime configs use H2 instead of SQLite to avoid deep-predicate
  upsert/commit errors.
- Training and inference artifacts for one run are saved under `results/train_<train_size>_infer_<inference_size>/`.

---

## Directory Structure

```text
mnist-add-minreal/
├── README.md
├── requirements.txt
├── setup.py
├── data/
│   ├── __init__.py
│   ├── loader.py
│   ├── generator.py
│   ├── raw/                   # created by torchvision MNIST download
│   └── experiment_mnist-*/    # generated PSL data
├── models/
│   ├── __init__.py
│   ├── transformer.py
│   └── deeppsl.py
├── psl/
│   ├── __init__.py
│   ├── runner.py
│   └── rules/
│       └── experiment__mnist-1.json
├── scripts/
│   ├── __init__.py
│   ├── check_env.py
│   ├── create_data.py
│   ├── pretrain_transformer.py
│   ├── neupsl_train.py
│   └── evaluate.py
├── notebooks/
│   ├── __init__.py
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_demo.ipynb
│   └── 03_inference_demo.ipynb
└── utils/
    ├── __init__.py
    ├── config.py
    ├── io.py
    └── metrics.py
```

---

## Module Reference

### `utils/config.py`

Primary interface:

```python
DatasetConfig(
    name="mnist-1",
    num_digits=1,
    class_size=10,
    pretrain_size=20000,
    pretrain_valid_size=5000,
    pretrain_overlap=0.4,
    train_size=1000,
    valid_size=1000,
    inference_size=5000,
    overlap=0.0,
    seed=None,
)
dataset_dir(dataset, data_root=DEFAULT_DATA_ROOT)
pretrain_dir(config, data_root=DEFAULT_DATA_ROOT)
neupsl_train_dir(config, data_root=DEFAULT_DATA_ROOT)
inference_dir(config, data_root=DEFAULT_DATA_ROOT)
```

Notes:

- `name` is currently restricted to `mnist-1`.
- `num_digits`, `max_number`, and `max_sum` are fixed for single-digit addition.
- `to_metadata_dict()` writes generated-dataset metadata and the
  `"data-source": "mnist"` marker.

### `data/loader.py`

Primary interface:

```python
normalize_images(images) -> np.ndarray
load_mnist_arrays(root=None, download=True) -> tuple[np.ndarray, np.ndarray]
```

Notes:

- Uses `torchvision.datasets.MNIST`.
- Concatenates train and test MNIST into one real-MNIST pool.
- Returns flattened normalized images and integer labels.

### `data/generator.py`

Primary interface:

```python
generate_experiment_datasets(config, features, labels, data_root=DEFAULT_DATA_ROOT) -> dict[str, Path]
```

Supporting functions:

| Function | Purpose |
|----------|---------|
| `digits_to_number()` | Convert digit sequence to integer |
| `digits_to_sum()` | Compute addition label from an image block |
| `generate_split()` | Create single-digit addition image blocks and sum labels |
| `create_entity_data_map()` | Writeable rows for image features and true digit labels |
| `create_image_data()` | Target rows for `NeuralClassifier` |
| `create_image_sum_data()` | Target/truth rows for sum predicates |
| `write_shared_data()` | Symbolic facts shared by all splits |
| `write_partition_data()` | Partition-specific train/valid/inference files |

Generated data includes:

- `entity-data-map.txt`
- `image-target-{partition}.txt`
- `image-sum-block-{partition}.txt`
- `image-sum-target-{partition}.txt`
- `image-sum-truth-{partition}.txt`
- `number-sum.txt`
- `possible-digits.txt`

### `models/transformer.py`

Primary interface:

```python
class MNISTTransformerClassifier(nn.Module):
    def __init__(self, num_classes=10)
    def forward(self, x)
    def logits(self, x)
    def predict(self, x, device="cpu") -> np.ndarray

save_checkpoint(path, model, metadata=None) -> None
load_checkpoint(path, map_location="cpu") -> MNISTTransformerClassifier
```

Notes:

- The classifier is a pixel-token Transformer: 784 scalar pixel tokens are
  embedded to 64 dimensions, processed by 2 encoder layers with 4 attention
  heads, adaptively pooled to 4 tokens and flattened to 256 features, then
  passed through the unchanged `256 -> 120 -> 84 ->
  class_size` output head.
- `forward()` returns softmax probabilities.
- `logits()` is used by the optional supervised helper.
- NeuPSL training updates this model through `MNISTDeepPSL.internal_fit()`.

### `models/deeppsl.py`

Primary interface:

```python
class MNISTDeepPSL(pslpython.deeppsl.model.DeepModel):
    def internal_init_model(self, application, options=None)
    def internal_predict(self, data, options=None)
    def internal_fit(self, data, gradients, options=None)
    def internal_eval(self, data, options=None)
    def internal_save(self, options=None)

MNISTAdditionModel = MNISTDeepPSL
```

Important options:

- `class-size`
- `learning-rate`
- `batch-size`
- `checkpoint-dir`
- `checkpoint-frequency`
- `random-seed`
- `save-path`
- `pretrained-path`
- `device`

Trace rows contain Transformer accuracy, sample probability vector, sample label, and
PSL gradient statistics. Neural forward/backward is batched inside each full-graph
NeuPSL step; PSL still receives full prediction and gradient arrays. The training
notebook reports Transformer vectors and gradient statistics at steps `10`, `20`,
and `30`, then parses final PSL atoms at step `50`.

### `psl/runner.py`

Primary interface:

```python
runner = PSLRunner(jar_path=None, psl_version="2.4.0")
config_path = runner.build_config(...)
payload = runner.run_runtime(config_path, output_dir)
completed = runner.run(config_path, output_dir, extra_options=None)
```

`build_config()` wires generated data files, rule templates, DeepModel options,
H2 database options, and learning/inference settings into a concrete
`psl-config.json`.

`run_runtime()` writes:

- `runtime-output.json`
- `learned-rules.json`
- `learned-rules.txt`
- `out.txt`
- `out.err`

`run()` is retained for Java CLI execution, but notebooks and default scripts
use `run_runtime()`.

### `scripts/`

Scripts are command-line wrappers over the Python interfaces.

| Script | Core role |
|--------|-----------|
| `check_env.py` | Calls `check_environment()` to validate Java and `pslpython.runtime` |
| `create_data.py` | Calls `load_mnist_arrays()` and `generate_experiment_datasets()` |
| `pretrain_transformer.py` | Supervised pretraining for `MNISTTransformerClassifier` on the generated split |
| `neupsl_train.py` | Builds `DatasetConfig`, `PSLRunner.build_config()`, and runs NeuPSL joint fine-tuning/inference |
| `evaluate.py` | Reads runtime output and logs into `metrics.json` and prints a compact summary |

### `notebooks/`

| Notebook | Current role |
|----------|--------------|
| `01_data_exploration.ipynb` | Real MNIST exploration; shows image pairs and generated PSL rows |
| `02_training_demo.ipynb` | One NeuPSL training run; step checkpoints; final DeepPredicate and ImageSum atoms |
| `03_inference_demo.ipynb` | Loads saved artifacts; shows multiple test examples in subplot grids |

`03_inference_demo.ipynb` currently includes two visualization blocks:

- image pair plus fused `ImageSum` soft vector per example,
- Transformer digit probabilities vs DeepPredicate digit atoms per digit.

---

## Data and Result Paths

Generated data:

```text
data/experiment_<dataset>/pretrain/size_<pretrain_size>-valid_<pretrain_valid_size>-overlap_<pretrain_overlap>/
data/experiment_<dataset>/neupsl-train/size_<train_size>-valid_<valid_size>-overlap_<overlap>/
data/experiment_<dataset>/inference/size_<inference_size>/
```

Training results:

```text
results/train_<train_size>_infer_<inference_size>/
```

Current default notebook artifact path:

```text
results/train_1000_infer_5000/
```

Expected result artifacts:

| Artifact | Purpose |
|----------|---------|
| `model.pt` | Trained PyTorch Transformer checkpoint |
| `pretrained-transformer.pt` | Optional supervised Transformer checkpoint used to initialize NeuPSL |
| `runtime-output.json` | PSL output with fused atoms and evaluations |
| `learned-rules.json` | Learned PSL rules |
| `learned-rules.txt` | Text dump of learned rules |
| `checkpoints/step_*.pt` | Step checkpoints used for inspection and resume |
| `psl-config.json` | Concrete runtime config |

Pretrained checkpoints:

```text
ckpt/pretrained-transformer.pt
```

---

## Development Status

| Component | Status | Notes |
|-----------|--------|-------|
| Real MNIST loading | Complete | No synthetic fallback |
| PSL data generation | Complete | Add1 and Add2 file generation |
| PyTorch Transformer | Complete | Transformer-based replaceable classifier |
| Transformer pretraining script | Complete | Supervised split-aligned checkpoint before NeuPSL |
| DeepPSL bridge | Complete | Transformer updated by PSL gradients |
| PSL runner | Complete | Runtime + CLI backends, H2 DB config |
| Training script | Complete | Saves model/rules/runtime/checkpoints |
| Evaluation/parser scripts | Complete | Reads runtime output and text logs |
| Data notebook | Complete | Real images and pair/sum visualization |
| Training notebook | Complete | One run, step checkpoints, final PSL atoms |
| Inference notebook | Complete | Multiple examples and subplot grids |
| README | Complete | Rewritten around Python APIs |

---

## Change Log

| Date | Change |
|------|--------|
| 2026-05-15 | Created minimal project structure and copied PSL rules |
| 2026-05-18 | Implemented initial PyTorch classifier, DeepPSL bridge, runner, data generator, scripts, and notebooks |
| 2026-05-20 | Regenerated full minimal implementation after docs-only state; added environment check and runnable notebooks |
| 2026-05-21 | Removed synthetic data path; standardized real MNIST generation |
| 2026-05-21 | Updated training notebook to one 50-step run with 10/20/30/50 observations |
| 2026-05-21 | Added final Transformer + DeepPredicate digit atoms and `ImageSum` soft vector display |
| 2026-05-21 | Updated inference notebook to show multiple test examples in subplot grids |
| 2026-05-21 | Rewrote README around project function, file structure, core APIs, and final CLI commands |
| 2026-05-21 | Replaced the old CNN naming with Transformer classifier module/class names |
| 2026-05-22 | Added supervised Transformer pretraining and pretrained NeuPSL initialization |

---

## Notes for Future Sessions

- Keep documentation aligned with Python function signatures, not only CLI flags.
- If notebook imports look stale, restart the Jupyter kernel; previous sessions hit
  cached imports after code changes.
- Clear heavy notebook outputs before committing unless rendered output is
  intentionally part of the artifact.
- `pslpython` may print a socket shutdown traceback after successful runs; check
  whether `runtime-output.json`, `learned-rules.json`, and `model.pt` were
  written before treating it as a failed run.
- For inference visualization, run the training notebook or training script first
  so `model.pt`, `runtime-output.json`, and `learned-rules.json` exist.
