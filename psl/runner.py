from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import uuid
from contextlib import redirect_stderr, redirect_stdout

from utils.config import DEFAULT_DATA_ROOT, DatasetConfig, dataset_dir, inference_dir, neupsl_train_dir
from utils.io import ensure_dir, load_json, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RULES_DIR = PROJECT_ROOT / "psl" / "rules"
DEFAULT_PSL_VERSION = "2.4.0"


class PSLRunner:
    def __init__(self, jar_path: str | Path | None = None, psl_version: str = DEFAULT_PSL_VERSION):
        self.psl_version = psl_version
        self.jar_path = Path(jar_path) if jar_path is not None else PROJECT_ROOT / "psl" / f"psl-cli-{psl_version}.jar"

    def ensure_jar(self) -> Path:
        if self.jar_path.exists():
            return self.jar_path
        ensure_dir(self.jar_path.parent)
        url = f"https://repo1.maven.org/maven2/org/linqs/psl-cli/{self.psl_version}/psl-cli-{self.psl_version}.jar"
        print(f"Downloading PSL CLI {self.psl_version} to {self.jar_path}")
        urllib.request.urlretrieve(url, self.jar_path)
        return self.jar_path

    def build_config(
        self,
        config: DatasetConfig,
        *,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        output_path: str | Path | None = None,
        learning_rate: float = 1e-3,
        gradient_steps: int | None = None,
        admm_iterations: int | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_frequency: int = 10,
        batch_size: int = 256,
        log_level: str = "INFO",
        rule_path: str | Path | None = None,
        model_path: str | Path | None = None,
        model_class: str = "BabyAIActionModel",
        random_seed: int | None = None,
        save_path: str | Path | None = None,
        pretrained_path: str | Path | None = None,
    ) -> Path:
        template = load_json(rule_path or (RULES_DIR / f"experiment__{config.name}.json"))
        psl_config = deepcopy(template)
        psl_config["options"]["runtime.log.level"] = log_level

        train_data_path = neupsl_train_dir(config, data_root).resolve()
        inference_data_path = inference_dir(config, data_root).resolve()
        model_path = Path(model_path).resolve() if model_path else (PROJECT_ROOT / "models" / "deeppsl.py").resolve()
        output_path = Path(output_path) if output_path is not None else train_data_path / f"{config.name}.json"
        save_path = Path(save_path).resolve() if save_path else output_path.parent.resolve() / "saved-networks" / "nesy-trained-pt"
        checkpoint_dir = Path(checkpoint_dir).resolve() if checkpoint_dir else output_path.parent.resolve() / "checkpoints"
        entity_data_map_path = output_path.parent.resolve() / "entity-data-map.txt"
        entity_type_map_path = output_path.parent.resolve() / "entity-type-map.txt"
        h2_path = output_path.parent.resolve() / f"psl_h2_{uuid.uuid4().hex}"
        _write_combined_deep_entity_map(
            entity_data_map_path,
            [train_data_path / "entity-data-map.txt", inference_data_path / "entity-data-map.txt"],
            [train_data_path / "entity-type-map.txt", inference_data_path / "entity-type-map.txt"],
        )
        _write_combined_entity_map(entity_type_map_path, [train_data_path / "entity-type-map.txt", inference_data_path / "entity-type-map.txt"])

        token_vocab = load_json(dataset_dir(config.name, data_root) / "token-vocab.json")
        type_vocab = load_json(dataset_dir(config.name, data_root) / "type-vocab.json")

        psl_config["options"].update({
            "runtime.db.type": "H2",
            "runtime.db.h2.inmemory": False,
            "runtime.db.h2.path": str(h2_path),
        })

        neural = psl_config["predicates"]["NeuralAction/2"]
        neural["options"].update({
            "model-path": f"{model_path}::{model_class}",
            "entity-data-map-path": str(entity_data_map_path),
            "entity-type-map-path": str(entity_type_map_path),
            "entity-argument-indexes": "0",
            "save-path": str(save_path),
            "action-size": config.action_size,
            "class-size": config.action_size,
            "vocab-size": len(token_vocab),
            "type-vocab-size": len(type_vocab),
            "max-seq-len": config.max_seq_len,
            "learning-rate": learning_rate,
            "batch-size": batch_size,
            "checkpoint-dir": str(checkpoint_dir),
            "checkpoint-frequency": int(checkpoint_frequency),
        })
        if pretrained_path is not None:
            neural["options"]["pretrained-path"] = str(Path(pretrained_path).resolve())
        if random_seed is not None:
            neural["options"]["random-seed"] = int(random_seed)
        neural["targets"]["learn"] = [str(train_data_path / "action-target-train.txt")]
        neural["targets"]["infer"] = [str(inference_data_path / "action-target-inference.txt")]

        action = psl_config["predicates"]["Action/2"]
        action["targets"]["learn"] = [str(train_data_path / "action-target-train.txt")]
        action["targets"]["infer"] = [str(inference_data_path / "action-target-inference.txt")]
        action["truth"]["learn"] = [str(train_data_path / "action-truth-train.txt")]
        action["truth"]["infer"] = [str(inference_data_path / "action-truth-inference.txt")]

        psl_config["predicates"]["InvalidAction/2"]["observations"]["learn"] = [str(train_data_path / "invalid-action-train.txt")]
        psl_config["predicates"]["InvalidAction/2"]["observations"]["infer"] = [str(inference_data_path / "invalid-action-inference.txt")]
        psl_config["predicates"]["PlausibleAction/2"]["observations"]["learn"] = [str(train_data_path / "plausible-action-train.txt")]
        psl_config["predicates"]["PlausibleAction/2"]["observations"]["infer"] = [str(inference_data_path / "plausible-action-inference.txt")]

        if gradient_steps is not None:
            psl_config["options"]["gradientdescent.numsteps"] = int(gradient_steps)
        if admm_iterations is not None:
            psl_config["learn"]["options"]["admmreasoner.maxiterations"] = int(admm_iterations)
            psl_config["infer"]["options"]["admmreasoner.maxiterations"] = int(admm_iterations)

        write_json(output_path, psl_config, indent=2)
        return output_path

    def run_runtime(self, config_path: str | Path, output_dir: str | Path) -> dict:
        output_dir = ensure_dir(output_dir)
        config = load_json(config_path)
        _prepend_python_to_path()

        import pslpython.runtime

        with (output_dir / "out.txt").open("w", encoding="utf-8") as out_file, (output_dir / "out.err").open("w", encoding="utf-8") as err_file:
            with redirect_stdout(out_file), redirect_stderr(err_file):
                result = pslpython.runtime.run(config, base_path=str(PROJECT_ROOT))
        write_json(output_dir / "runtime-output.json", result)
        learned_rules = result.get("rules", [])
        if learned_rules:
            write_json(output_dir / "learned-rules.json", learned_rules)
            (output_dir / "learned-rules.txt").write_text(
                "\n".join(rule if isinstance(rule, str) else json.dumps(rule, sort_keys=True) for rule in learned_rules) + "\n",
                encoding="utf-8",
            )
        return result

    def run(self, config_path: str | Path, output_dir: str | Path, extra_options: list[str] | None = None) -> subprocess.CompletedProcess:
        if shutil.which("java") is None:
            raise RuntimeError("java was not found in PATH; Java is required to run PSL.")
        jar_path = self.ensure_jar()
        output_dir = ensure_dir(output_dir)
        command = ["java", "-jar", str(jar_path), "--config", str(config_path), "--output", str(output_dir)]
        if extra_options:
            command.extend(extra_options)
        env = os.environ.copy()
        env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=True, text=True, capture_output=True, env=env)
        (output_dir / "out.txt").write_text(completed.stdout, encoding="utf-8")
        (output_dir / "out.err").write_text(completed.stderr, encoding="utf-8")
        return completed


def run_psl_experiment(config_path: str | Path, output_dir: str | Path, extra_options: list[str] | None = None):
    runner = PSLRunner()
    return runner, runner.run(config_path, output_dir, extra_options)


def _write_combined_entity_map(output_path: Path, input_paths: list[Path]) -> None:
    ensure_dir(output_path.parent)
    rows_by_entity = {}
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Missing entity map: {input_path}")
        for line in input_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows_by_entity.setdefault(line.split("\t", 1)[0], line)
    output_path.write_text("\n".join(rows_by_entity[key] for key in sorted(rows_by_entity, key=int)) + "\n", encoding="utf-8")


def _write_combined_deep_entity_map(output_path: Path, data_paths: list[Path], type_paths: list[Path]) -> None:
    ensure_dir(output_path.parent)
    data_by_entity = {}
    type_by_entity = {}
    for input_path in data_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Missing entity data map: {input_path}")
        for line in input_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            values = line.split("\t")
            data_by_entity.setdefault(values[0], values)
    for input_path in type_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Missing entity type map: {input_path}")
        for line in input_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            values = line.split("\t")
            type_by_entity.setdefault(values[0], values)

    rows = []
    for entity_id in sorted(data_by_entity, key=int):
        data_row = data_by_entity[entity_id]
        type_row = type_by_entity[entity_id]
        rows.append("\t".join([entity_id, *data_row[1:-1], *type_row[1:], data_row[-1]]))
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _prepend_python_to_path() -> None:
    python_dir = str(Path(sys.executable).parent)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if python_dir not in path_parts:
        os.environ["PATH"] = os.pathsep.join([python_dir, *path_parts])
