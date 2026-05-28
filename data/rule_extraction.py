from __future__ import annotations

from collections import Counter
from pathlib import Path

from data.sequence_extraction import ACTION_NAMES, StepExample
from utils.io import ensure_dir, write_json, write_psl


BASE_PSL_RULES = [
    "1.0: NeuralAction(S, A) -> Action(S, A) ^2",
    "1.0: InvalidAction(S, A) -> ~Action(S, A) ^2",
    "1.0: PlausibleAction(S, A) -> Action(S, A) ^2",
    "Action(S, +A) = 1 .",
]


def derive_action_facts(example: StepExample) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    features = example.features
    invalid = set()
    plausible = set()

    if features["front_type"] == "wall":
        invalid.add("forward")
    if features["front_type"] != "door":
        invalid.add("toggle")
    if features["front_type"] == "door" and features["front_state"] == "closed":
        plausible.add("toggle")
    if features["front_type"] not in {"ball", "box", "key"} or features["hand_type"] != "empty":
        invalid.add("pickup")
    if features["mission_kind"] == "pickup" and features["front_type"] == features["target_type"]:
        plausible.add("pickup")
    if features["mission_kind"] == "open" and features["front_type"] == "door" and features["front_state"] == "closed":
        plausible.add("toggle")
    if features["target_distance"] == "near" and features["mission_kind"] == "goto":
        plausible.add("done")
    if features["mission_kind"] == "pickup" and features["hand_type"] == features["target_type"]:
        plausible.add("done")
    if features["mission_kind"] == "open" and features["front_state"] == "open":
        plausible.add("done")
    if features["hand_type"] == "empty":
        invalid.add("drop")

    invalid -= plausible
    invalid_rows = [(example.step_id, action) for action in sorted(invalid) if action in ACTION_NAMES]
    plausible_rows = [(example.step_id, action) for action in sorted(plausible) if action in ACTION_NAMES]
    return invalid_rows, plausible_rows


def write_rule_observations(out_dir: str | Path, partition: str, examples: list[StepExample]) -> dict[str, int]:
    invalid_rows = []
    plausible_rows = []
    for example in examples:
        invalid, plausible = derive_action_facts(example)
        invalid_rows.extend(invalid)
        plausible_rows.extend(plausible)

    out_dir = ensure_dir(out_dir)
    write_psl(out_dir / f"invalid-action-{partition}.txt", invalid_rows)
    write_psl(out_dir / f"plausible-action-{partition}.txt", plausible_rows)
    return {"invalid": len(invalid_rows), "plausible": len(plausible_rows)}


def write_split_rule_library(path: str | Path, examples: list[StepExample], *, inherited_rules: list[dict] | None = None) -> list[dict]:
    inherited_rules = inherited_rules or []
    counter = Counter()
    for example in examples:
        invalid, plausible = derive_action_facts(example)
        for _, action in invalid:
            counter[f"invalid::{action}"] += 1
        for _, action in plausible:
            counter[f"plausible::{action}"] += 1

    rules = list(inherited_rules)
    for name, support in counter.most_common():
        kind, action = name.split("::", 1)
        rules.append(
            {
                "name": f"{kind}-{action}",
                "source": "derived-fact",
                "action": action,
                "support": support,
                "template": "InvalidAction(S,A) -> ~Action(S,A)" if kind == "invalid" else "PlausibleAction(S,A) -> Action(S,A)",
            }
        )
    path = Path(path)
    ensure_dir(path)
    write_json(path / "rule-library.json", rules)
    (path / "psl-rules.txt").write_text("\n".join(BASE_PSL_RULES) + "\n", encoding="utf-8")
    return rules
