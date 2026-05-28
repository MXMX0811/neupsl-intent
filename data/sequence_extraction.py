from __future__ import annotations

from dataclasses import dataclass
import random
import re
from typing import Iterable


ACTION_NAMES = ("left", "right", "forward", "pickup", "drop", "toggle", "done")
COLORS = ("red", "green", "blue", "purple", "yellow", "grey")
OBJECTS = ("ball", "box", "key", "door")
DIRECTIONS = ("east", "south", "west", "north")
SPECIAL_TOKENS = ("[PAD]", "[CLS]", "[SEP]", "[UNK]")
TYPE_NAMES = (
    "SPECIAL",
    "MISSION",
    "OBS_AGENT",
    "OBS_FRONT",
    "OBS_TARGET",
    "OBS_CARRYING",
    "OBS_WORLD",
    "HISTORY",
)


@dataclass(frozen=True)
class StepExample:
    step_id: int
    episode_id: str
    env_id: str
    step_index: int
    mission: str
    mission_tokens: list[str]
    observation_tokens: list[str]
    observation_types: list[str]
    history_actions: list[str]
    action: str
    features: dict

    def tokens_and_types(self, max_seq_len: int) -> tuple[list[str], list[str]]:
        tokens = ["[CLS]"]
        types = ["SPECIAL"]
        tokens.extend(self.mission_tokens)
        types.extend(["MISSION"] * len(self.mission_tokens))
        tokens.append("[SEP]")
        types.append("SPECIAL")
        tokens.extend(self.observation_tokens)
        types.extend(self.observation_types)
        tokens.append("[SEP]")
        types.append("SPECIAL")
        tokens.extend(self.history_actions)
        types.extend(["HISTORY"] * len(self.history_actions))
        tokens.append("[SEP]")
        types.append("SPECIAL")

        tokens = tokens[:max_seq_len]
        types = types[:max_seq_len]
        if len(tokens) < max_seq_len:
            pad = max_seq_len - len(tokens)
            tokens.extend(["[PAD]"] * pad)
            types.extend(["SPECIAL"] * pad)
        return tokens, types


def build_action_vocab() -> dict[str, int]:
    return {name: index for index, name in enumerate(ACTION_NAMES)}


def build_type_vocab() -> dict[str, int]:
    return {name: index for index, name in enumerate(TYPE_NAMES)}


def build_token_vocab(examples: Iterable[StepExample]) -> dict[str, int]:
    vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for example in examples:
        tokens, _ = example.tokens_and_types(10_000)
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def encode_example(
    example: StepExample,
    token_vocab: dict[str, int],
    type_vocab: dict[str, int],
    action_vocab: dict[str, int],
    max_seq_len: int,
) -> dict:
    tokens, types = example.tokens_and_types(max_seq_len)
    unk_id = token_vocab["[UNK]"]
    pad_id = token_vocab["[PAD]"]
    token_ids = [token_vocab.get(token, unk_id) for token in tokens]
    type_ids = [type_vocab[token_type] for token_type in types]
    if len(token_ids) < max_seq_len:
        token_ids.extend([pad_id] * (max_seq_len - len(token_ids)))
        type_ids.extend([type_vocab["SPECIAL"]] * (max_seq_len - len(type_ids)))
    return {
        "step_id": example.step_id,
        "episode_id": example.episode_id,
        "env_id": example.env_id,
        "step_index": example.step_index,
        "mission": example.mission,
        "tokens": tokens,
        "types": types,
        "token_ids": token_ids[:max_seq_len],
        "type_ids": type_ids[:max_seq_len],
        "action": example.action,
        "action_id": action_vocab[example.action],
        "features": example.features,
    }


def generate_step_examples(
    *,
    env_ids: tuple[str, ...],
    episode_count: int,
    start_step_id: int,
    seed: int,
    min_episode_len: int = 3,
    mission_encoding: str = "surface",
    backend: str = "auto",
) -> list[StepExample]:
    if backend == "minigrid" or (backend == "auto" and _has_minigrid()):
        try:
            return _generate_minigrid_examples(
                env_ids=env_ids,
                episode_count=episode_count,
                start_step_id=start_step_id,
                seed=seed,
                min_episode_len=min_episode_len,
                mission_encoding=mission_encoding,
            )
        except Exception:
            if backend == "minigrid":
                raise
    return _generate_synthetic_examples(
        env_ids=env_ids,
        episode_count=episode_count,
        start_step_id=start_step_id,
        seed=seed,
        min_episode_len=min_episode_len,
        mission_encoding=mission_encoding,
    )


def _generate_synthetic_examples(
    *,
    env_ids: tuple[str, ...],
    episode_count: int,
    start_step_id: int,
    seed: int,
    min_episode_len: int,
    mission_encoding: str,
) -> list[StepExample]:
    rng = random.Random(seed)
    examples: list[StepExample] = []
    next_step_id = start_step_id
    for episode_index in range(episode_count):
        env_id = env_ids[episode_index % len(env_ids)]
        mission_kind = _mission_kind(env_id)
        target_type = _target_type(env_id, rng)
        target_color = rng.choice(COLORS)
        mission = _mission_text(mission_kind, target_type, target_color)
        actions = _expert_actions(mission_kind, rng, min_episode_len)
        episode_id = f"{_slug(env_id)}-{seed}-{episode_index:05d}"

        for step_index, action in enumerate(actions):
            features = _synthetic_features(
                mission_kind=mission_kind,
                target_type=target_type,
                target_color=target_color,
                next_action=action,
                step_index=step_index,
                episode_len=len(actions),
                rng=rng,
            )
            mission_tokens = _mission_tokens(mission, features, mission_encoding)
            observation_tokens, observation_types = _observation_tokens(features)
            examples.append(
                StepExample(
                    step_id=next_step_id,
                    episode_id=episode_id,
                    env_id=env_id,
                    step_index=step_index,
                    mission=mission,
                    mission_tokens=mission_tokens,
                    observation_tokens=observation_tokens,
                    observation_types=observation_types,
                    history_actions=actions[:step_index],
                    action=action,
                    features=features,
                )
            )
            next_step_id += 1
    return examples


def _has_minigrid() -> bool:
    try:
        import gymnasium  # noqa: F401
        import minigrid  # noqa: F401
    except ImportError:
        return False
    return True


def _generate_minigrid_examples(**kwargs) -> list[StepExample]:
    import gymnasium as gym
    import minigrid  # noqa: F401
    from minigrid.utils.baby_ai_bot import BabyAIBot

    env_ids = kwargs["env_ids"]
    episode_count = kwargs["episode_count"]
    start_step_id = kwargs["start_step_id"]
    seed = kwargs["seed"]
    min_episode_len = kwargs["min_episode_len"]
    mission_encoding = kwargs["mission_encoding"]

    examples: list[StepExample] = []
    next_step_id = start_step_id
    generated = 0
    attempts = 0
    max_attempts = max(episode_count * 20, 100)
    while generated < episode_count and attempts < max_attempts:
        env_id = env_ids[generated % len(env_ids)]
        env = gym.make(env_id)
        episode_seed = seed + attempts
        attempts += 1
        try:
            env.reset(seed=episode_seed)
            bot = BabyAIBot(env)
            action_lookup = _action_lookup(env)
            episode_steps = []
            history: list[str] = []
            last_action = None
            terminated = False
            truncated = False
            for step_index in range(int(env.unwrapped.max_steps)):
                features = _minigrid_features(env)
                action_value = bot.replan(last_action)
                action = action_lookup[int(action_value)]
                mission = env.unwrapped.mission
                mission_tokens = _mission_tokens(mission, features, mission_encoding)
                observation_tokens, observation_types = _observation_tokens(features)
                episode_steps.append(
                    (
                        step_index,
                        mission,
                        mission_tokens,
                        observation_tokens,
                        observation_types,
                        list(history),
                        action,
                        features,
                    )
                )
                _, reward, terminated, truncated, _ = env.step(action_value)
                history.append(action)
                last_action = action_value
                if terminated or truncated:
                    break

            if terminated and not truncated and len(episode_steps) >= min_episode_len and reward > 0:
                episode_id = f"{_slug(env_id)}-{episode_seed:05d}"
                for step in episode_steps:
                    step_index, mission, mission_tokens, observation_tokens, observation_types, history_actions, action, features = step
                    examples.append(
                        StepExample(
                            step_id=next_step_id,
                            episode_id=episode_id,
                            env_id=env_id,
                            step_index=step_index,
                            mission=mission,
                            mission_tokens=mission_tokens,
                            observation_tokens=observation_tokens,
                            observation_types=observation_types,
                            history_actions=history_actions,
                            action=action,
                            features=features,
                        )
                    )
                    next_step_id += 1
                generated += 1
        finally:
            env.close()

    if generated < episode_count:
        raise RuntimeError(f"Generated only {generated}/{episode_count} successful BabyAI episodes after {attempts} attempts.")
    return examples


def _action_lookup(env) -> dict[int, str]:
    return {
        int(env.unwrapped.actions.left): "left",
        int(env.unwrapped.actions.right): "right",
        int(env.unwrapped.actions.forward): "forward",
        int(env.unwrapped.actions.pickup): "pickup",
        int(env.unwrapped.actions.drop): "drop",
        int(env.unwrapped.actions.toggle): "toggle",
        int(env.unwrapped.actions.done): "done",
    }


def _minigrid_features(env) -> dict:
    mission_kind, target_type, target_color, target_positions = _target_from_instr(env.unwrapped.instrs)
    front_cell = env.unwrapped.grid.get(*env.unwrapped.front_pos)
    carrying = env.unwrapped.carrying
    relation, distance = _target_relation_distance(env, target_positions)
    front_type = "empty" if front_cell is None else front_cell.type
    front_color = "none" if front_cell is None or front_cell.color is None else front_cell.color
    front_state = "none"
    if front_type == "door":
        if getattr(front_cell, "is_locked", False):
            front_state = "locked"
        elif getattr(front_cell, "is_open", False):
            front_state = "open"
        else:
            front_state = "closed"
    return {
        "agent_dir": DIRECTIONS[int(env.unwrapped.agent_dir)],
        "front_type": front_type,
        "front_color": front_color,
        "front_state": front_state,
        "hand_type": "empty" if carrying is None else carrying.type,
        "hand_color": "none" if carrying is None or carrying.color is None else carrying.color,
        "target_type": target_type,
        "target_color": target_color,
        "target_relation": relation,
        "target_distance": distance,
        "mission_kind": mission_kind,
    }


def _target_from_instr(instr) -> tuple[str, str, str, list[tuple[int, int]]]:
    name = type(instr).__name__.lower()
    if "pickup" in name:
        mission_kind = "pickup"
    elif "open" in name:
        mission_kind = "open"
    else:
        mission_kind = "goto"
    desc = getattr(instr, "desc", None)
    target_type = getattr(desc, "type", "unknown")
    target_color = getattr(desc, "color", "unknown")
    target_positions = []
    obj_poss = getattr(desc, "obj_poss", None)
    if obj_poss:
        target_positions = [tuple(int(value) for value in pos) for pos in obj_poss]
    return mission_kind, target_type, target_color, target_positions


def _target_relation_distance(env, target_positions: list[tuple[int, int]]) -> tuple[str, str]:
    if not target_positions:
        return "unknown", "unknown"
    agent_x, agent_y = env.unwrapped.agent_pos
    target_pos = min(
        target_positions,
        key=lambda pos: abs(pos[0] - int(agent_x)) + abs(pos[1] - int(agent_y)),
    )
    dx = target_pos[0] - int(agent_x)
    dy = target_pos[1] - int(agent_y)
    dist = abs(dx) + abs(dy)
    if dist <= 1:
        distance = "near"
    elif dist <= 4:
        distance = "mid"
    else:
        distance = "far"

    forward = tuple(int(value) for value in env.unwrapped.dir_vec)
    right = tuple(int(value) for value in env.unwrapped.right_vec)
    f_dot = dx * forward[0] + dy * forward[1]
    r_dot = dx * right[0] + dy * right[1]
    if abs(f_dot) >= abs(r_dot):
        relation = "ahead" if f_dot >= 0 else "behind"
    else:
        relation = "right" if r_dot >= 0 else "left"
    return relation, distance


def _mission_kind(env_id: str) -> str:
    lowered = env_id.lower()
    if "open" in lowered:
        return "open"
    if "pickup" in lowered:
        return "pickup"
    return "goto"


def _target_type(env_id: str, rng: random.Random) -> str:
    kind = _mission_kind(env_id)
    if kind == "open":
        return "door"
    if kind == "pickup":
        return rng.choice(("ball", "box", "key"))
    return rng.choice(("ball", "box", "key"))


def _mission_text(kind: str, target_type: str, target_color: str) -> str:
    if kind == "open":
        return f"open the {target_color} door"
    if kind == "pickup":
        return f"pick up the {target_color} {target_type}"
    return f"go to the {target_color} {target_type}"


def _mission_tokens(mission: str, features: dict, mission_encoding: str) -> list[str]:
    if mission_encoding == "structured":
        return ["mission", "kind", features["mission_kind"], "target", features["target_type"], features["target_color"]]
    return re.findall(r"[a-z]+", mission.lower())


def _expert_actions(kind: str, rng: random.Random, min_episode_len: int) -> list[str]:
    approach_len = rng.randint(max(min_episode_len - 1, 2), max(min_episode_len + 3, 5))
    actions = []
    for _ in range(approach_len):
        actions.append(rng.choice(("left", "right", "forward", "forward")))
    if kind == "pickup":
        actions.extend(["pickup", "done"])
    elif kind == "open":
        actions.extend(["toggle", "done"])
    else:
        actions.append("done")
    return actions


def _synthetic_features(
    *,
    mission_kind: str,
    target_type: str,
    target_color: str,
    next_action: str,
    step_index: int,
    episode_len: int,
    rng: random.Random,
) -> dict:
    near_goal = step_index >= episode_len - 2
    front_type = "empty"
    front_color = "none"
    front_state = "none"
    if next_action == "forward":
        front_type = rng.choice(("empty", target_type))
        front_color = target_color if front_type == target_type else "none"
    elif next_action == "pickup":
        front_type = target_type
        front_color = target_color
    elif next_action == "toggle":
        front_type = "door"
        front_color = target_color
        front_state = "closed"
    elif next_action == "done":
        front_type = target_type
        front_color = target_color
        front_state = "open" if mission_kind == "open" else "none"
    elif rng.random() < 0.15:
        front_type = "wall"

    carrying_type = "empty"
    carrying_color = "none"
    if mission_kind == "pickup" and step_index > episode_len - 2:
        carrying_type = target_type
        carrying_color = target_color

    return {
        "agent_dir": rng.choice(DIRECTIONS),
        "front_type": front_type,
        "front_color": front_color,
        "front_state": front_state,
        "hand_type": carrying_type,
        "hand_color": carrying_color,
        "target_type": target_type,
        "target_color": target_color,
        "target_relation": "ahead" if near_goal else rng.choice(("left", "right", "behind", "unknown")),
        "target_distance": "near" if near_goal else rng.choice(("mid", "far")),
        "mission_kind": mission_kind,
    }


def _observation_tokens(features: dict) -> tuple[list[str], list[str]]:
    groups = [
        (["agent", "facing", features["agent_dir"]], "OBS_AGENT"),
        (["front", features["front_type"]], "OBS_FRONT"),
        (["front", "color", features["front_color"]], "OBS_FRONT"),
        (["front", "state", features["front_state"]], "OBS_FRONT"),
        (["hand", features["hand_type"]], "OBS_CARRYING"),
        (["hand", "color", features["hand_color"]], "OBS_CARRYING"),
        (["target", features["target_type"], features["target_color"]], "OBS_TARGET"),
        (["target", "relation", features["target_relation"]], "OBS_TARGET"),
        (["target", "distance", features["target_distance"]], "OBS_TARGET"),
        (["mission", "kind", features["mission_kind"]], "OBS_WORLD"),
    ]
    tokens: list[str] = []
    types: list[str] = []
    for group_tokens, group_type in groups:
        tokens.extend(group_tokens)
        types.extend([group_type] * len(group_tokens))
    return tokens, types


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
