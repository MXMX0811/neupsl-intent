from __future__ import annotations

import numpy as np


def one_hot(labels, num_classes: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    encoded = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


def categorical_accuracy(y_pred, y_true) -> float:
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)

    if len(y_true) == 0:
        return 0.0

    return float(np.mean(np.argmax(y_pred, axis=1) == y_true))


def digit_pair_sum_accuracy(digit_probs, entity_ids, sum_truth_rows, num_digits: int) -> float:
    predictions = {int(entity_id): int(np.argmax(probs)) for entity_id, probs in zip(entity_ids, digit_probs)}
    correct = 0
    total = 0

    for row in sum_truth_rows:
        *image_ids, sum_value, truth = [int(float(value)) for value in row]
        if truth != 1:
            continue

        left = _digits_to_number([predictions[image_id] for image_id in image_ids[:num_digits]])
        right = _digits_to_number([predictions[image_id] for image_id in image_ids[num_digits:]])
        correct += int(left + right == sum_value)
        total += 1

    return 0.0 if total == 0 else correct / total


def _digits_to_number(digits) -> int:
    value = 0
    for digit in digits:
        value = value * 10 + int(digit)
    return value
