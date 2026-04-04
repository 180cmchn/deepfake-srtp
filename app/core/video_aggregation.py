from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import torch


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _longest_positive_run(values: Iterable[float], threshold: float) -> float:
    longest = 0
    current = 0
    values_list = list(values)
    if not values_list:
        return 0.0
    for value in values_list:
        if value >= threshold:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest / float(len(values_list))


def aggregate_probability_sequence(
    frame_probabilities: List[Dict[str, float]],
    *,
    confidence_threshold: float,
    topk_ratio: float,
    mean_weight: float,
    peak_weight: float,
    persistence_weight: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    if not frame_probabilities:
        empty = {"fake": 0.0, "real": 0.0}
        details = {
            "mean_fake_probability": 0.0,
            "topk_fake_probability": 0.0,
            "positive_ratio": 0.0,
            "longest_positive_run_ratio": 0.0,
            "aggregated_fake_probability": 0.0,
        }
        return empty, details

    fake_scores = [
        _clamp_probability(item.get("fake", 0.0)) for item in frame_probabilities
    ]
    mean_fake = sum(fake_scores) / len(fake_scores)
    topk_count = max(
        1,
        min(len(fake_scores), int(math.ceil(len(fake_scores) * max(0.05, topk_ratio)))),
    )
    topk_fake = sum(sorted(fake_scores, reverse=True)[:topk_count]) / float(topk_count)
    positive_ratio = sum(
        score >= confidence_threshold for score in fake_scores
    ) / float(len(fake_scores))
    longest_positive_run_ratio = _longest_positive_run(
        fake_scores, confidence_threshold
    )
    persistence_signal = (positive_ratio + longest_positive_run_ratio) / 2.0
    if positive_ratio == 0.0 and longest_positive_run_ratio == 0.0:
        persistence_signal = mean_fake

    raw_weights = {
        "mean": max(0.0, mean_weight),
        "peak": max(0.0, peak_weight),
        "persistence": max(0.0, persistence_weight),
    }
    total_weight = sum(raw_weights.values()) or 1.0
    normalized_weights = {
        key: value / total_weight for key, value in raw_weights.items()
    }
    aggregated_fake = (
        (normalized_weights["mean"] * mean_fake)
        + (normalized_weights["peak"] * topk_fake)
        + (normalized_weights["persistence"] * persistence_signal)
    )
    aggregated_fake = _clamp_probability(aggregated_fake)
    probabilities = {
        "fake": aggregated_fake,
        "real": _clamp_probability(1.0 - aggregated_fake),
    }
    details = {
        "mean_fake_probability": mean_fake,
        "topk_fake_probability": topk_fake,
        "positive_ratio": positive_ratio,
        "longest_positive_run_ratio": longest_positive_run_ratio,
        "aggregated_fake_probability": aggregated_fake,
    }
    return probabilities, details


def aggregate_logit_sequence(
    logits_list: List[torch.Tensor],
    *,
    confidence_threshold: float,
    topk_ratio: float,
    mean_weight: float,
    peak_weight: float,
    persistence_weight: float,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if not logits_list:
        empty = torch.log(torch.tensor([1e-6, 1.0], dtype=torch.float32, device=device))
        return empty, {
            "mean_fake_probability": 0.0,
            "topk_fake_probability": 0.0,
            "positive_ratio": 0.0,
            "longest_positive_run_ratio": 0.0,
            "aggregated_fake_probability": 0.0,
        }

    probability_sequence: List[Dict[str, float]] = []
    for logits in logits_list:
        probabilities = torch.softmax(logits.float(), dim=0)
        probability_sequence.append(
            {
                "fake": float(probabilities[0].item()),
                "real": float(probabilities[1].item()),
            }
        )

    aggregated_probabilities, details = aggregate_probability_sequence(
        probability_sequence,
        confidence_threshold=confidence_threshold,
        topk_ratio=topk_ratio,
        mean_weight=mean_weight,
        peak_weight=peak_weight,
        persistence_weight=persistence_weight,
    )
    probabilities_tensor = torch.tensor(
        [aggregated_probabilities["fake"], aggregated_probabilities["real"]],
        dtype=torch.float32,
        device=device,
    ).clamp(min=1e-6)
    return torch.log(probabilities_tensor), details
