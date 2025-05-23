# -*- coding: utf-8 -*-
"""Utility functions for computing evaluation metrics and generating plots."""

from __future__ import annotations

import json
from typing import Dict, List, Sequence
import math
import itertools

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np


# --------------------------- Ranking metrics ---------------------------

def dcg(relevances: Sequence[float]) -> float:
    return sum((rel / math.log2(idx + 2) for idx, rel in enumerate(relevances)))


def ndcg(recommended: Sequence[float], ideal: Sequence[float], k: int | None = None) -> float:
    if k is not None:
        recommended = recommended[:k]
        ideal = ideal[:k]
    idcg = dcg(sorted(ideal, reverse=True))
    if idcg == 0:
        return 0.0
    return dcg(recommended) / idcg


def average_precision(recommended: Sequence[int], relevant_set: set[int]) -> float:
    hits = 0
    sum_prec = 0.0
    for i, item in enumerate(recommended, 1):
        if item in relevant_set:
            hits += 1
            sum_prec += hits / i
    if not relevant_set:
        return 0.0
    return sum_prec / len(relevant_set)


# --------------------------- Pareto metrics ---------------------------

def hypervolume(points: List[Sequence[float]], reference: Sequence[float]) -> float:
    arr = np.array(points)
    ref = np.array(reference)
    diff = np.clip(ref - arr, a_min=0, a_max=None)
    volume = np.prod(diff, axis=1)
    return float(np.sum(volume))


def pareto_coverage(test_points: List[Sequence[float]], reference_points: List[Sequence[float]]) -> float:
    ref_set = set(map(tuple, reference_points))
    hits = sum(1 for p in test_points if tuple(p) in ref_set)
    return hits / len(reference_points) if reference_points else 0.0


# --------------------------- Visualization helpers ---------------------------

def plot_ndcg_map(values: Dict[str, Dict[str, float]], path: str) -> None:
    labels = list(values)
    ndcg_vals = [values[l].get("NDCG", 0) for l in labels]
    map_vals = [values[l].get("MAP", 0) for l in labels]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, ndcg_vals, width, label="NDCG")
    plt.bar(x + width / 2, map_vals, width, label="MAP")
    plt.xticks(x, labels)
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)


def plot_pareto_3d(points_dict: Dict[str, List[Sequence[float]]], objectives: List[str], path: str) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for label, pts in points_dict.items():
        pts = np.array(pts)
        if pts.size == 0:
            continue
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], label=label)
    ax.set_xlabel(objectives[0])
    ax.set_ylabel(objectives[1])
    ax.set_zlabel(objectives[2])
    ax.legend()
    plt.tight_layout()
    plt.savefig(path)


def plot_radar(scores: Dict[str, Sequence[float]], labels: List[str], path: str) -> None:
    angles = np.linspace(0, 2 * math.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    plt.figure()
    ax = plt.subplot(111, polar=True)
    for name, vals in scores.items():
        data = list(vals) + [vals[0]]
        ax.plot(angles, data, label=name)
        ax.fill(angles, data, alpha=0.1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()
    plt.savefig(path)


def plot_diversity_box(points_dict: Dict[str, List[Sequence[float]]], objectives: List[str], path: str) -> None:
    data = []
    labels = []
    for name, pts in points_dict.items():
        arr = np.array(pts)
        if arr.size == 0:
            continue
        for idx in range(arr.shape[1]):
            data.append(arr[:, idx])
            labels.append(f"{name}-{objectives[idx]}")
    plt.figure(figsize=(max(6, len(data)), 4))
    plt.boxplot(data, labels=labels, vert=True)
    plt.xticks(rotation=45)
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(path)


def plot_list_diversity(counts_dict: Dict[str, Dict[str, int]], path: str) -> None:
    categories = sorted({c for counts in counts_dict.values() for c in counts})
    x = np.arange(len(counts_dict))
    width = 0.8 / len(categories)
    plt.figure(figsize=(8, 4))
    for i, cat in enumerate(categories):
        vals = [counts_dict[alg].get(cat, 0) for alg in counts_dict]
        plt.bar(x + i * width, vals, width, label=cat)
    plt.xticks(x + width * (len(categories) - 1) / 2, list(counts_dict))
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)


def plot_training_curve(rewards_dict: Dict[str, Sequence[float]], path: str) -> None:
    plt.figure()
    for name, rewards in rewards_dict.items():
        plt.plot(rewards, label=name)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)


def plot_time_cost(time_dict: Dict[str, float], path: str) -> None:
    labels = list(time_dict)
    times = [time_dict[l] for l in labels]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, times)
    plt.ylabel("Seconds")
    plt.tight_layout()
    plt.savefig(path)


if __name__ == "__main__":
    # Example usage with dummy data
    demo_values = {
        "weighted": {"NDCG": 0.7, "MAP": 0.6},
        "evolution": {"NDCG": 0.75, "MAP": 0.65},
    }
    plot_ndcg_map(demo_values, "demo_rank.png")

    demo_points = {
        "alg1": [[1, 2, 3], [2, 1, 4]],
        "alg2": [[1.5, 2.2, 2], [2.5, 1, 3]],
    }
    plot_pareto_3d(demo_points, ["price", "distance", "area"], "demo_pareto3d.png")

    radar_scores = {"alg1": [0.8, 0.6, 0.7], "alg2": [0.9, 0.5, 0.6]}
    plot_radar(radar_scores, ["price", "area", "distance"], "demo_radar.png")

    plot_diversity_box(demo_points, ["price", "distance", "area"], "demo_box.png")

    counts = {"alg1": {"apartment": 3, "villa": 2}, "alg2": {"apartment": 4, "villa": 1}}
    plot_list_diversity(counts, "demo_list_div.png")

    rewards = {"rl": [1, 2, 3, 4], "hybrid": [2, 3, 3, 5]}
    plot_training_curve(rewards, "demo_curve.png")

    times = {"weighted": 0.5, "rl": 3.2, "hybrid": 5.0}
    plot_time_cost(times, "demo_time.png")
