import os
import json
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from data_loader import load_data
from utils import calculate_score, rank_properties
from evolutionary_optimizer import EvolutionaryOptimizer, optimize_with_hybrid
from rl_agent import RLRanker
from graph_builder import build_graph
from gnn_model import train_gnn


def ndcg_at_k(scores, k):
    dcg = sum((2 ** s - 1) / math.log2(i + 2) for i, s in enumerate(scores[:k]))
    ideal = sorted(scores, reverse=True)
    idcg = sum((2 ** s - 1) / math.log2(i + 2) for i, s in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(ranked, relevant, k):
    hits = 0
    ap = 0.0
    for i, h in enumerate(ranked[:k], 1):
        if h['id'] in relevant:
            hits += 1
            ap += hits / i
    denom = min(k, len(relevant))
    return ap / denom if denom > 0 else 0.0


def evaluate_ranking(ranked, all_houses, criteria, weights, k=10):
    scores = {h['id']: calculate_score(h, criteria, weights) for h in all_houses}
    rel_scores = [scores[h['id']] for h in ranked]
    ideal_top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    relevant = set(hid for hid, _ in ideal_top)
    ndcg = ndcg_at_k(rel_scores, k)
    m_ap = map_at_k(ranked, relevant, k)
    return ndcg, m_ap


def prepare_embeddings(houses, schools, hospitals, state_dim=16):
    graph = build_graph(houses, schools, hospitals)
    embeddings = train_gnn(graph, embedding_dim=state_dim, epochs=50)
    for h in houses:
        hid = h['id']
        h['embedding'] = embeddings[hid] if hid < len(embeddings) else [0.0] * state_dim


def prepare_simple_embeddings(houses):
    for h in houses:
        h['embedding'] = [h.get('price_norm', 0.0), h.get('bedrooms_norm', 0.0),
                          h.get('bathrooms_norm', 0.0), h.get('area_norm', 0.0)]


def pareto_coverage(ref, cand, criteria):
    """Coverage of cand by ref (ratio of solutions in cand dominated by ref)."""
    opt = EvolutionaryOptimizer(criteria)
    if not cand:
        return 0.0
    covered = 0
    for b in cand:
        for a in ref:
            if opt.dominates(a, b) or opt.evaluate_objectives(a) == opt.evaluate_objectives(b):
                covered += 1
                break
    return covered / len(cand)


def attribute_scores(house, criteria):
    keys = ['price', 'bedrooms', 'bathrooms', 'area']
    if 'house_type' in criteria:
        keys.append('house_type')
    if criteria.get('hospital_nearby'):
        keys.append('hospital_nearby')
    if criteria.get('school_nearby'):
        keys.append('school_nearby')
    scores = {}
    for k in keys:
        score = calculate_score(house, {k: criteria.get(k)}, {k: 1.0})
        scores[k] = score
    return scores


def plot_radar(scores, path):
    labels = list(scores.keys())
    values = list(scores.values())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], labels)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_metric_bars(metrics, metric_key, path):
    names = list(metrics.keys())
    vals = [metrics[n][metric_key] for n in names]
    plt.figure()
    plt.bar(names, vals)
    plt.ylabel(metric_key.upper())
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_pareto_3d(front, obj_keys, path):
    if len(obj_keys) < 3 or not front:
        return
    xs = [h.get(obj_keys[0], 0.0) for h in front]
    ys = [h.get(obj_keys[1], 0.0) for h in front]
    zs = [h.get(obj_keys[2], 0.0) for h in front]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs)
    ax.set_xlabel(obj_keys[0])
    ax.set_ylabel(obj_keys[1])
    ax.set_zlabel(obj_keys[2])
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def run_evaluation(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    houses, schools, hospitals, prefs, criteria = load_data(
        'updated_houses_with_price_history.json',
        'facilities.geojson',
        'user_preferences.json',
        'updated_criteria.json'
    )
    weights = prefs.get('weights', {})

    # Weighted baseline
    weighted_ranked = [h for h, _ in rank_properties(houses, criteria, weights)]
    ndcg_w, map_w = evaluate_ranking(weighted_ranked, houses, criteria, weights)

    # RL without GNN
    prepare_simple_embeddings(houses)
    rl_agent = RLRanker(criteria, prefs, state_dim=4)
    rewards_rl = rl_agent.train(houses, episodes=50, verbose=True)
    rl_ranked = rl_agent.rank(houses)
    ndcg_rl, map_rl = evaluate_ranking(rl_ranked, houses, criteria, weights)

    plt.figure()
    plt.plot(rewards_rl)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rl_rewards.png'))

    # NSGA-II
    evo = EvolutionaryOptimizer(criteria)
    nsga_front, hv_hist_nsga = evo.optimize_nsga2(houses, track=True)
    ndcg_nsga, map_nsga = evaluate_ranking(nsga_front, houses, criteria, weights)
    plt.figure()
    plt.plot(hv_hist_nsga)
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'nsga_hv.png'))

    # Hybrid
    hybrid_front, hv_hist_hybrid = optimize_with_hybrid(
        houses, criteria, save_pareto_path=os.path.join(output_dir, 'hybrid_pareto.json'), track=True)
    prepare_embeddings(houses, schools, hospitals)
    hybrid_agent = RLRanker(criteria, prefs, state_dim=16)
    rewards_hybrid = hybrid_agent.train(houses, episodes=50, verbose=True)
    hybrid_ranked = hybrid_agent.rank(houses)
    ndcg_hybrid, map_hybrid = evaluate_ranking(hybrid_ranked, houses, criteria, weights)

    plt.figure()
    plt.plot(hv_hist_hybrid)
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hybrid_hv.png'))

    # Additional metrics
    hv_w = evo.compute_hypervolume(weighted_ranked[:20])
    hv_rl = evo.compute_hypervolume(rl_ranked[:20])
    hv_nsga = evo.compute_hypervolume(nsga_front)
    hv_hybrid = evo.compute_hypervolume(hybrid_front)

    coverage_hybrid = pareto_coverage(hybrid_front, nsga_front, criteria)
    coverage_nsga = pareto_coverage(nsga_front, hybrid_front, criteria)

    plot_pareto_3d(nsga_front, evo.objective_keys,
                   os.path.join(output_dir, 'nsga_pareto3d.png'))
    plot_pareto_3d(hybrid_front, evo.objective_keys,
                   os.path.join(output_dir, 'hybrid_pareto3d.png'))

    for name, ranked in [('weighted', weighted_ranked),
                         ('rl_no_gnn', rl_ranked),
                         ('nsga2', nsga_front),
                         ('hybrid', hybrid_front)]:
        if ranked:
            scores = attribute_scores(ranked[0], criteria)
            plot_radar(scores, os.path.join(output_dir, f'{name}_radar.png'))

    plt.figure()
    plt.plot(rewards_hybrid)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hybrid_rewards.png'))

    metrics = {
        'weighted': {'ndcg': ndcg_w, 'map': map_w, 'hv': hv_w},
        'rl_no_gnn': {'ndcg': ndcg_rl, 'map': map_rl, 'hv': hv_rl},
        'nsga2': {'ndcg': ndcg_nsga, 'map': map_nsga, 'hv': hv_nsga},
        'hybrid': {'ndcg': ndcg_hybrid, 'map': map_hybrid, 'hv': hv_hybrid}
    }
    plot_metric_bars(metrics, 'ndcg', os.path.join(output_dir, 'ndcg_bar.png'))
    plot_metric_bars(metrics, 'map', os.path.join(output_dir, 'map_bar.png'))
    plot_metric_bars(metrics, 'hv', os.path.join(output_dir, 'hv_bar.png'))

    with open(os.path.join(output_dir, 'coverage.json'), 'w', encoding='utf-8') as f:
        json.dump({'hybrid_vs_nsga': coverage_hybrid, 'nsga_vs_hybrid': coverage_nsga}, f, ensure_ascii=False, indent=4)
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Evaluate all methods')
    parser.add_argument('--out', default='eval_plots', help='Output directory')
    args = parser.parse_args()
    run_evaluation(args.out)


if __name__ == '__main__':
    main()
