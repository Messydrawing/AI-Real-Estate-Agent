import os
import time
import math
import json
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt

from data_loader import load_data
from graph_builder import build_graph
from gnn_model import train_gnn
from evolutionary_optimizer import EvolutionaryOptimizer, optimize_with_hybrid
from rl_agent import RLRanker
from utils import rank_properties, calculate_score


def ndcg_at_k(relevances, k=10):
    def dcg(vals):
        return sum((2 ** v - 1) / math.log2(i + 2) for i, v in enumerate(vals))
    r = relevances[:k]
    ideal = sorted(relevances, reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(r) / idcg if idcg > 0 else 0.0


def average_precision(rels):
    score = 0.0
    hits = 0
    for idx, r in enumerate(rels, 1):
        if r:
            hits += 1
            score += hits / idx
    return score / hits if hits else 0.0


def map_at_k(relevances, k=10, thresh=70):
    rel = [1 if v >= thresh else 0 for v in relevances[:k]]
    return average_precision(rel)


def prepare_embeddings_no_gnn(houses):
    for h in houses:
        h['embedding'] = [
            h.get('price_norm', 0.0),
            h.get('bedrooms_norm', 0.0),
            h.get('bathrooms_norm', 0.0),
            h.get('area_norm', 0.0),
        ]


def run_weighted(houses, prefs, criteria):
    scored = rank_properties(houses, criteria, prefs.get('weights', {}), top_n=10)
    return [h for h, _ in scored]


def run_rl_no_gnn(houses, prefs, criteria):
    prepare_embeddings_no_gnn(houses)
    ranker = RLRanker(criteria, prefs, state_dim=4)
    rewards = ranker.train(houses, episodes=20)
    ranked = ranker.rank(houses)[:10]
    return ranked, rewards


def run_evolution(houses, criteria, prefs):
    opt = EvolutionaryOptimizer(criteria)
    front, hv_hist = opt.optimize_nsga2(houses, track=True)
    ranked = [h for h, _ in rank_properties(front, criteria, prefs.get('weights', {}), top_n=10)]
    return front, ranked, hv_hist


def run_hybrid(houses, schools, hospitals, prefs, criteria):
    front, hv_hist = optimize_with_hybrid(houses, criteria, track=True)
    if not front:
        front = houses
    graph = build_graph(front, schools, hospitals)
    embeds = train_gnn(graph, embedding_dim=16, epochs=50)
    for h in front:
        hid = h['id']
        h['embedding'] = embeds[hid] if hid < len(embeds) else [0.0] * 16
    ranker = RLRanker(criteria, prefs, state_dim=16)
    rewards = ranker.train(front, episodes=20)
    ranked = ranker.rank(front)[:10]
    return front, ranked, rewards, hv_hist


def ranking_metrics(ranked, all_scores):
    rels = [calculate_score(h, criteria, prefs.get('weights', {})) for h in ranked]
    ndcg = ndcg_at_k(rels, k=len(ranked))
    mp = map_at_k(rels, k=len(ranked))
    return ndcg, mp


def compute_global_front(sets, criteria):
    optimizer = EvolutionaryOptimizer(criteria)
    union = []
    for s in sets:
        union.extend(s)
    return optimizer.find_pareto_front(union)


def coverage(alg_set, global_front):
    g_ids = {h['id'] for h in global_front}
    a_ids = {h['id'] for h in alg_set}
    return len(g_ids & a_ids) / len(g_ids) if g_ids else 0.0


def plot_rewards(rewards_dict, out_path):
    plt.figure()
    for name, rewards in rewards_dict.items():
        plt.plot(range(1, len(rewards) + 1), rewards, label=name)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)


def plot_hv(histories, out_path):
    plt.figure()
    for name, hist in histories.items():
        plt.plot(range(1, len(hist) + 1), hist, label=name)
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)


def main():
    parser = argparse.ArgumentParser(description='Evaluate algorithms')
    parser.add_argument('--out', default='eval_plots', help='output folder')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    houses, schools, hospitals, prefs, criteria = load_data(
        'updated_houses_with_price_history.json',
        'facilities.geojson',
        'user_preferences.json',
        'updated_criteria.json'
    )

    weighted = run_weighted(houses, prefs, criteria)
    rl_ranked, rl_rewards = run_rl_no_gnn(houses, prefs, criteria)
    evo_front, evo_ranked, evo_hv = run_evolution(houses, criteria, prefs)
    hyb_front, hyb_ranked, hyb_rewards, hyb_hv = run_hybrid(houses, schools, hospitals, prefs, criteria)

    # ranking metrics
    all_scores = [calculate_score(h, criteria, prefs.get('weights', {})) for h in houses]
    metrics = {}
    metrics['weighted'] = ranking_metrics(weighted, all_scores)
    metrics['rl'] = ranking_metrics(rl_ranked, all_scores)
    metrics['evolution'] = ranking_metrics(evo_ranked, all_scores)
    metrics['hybrid'] = ranking_metrics(hyb_ranked, all_scores)
    with open(os.path.join(args.out, 'ranking_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    global_front = compute_global_front([weighted, evo_front, rl_ranked, hyb_front], criteria)
    cov = {
        'weighted': coverage(weighted, global_front),
        'rl': coverage(rl_ranked, global_front),
        'evolution': coverage(evo_front, global_front),
        'hybrid': coverage(hyb_front, global_front)
    }
    with open(os.path.join(args.out, 'coverage.json'), 'w', encoding='utf-8') as f:
        json.dump(cov, f, ensure_ascii=False, indent=4)

    plot_rewards({'rl_no_gnn': rl_rewards, 'hybrid_rl': hyb_rewards}, os.path.join(args.out, 'rl_rewards.png'))
    plot_hv({'evolution': evo_hv, 'hybrid': hyb_hv}, os.path.join(args.out, 'hv_curve.png'))


if __name__ == '__main__':
    main()
