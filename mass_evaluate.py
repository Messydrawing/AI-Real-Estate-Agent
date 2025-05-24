import os
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from data_loader import load_houses, load_facilities, filter_houses
from utils import compute_statistics, normalize_features, rank_properties
from evolutionary_optimizer import EvolutionaryOptimizer, optimize_with_hybrid
from rl_agent import RLRanker
from graph_builder import build_graph
from gnn_model import train_gnn
from evaluation import evaluate_ranking, prepare_simple_embeddings, plot_metric_bars

torch.set_num_threads(8)
os.environ.setdefault("OMP_NUM_THREADS", "8")


def generate_random_prefs(stats, house_types):
    weights = {
        'price': random.random(),
        'bedrooms': random.random(),
        'bathrooms': random.random(),
        'area': random.random(),
        'house_type': random.random(),
        'hospital_nearby': random.random(),
        'school_nearby': random.random()
    }
    criteria = {
        'price': sorted(random.sample([stats['price'][0], stats['price'][1],
                                      random.uniform(stats['price'][0], stats['price'][1])], 2)),
        'bedrooms': sorted(random.sample([stats['bedrooms'][0], stats['bedrooms'][1],
                                         random.uniform(stats['bedrooms'][0], stats['bedrooms'][1])], 2)),
        'bathrooms': sorted(random.sample([stats['bathrooms'][0], stats['bathrooms'][1],
                                          random.uniform(stats['bathrooms'][0], stats['bathrooms'][1])], 2)),
    }
    if 'area' in stats:
        criteria['area'] = sorted(random.sample([stats['area'][0], stats['area'][1],
                                                 random.uniform(stats['area'][0], stats['area'][1])], 2))
    criteria['house_type'] = random.choice(list(house_types)) if house_types else ''
    criteria['hospital_nearby'] = random.choice([True, False])
    criteria['school_nearby'] = random.choice([True, False])
    return weights, criteria


def evaluate_one(all_houses, schools, hospitals, weights, criteria):
    houses = filter_houses(list(all_houses), criteria, schools, hospitals)
    if not houses:
        return None
    stats = compute_statistics(houses)
    normalize_features(houses, stats)

    evo = EvolutionaryOptimizer(criteria)

    weighted_ranked = [h for h, _ in rank_properties(houses, criteria, weights)]
    ndcg_w, map_w = evaluate_ranking(weighted_ranked, houses, criteria, weights)
    hv_w = evo.compute_hypervolume(weighted_ranked[:20])

    prepare_simple_embeddings(houses)
    rl_agent = RLRanker(criteria, {'weights': weights}, state_dim=4)
    rl_agent.train(houses, episodes=50)
    rl_ranked = rl_agent.rank(houses)
    ndcg_rl, map_rl = evaluate_ranking(rl_ranked, houses, criteria, weights)
    hv_rl = evo.compute_hypervolume(rl_ranked[:20])

    nsga_front = evo.optimize_nsga2(houses)
    ndcg_nsga, map_nsga = evaluate_ranking(nsga_front, houses, criteria, weights)
    hv_nsga = evo.compute_hypervolume(nsga_front)

    hybrid_front = optimize_with_hybrid(
        houses, criteria, preferences={'weights': weights}, rl_episodes=30)
    if not hybrid_front:
        hybrid_front = houses
    graph = build_graph(hybrid_front, schools, hospitals)
    embeddings = train_gnn(graph, embedding_dim=32, epochs=50)
    for h in hybrid_front:
        hid = h['id']
        h['embedding'] = embeddings[hid] if hid < len(embeddings) else [0.0] * 32
    ranker = RLRanker(criteria, {'weights': weights}, state_dim=32)
    ranker.train(hybrid_front, episodes=50)
    hybrid_ranked = ranker.rank(hybrid_front)
    ndcg_h, map_h = evaluate_ranking(hybrid_ranked, houses, criteria, weights)
    hv_h = evo.compute_hypervolume(hybrid_front)

    return {
        'weighted': {'ndcg': ndcg_w, 'map': map_w, 'hv': hv_w},
        'rl_no_gnn': {'ndcg': ndcg_rl, 'map': map_rl, 'hv': hv_rl},
        'nsga2': {'ndcg': ndcg_nsga, 'map': map_nsga, 'hv': hv_nsga},
        'hybrid': {'ndcg': ndcg_h, 'map': map_h, 'hv': hv_h}
    }


def main():
    all_houses = load_houses('updated_houses_with_price_history.json')
    schools, hospitals = load_facilities('facilities.geojson')
    stats_all = compute_statistics(all_houses)
    house_types = set(h.get('house_type', '') for h in all_houses if h.get('house_type'))

    results = []
    for _ in range(50):
        weights, criteria = generate_random_prefs(stats_all, house_types)
        metrics = evaluate_one(all_houses, schools, hospitals, weights, criteria)
        if metrics:
            results.append(metrics)

    if not results:
        print('No valid evaluations generated')
        return

    methods = ['weighted', 'rl_no_gnn', 'nsga2', 'hybrid']
    avg = {}
    for m in methods:
        avg[m] = {
            'ndcg': float(np.mean([r[m]['ndcg'] for r in results])),
            'map': float(np.mean([r[m]['map'] for r in results])),
            'hv': float(np.mean([r[m]['hv'] for r in results]))
        }

    with open('random_eval_summary.json', 'w', encoding='utf-8') as f:
        json.dump({'average_metrics': avg, 'samples': len(results)}, f,
                  ensure_ascii=False, indent=4)

    plot_metric_bars(avg, 'ndcg', 'random_ndcg_bar.png')
    plot_metric_bars(avg, 'map', 'random_map_bar.png')
    plot_metric_bars(avg, 'hv', 'random_hv_bar.png')
    print('Saved random_eval_summary.json and metric bar charts')


if __name__ == '__main__':
    main()
