import os
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from data_loader import load_houses, load_facilities, filter_houses
from utils import compute_statistics, normalize_features, rank_properties, calculate_score
from evolutionary_optimizer import optimize_with_hybrid
from rl_agent import RLRanker
from graph_builder import build_graph
from gnn_model import train_gnn
from evaluation import evaluate_ranking

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

    weighted_ranked = [h for h, _ in rank_properties(houses, criteria, weights)]
    ndcg_w, map_w = evaluate_ranking(weighted_ranked, houses, criteria, weights)

    hybrid_front = optimize_with_hybrid(houses, criteria)
    if not hybrid_front:
        hybrid_front = houses
    graph = build_graph(hybrid_front, schools, hospitals)
    embeddings = train_gnn(graph, embedding_dim=16, epochs=20)
    for h in hybrid_front:
        hid = h['id']
        h['embedding'] = embeddings[hid] if hid < len(embeddings) else [0.0] * 16
    ranker = RLRanker(criteria, {'weights': weights}, state_dim=16)
    ranker.train(hybrid_front, episodes=20)
    hybrid_ranked = ranker.rank(hybrid_front)
    ndcg_h, map_h = evaluate_ranking(hybrid_ranked, houses, criteria, weights)
    return {'weighted': {'ndcg': ndcg_w, 'map': map_w},
            'hybrid': {'ndcg': ndcg_h, 'map': map_h}}


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

    avg = {
        'weighted': {
            'ndcg': float(np.mean([r['weighted']['ndcg'] for r in results])),
            'map': float(np.mean([r['weighted']['map'] for r in results]))
        },
        'hybrid': {
            'ndcg': float(np.mean([r['hybrid']['ndcg'] for r in results])),
            'map': float(np.mean([r['hybrid']['map'] for r in results]))
        }
    }

    with open('random_eval_summary.json', 'w', encoding='utf-8') as f:
        json.dump({'average_metrics': avg, 'samples': len(results)}, f,
                  ensure_ascii=False, indent=4)

    labels = ['weighted', 'hybrid']
    ndcg_vals = [avg['weighted']['ndcg'], avg['hybrid']['ndcg']]
    map_vals = [avg['weighted']['map'], avg['hybrid']['map']]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure()
    plt.bar(x - width/2, ndcg_vals, width, label='NDCG')
    plt.bar(x + width/2, map_vals, width, label='MAP')
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig('random_eval_bar.png')
    print('Saved random_eval_summary.json and random_eval_bar.png')


if __name__ == '__main__':
    main()
