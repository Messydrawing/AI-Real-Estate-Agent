import json
import math
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic

from data_loader import load_data
from utils import calculate_score
from graph_builder import build_graph
from gnn_model import train_gnn
from rl_agent import RLRanker
from evaluation import prepare_simple_embeddings


def neighborhood_consistency(scored, d=2.0, epsilon=0.1):
    pairs = 0
    consistent = 0
    for i in range(len(scored)):
        hi, si = scored[i]
        if 'coordinates' not in hi or hi['coordinates'] is None:
            continue
        for j in range(i + 1, len(scored)):
            hj, sj = scored[j]
            if 'coordinates' not in hj or hj['coordinates'] is None:
                continue
            dist = geodesic(hi['coordinates'], hj['coordinates']).kilometers
            if dist < d:
                pairs += 1
                if abs(si - sj) < epsilon:
                    consistent += 1
    if pairs == 0:
        return 0.0
    return math.log(consistent + 1) / math.log(pairs + 1)


def spatial_coverage(recommended, m=10, n=10):
    coords = [h['coordinates'] for h in recommended if h.get('coordinates')]
    if not coords:
        return 0.0
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    grid = np.zeros((m, n))
    for h in recommended:
        if not h.get('coordinates'):
            continue
        lat, lon = h['coordinates']
        i = min(int((lat - lat_min) / (lat_max - lat_min + 1e-6) * m), m - 1)
        j = min(int((lon - lon_min) / (lon_max - lon_min + 1e-6) * n), n - 1)
        grid[i, j] += 1
    return np.std(grid)


def boxplot_diffs(b_scores, g_scores, d=2.0):
    b_diffs = []
    g_diffs = []
    for i, (hi, bi) in enumerate(b_scores):
        if not hi.get('coordinates'):
            continue
        for j in range(i + 1, len(b_scores)):
            hj, bj = b_scores[j]
            if not hj.get('coordinates'):
                continue
            dist = geodesic(hi['coordinates'], hj['coordinates']).kilometers
            if dist < d:
                b_diffs.append(abs(bi - bj))
    for i, (hi, gi) in enumerate(g_scores):
        if not hi.get('coordinates'):
            continue
        for j in range(i + 1, len(g_scores)):
            hj, gj = g_scores[j]
            if not hj.get('coordinates'):
                continue
            dist = geodesic(hi['coordinates'], hj['coordinates']).kilometers
            if dist < d:
                g_diffs.append(abs(gi - gj))
    plt.figure()
    plt.boxplot([b_diffs, g_diffs], labels=['Baseline', 'GNN'])
    plt.ylabel('Score diff within cell')
    plt.tight_layout()
    plt.savefig('neighbor_consistency_box.png')
    plt.close()


def plot_heatmap(recommended, m=10, n=10, path='heatmap.png'):
    coords = [h['coordinates'] for h in recommended if h.get('coordinates')]
    if not coords:
        return
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    grid = np.zeros((m, n))
    for h in recommended:
        if not h.get('coordinates'):
            continue
        lat, lon = h['coordinates']
        i = min(int((lat - lat_min) / (lat_max - lat_min + 1e-6) * m), m - 1)
        j = min(int((lon - lon_min) / (lon_max - lon_min + 1e-6) * n), n - 1)
        grid[i, j] += 1
    plt.figure()
    plt.imshow(grid, origin='lower')
    plt.colorbar(label='count')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def map_scatter(b_scores, g_scores, path='map_scatter.png'):
    plt.figure()
    for house, _ in b_scores:
        if house.get('coordinates'):
            plt.scatter(house['coordinates'][1], house['coordinates'][0], c='blue', marker='o')
    for house, _ in g_scores:
        if house.get('coordinates'):
            plt.scatter(house['coordinates'][1], house['coordinates'][0], c='red', marker='x')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def run(output_json='spatial_metrics.json', top_n=50):
    houses, schools, hospitals, prefs, criteria = load_data(
        'updated_houses_with_price_history.json',
        'facilities.geojson',
        'user_preferences.json',
        'updated_criteria.json',
    )

    # Baseline RL without facilities
    prepare_simple_embeddings(houses)
    rl_base = RLRanker(criteria, prefs, state_dim=4)
    rl_base.train(houses, episodes=50)
    base_scores = rl_base.rank_with_scores(houses)[:top_n]

    # GNN model
    graph = build_graph(houses, schools, hospitals)
    embeddings = train_gnn(graph, embedding_dim=32, epochs=50)
    for h in houses:
        hid = h['id']
        h['embedding'] = embeddings[hid] if hid < len(embeddings) else [0.0] * 32
    rl_gnn = RLRanker(criteria, prefs, state_dim=32)
    rl_gnn.train(houses, episodes=50)
    gnn_scores = rl_gnn.rank_with_scores(houses)[:top_n]

    metrics = {
        'baseline_nc': neighborhood_consistency(base_scores),
        'gnn_nc': neighborhood_consistency(gnn_scores),
        'baseline_cov': spatial_coverage([h for h, _ in base_scores]),
        'gnn_cov': spatial_coverage([h for h, _ in gnn_scores])
    }

    map_scatter(base_scores, gnn_scores)
    boxplot_diffs(base_scores, gnn_scores)
    plot_heatmap([h for h, _ in gnn_scores], path='gnn_heatmap.png')
    plot_heatmap([h for h, _ in base_scores], path='baseline_heatmap.png')

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    run()
