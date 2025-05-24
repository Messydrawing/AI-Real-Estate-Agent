import argparse
import json
from pprint import pprint
import os
import torch

from data_loader import load_data, load_user_preferences, load_criteria
from graph_builder import build_graph
from gnn_model import train_gnn
from evolutionary_optimizer import optimize_with_hybrid
from rl_agent import RLRanker
from utils import parse_natural_language, apply_adjustments, calculate_score


def main():
    parser = argparse.ArgumentParser(description="Train real estate recommender")
    parser.add_argument('--nl', type=str, help='\u81ea\u7136\u8bed\u8a00\u504f\u597d\u63cf\u8ff0')
    parser.add_argument('--threads', type=int, default=8,
                        help='Number of CPU threads to use')
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    if args.nl:
        prefs = load_user_preferences('user_preferences.json')
        crit = load_criteria('updated_criteria.json')
        adj = parse_natural_language(args.nl)
        apply_adjustments(crit, prefs, adj)

    houses, schools, hospitals, preferences, criteria = load_data(
        'updated_houses_with_price_history.json',
        'facilities.geojson',
        'user_preferences.json',
        'updated_criteria.json'
    )
    if not houses:
        print('No houses after filtering.')
        return

    candidate_houses = optimize_with_hybrid(houses, criteria)
    if not candidate_houses:
        candidate_houses = houses

    graph = build_graph(candidate_houses, schools, hospitals)
    embeddings = train_gnn(graph, embedding_dim=16, epochs=50)
    for h in candidate_houses:
        hid = h['id']
        if hid < len(embeddings):
            h['embedding'] = embeddings[hid]
        else:
            h['embedding'] = [0.0] * 16

    ranker = RLRanker(criteria, preferences, state_dim=16)
    ranker.train(candidate_houses, episodes=50)
    ranked = ranker.rank(candidate_houses)

    scored = []
    for h in ranked:
        score = calculate_score(h, criteria, preferences.get("weights", {}))
        scored.append({"address": h.get("address"), "score": score})
    pprint(scored[:10])

if __name__ == '__main__':
    main()

