import json
from pprint import pprint

from data_loader import load_data
from graph_builder import build_graph
from gnn_model import train_gnn
from evolutionary_optimizer import optimize_with_hybrid
from rl_agent import RLRanker


def main():
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

