import json
from data_loader import load_data
from graph_builder import build_graph
from gnn_model import train_gnn
from rl_agent import RLRanker


PHASE_PREFERENCES = {
    'A': {'weights': {'price': 2, 'school_nearby': 1, 'area': 1}},
    'B': {'weights': {'price': 1, 'school_nearby': 2, 'area': 1}},
    'C': {'weights': {'price': 1, 'school_nearby': 1, 'area': 2}},
}


def run(output_path='dynamic_scores.json'):
    houses, schools, hospitals, prefs, criteria = load_data(
        'updated_houses_with_price_history.json',
        'facilities.geojson',
        'user_preferences.json',
        'updated_criteria.json',
    )
    graph = build_graph(houses, schools, hospitals)
    embeddings = train_gnn(graph, embedding_dim=32, epochs=50)
    for h in houses:
        hid = h['id']
        h['embedding'] = embeddings[hid] if hid < len(embeddings) else [0.0] * 32

    ranker = RLRanker(criteria, prefs, state_dim=32)
    stage_scores = {}
    for stage, pref in PHASE_PREFERENCES.items():
        ranker.update_weights(pref['weights'])
        scores = ranker.train(houses, episodes=20, verbose=True)
        stage_scores[stage] = scores
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stage_scores, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    run()
