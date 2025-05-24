import argparse
import json
from data_loader import load_user_preferences, load_criteria, load_data
from utils import parse_natural_language, apply_adjustments, calculate_score
from evolutionary_optimizer import optimize_with_hybrid
from graph_builder import build_graph
from gnn_model import train_gnn
from rl_agent import RLRanker


def run_pipeline():
    houses, schools, hospitals, prefs, criteria = load_data(
        'updated_houses_with_price_history.json',
        'facilities.geojson',
        'user_preferences.json',
        'updated_criteria.json'
    )
    candidates = optimize_with_hybrid(houses, criteria)
    if not candidates:
        candidates = houses
    graph = build_graph(candidates, schools, hospitals)
    embeddings = train_gnn(graph, embedding_dim=32, epochs=100)
    for h in candidates:
        hid = h['id']
        h['embedding'] = embeddings[hid] if hid < len(embeddings) else [0.0] * 16
    ranker = RLRanker(criteria, prefs, state_dim=32)
    ranker.train(candidates, episodes=100, verbose=True)
    ranked = ranker.rank(candidates)[:5]
    for h in ranked:
        score = calculate_score(h, criteria, prefs.get('weights', {}))
        print(h.get('address'), score)


def main():
    parser = argparse.ArgumentParser(description='Interactive preference update')
    parser.add_argument('--file', default='adjust_order.json',
                        help='JSON file containing natural language description or adjustment dict')
    parser.add_argument('--nl', type=str, help='Natural language text directly')
    args = parser.parse_args()

    if args.nl:
        text = args.nl
        adjust = parse_natural_language(text)
    else:
        with open(args.file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'text' in data:
                text = data['text']
                adjust = parse_natural_language(text)
            else:
                adjust = data

    prefs = load_user_preferences('user_preferences.json')
    crit = load_criteria('updated_criteria.json')
    apply_adjustments(crit, prefs, adjust)

    run_pipeline()


if __name__ == '__main__':
    main()
