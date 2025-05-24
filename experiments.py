import argparse
from data_loader import load_data
from graph_builder import build_graph
from gnn_model import train_gnn
from evolutionary_optimizer import EvolutionaryOptimizer, optimize_with_hybrid
from rl_agent import RLRanker
from utils import rank_properties, calculate_score


def prepare_embeddings(houses, schools, hospitals, state_dim=32):
    graph = build_graph(houses, schools, hospitals)
    embeddings = train_gnn(graph, embedding_dim=state_dim, epochs=100)
    for h in houses:
        hid = h['id']
        if hid < len(embeddings):
            h['embedding'] = embeddings[hid]
        else:
            h['embedding'] = [0.0] * state_dim


def weighted_only():
    houses, schools, hospitals, prefs, criteria = load_data(
        'updated_houses_with_price_history.json',
        'facilities.geojson',
        'user_preferences.json',
        'updated_criteria.json'
    )
    scored = rank_properties(houses, criteria, prefs.get('weights', {}), top_n=10)
    print('Top results (weighted sort):')
    for h, s in scored:
        print(h.get('address'), s)


def evolution_only():
    houses, schools, hospitals, prefs, criteria = load_data(
        'updated_houses_with_price_history.json',
        'facilities.geojson',
        'user_preferences.json',
        'updated_criteria.json'
    )
    optimizer = EvolutionaryOptimizer(criteria)
    front = optimizer.optimize_nsga2(houses)
    scored = rank_properties(front, criteria, prefs.get('weights', {}), top_n=10)
    optimizer.export_front_points(front, 'evolution_pareto.json')
    print('Top results (evolution only):')
    for h, s in scored:
        print(h.get('address'), s)


def rl_only():
    houses, schools, hospitals, prefs, criteria = load_data(
        'updated_houses_with_price_history.json',
        'facilities.geojson',
        'user_preferences.json',
        'updated_criteria.json'
    )
    prepare_embeddings(houses, schools, hospitals)
    ranker = RLRanker(criteria, prefs, state_dim=32)
    ranker.train(houses, episodes=100, verbose=True)
    ranked = ranker.rank(houses)[:10]
    print('Top results (RL only):')
    for h in ranked:
        score = calculate_score(h, criteria, prefs.get('weights', {}))
        print(h.get('address'), score)


def hybrid():
    houses, schools, hospitals, prefs, criteria = load_data(
        'updated_houses_with_price_history.json',
        'facilities.geojson',
        'user_preferences.json',
        'updated_criteria.json'
    )
    candidates = optimize_with_hybrid(houses, criteria, save_pareto_path='hybrid_pareto.json')
    if not candidates:
        candidates = houses
    prepare_embeddings(candidates, schools, hospitals)
    ranker = RLRanker(criteria, prefs, state_dim=32)
    ranker.train(candidates, episodes=100, verbose=True)
    ranked = ranker.rank(candidates)[:10]
    print('Top results (hybrid):')
    for h in ranked:
        score = calculate_score(h, criteria, prefs.get('weights', {}))
        print(h.get('address'), score)


def main():
    parser = argparse.ArgumentParser(description='Run comparison experiments')
    parser.add_argument('mode', choices=['weighted', 'evolution', 'rl', 'hybrid'])
    args = parser.parse_args()

    if args.mode == 'weighted':
        weighted_only()
    elif args.mode == 'evolution':
        evolution_only()
    elif args.mode == 'rl':
        rl_only()
    else:
        hybrid()


if __name__ == '__main__':
    main()
