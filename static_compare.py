import matplotlib.pyplot as plt

from data_loader import load_data
from evolutionary_optimizer import EvolutionaryOptimizer, optimize_with_hybrid
from rl_agent import RLRanker
from evaluation import evaluate_ranking
from utils import rank_properties


def run(output_path='method_comparison.png'):
    houses, schools, hospitals, prefs, criteria = load_data(
        'updated_houses_with_price_history.json',
        'facilities.geojson',
        'user_preferences.json',
        'updated_criteria.json',
    )
    weights = prefs.get('weights', {})

    # Weighted baseline
    weighted_ranked = [h for h, _ in rank_properties(houses, criteria, weights, top_n=10)]
    optimizer = EvolutionaryOptimizer(criteria)
    ndcg_w, _ = evaluate_ranking(weighted_ranked, houses, criteria, weights)
    hv_w = optimizer.compute_hypervolume(weighted_ranked)

    # Hybrid method with larger search and slower epsilon decay
    pareto_front = optimizer.optimize_hybrid(houses, generations=100, population_size=50)
    prefs_hybrid = dict(prefs)
    prefs_hybrid['epsilon_decay'] = 0.99
    ranker = RLRanker(criteria, prefs_hybrid, state_dim=32)
    ranker.train(pareto_front, episodes=100, verbose=False)
    hybrid_ranked = ranker.rank(pareto_front)[:10]
    ndcg_h, _ = evaluate_ranking(hybrid_ranked, houses, criteria, weights)
    hv_h = optimizer.compute_hypervolume(hybrid_ranked)

    labels = ['Weighted', 'Hybrid']
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].bar(labels, [ndcg_w, ndcg_h])
    axes[0].set_ylabel('NDCG@10')
    axes[1].bar(labels, [hv_w, hv_h])
    axes[1].set_ylabel('Hypervolume')
    plt.tight_layout()
    plt.savefig(output_path)


if __name__ == '__main__':
    run()
