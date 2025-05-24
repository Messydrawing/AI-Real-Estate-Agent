import random
import json

from utils import rank_properties

import matplotlib.pyplot as plt



class EvolutionaryOptimizer:
    """多目标进化优化，用于逼近帕累托前沿"""

    def __init__(self, criteria: dict):
        self.criteria = criteria
        self.objective_keys = ['price', 'bedrooms', 'bathrooms', 'area']
        if criteria.get('school_nearby', False):
            self.objective_keys.append('distance_to_nearest_school')
        if criteria.get('hospital_nearby', False):
            self.objective_keys.append('distance_to_nearest_hospital')

    def evaluate_objectives(self, house: dict) -> list:
        values = []
        for key in self.objective_keys:
            if house.get(key) is None:
                values.append(float('inf'))
                continue
            val = house[key]
            if key in ['bedrooms', 'bathrooms', 'area']:
                values.append(-float(val))
            else:
                values.append(float(val))
        return values

    def dominates(self, a: dict, b: dict) -> bool:
        va = self.evaluate_objectives(a)
        vb = self.evaluate_objectives(b)
        better_or_equal = all(x <= y for x, y in zip(va, vb))
        strictly_better = any(x < y for x, y in zip(va, vb))
        return better_or_equal and strictly_better

    def find_pareto_front(self, houses: list) -> list:
        front = []
        for h in houses:
            dominated = False
            for other in houses:
                if other is h:
                    continue
                if self.dominates(other, h):
                    dominated = True
                    break
            if not dominated:
                front.append(h)
        return front

    def compute_hypervolume(self, houses: list) -> float:
        """简单计算价格-面积二维下的超体积 (越大越好)"""
        points = []
        for h in houses:
            if h.get('price') is None or h.get('area') is None:
                continue
            points.append((float(h['price']), -float(h['area'])))
        if not points:
            return 0.0
        ref_price = max(p[0] for p in points) * 1.1
        ref_area = min(p[1] for p in points) * 1.1
        pts = sorted(points, key=lambda x: x[0])
        hv = 0.0
        prev_y = ref_area
        for x, y in pts:
            hv += max(0.0, ref_price - x) * max(0.0, prev_y - y)
            prev_y = y
        return hv

    def optimize_nsga2(self, houses: list, generations: int = 50, population_size: int = 20,
                        track: bool = False):
        """运行简化的 NSGA-II, 可选记录每代的超体积
        打印每一代的进度信息
        """
        if not houses:
            return [] if not track else ([], [])
        population = random.sample(houses, min(population_size, len(houses)))
        history = []
        for gen in range(generations):
            new_pop = population.copy()
            pareto = self.find_pareto_front(population)
            while len(new_pop) < population_size:
                new_pop.append(random.choice(pareto))
            if random.random() < 0.3:
                remaining = [h for h in houses if h not in new_pop]
                if remaining:
                    new_pop[random.randrange(len(new_pop))] = random.choice(remaining)
            population = new_pop
            if track:
                history.append(self.compute_hypervolume(population))
            print(f'NSGA-II generation {gen + 1}/{generations} completed')
        final_front = []
        seen = set()
        for h in self.find_pareto_front(population):
            hid = h.get('id')
            if hid not in seen:
                final_front.append(h)
                seen.add(hid)
        if track:
            return final_front, history
        return final_front

    def optimize_moead(self, houses: list, population_size: int = 20) -> list:
        if not houses:
            return []
        dim = len(self.objective_keys)
        solutions = []
        for _ in range(population_size):
            weights = [random.random() for _ in range(dim)]
            total = sum(weights)
            weights = [w / total for w in weights] if total > 0 else [1.0 / dim] * dim
            best_house = None
            best_val = float('inf')
            for h in houses:
                obj_vals = self.evaluate_objectives(h)
                val = sum(w * v for w, v in zip(weights, obj_vals))
                if val < best_val:
                    best_val = val
                    best_house = h
            if best_house:
                solutions.append(best_house)
        return self.find_pareto_front(solutions)

    def optimize_hybrid(self, houses: list, generations: int = 50, population_size: int = 20,
                        track: bool = False):
        """NSGA-II 与 MOEA/D 协同优化，track=True 时返回超体积历史
        在两个阶段之间打印进度信息
        """
        print('Hybrid optimization: running NSGA-II stage')
        if track:
            nsga_front, history = self.optimize_nsga2(
                houses, generations, population_size, track=True)
        else:
            nsga_front = self.optimize_nsga2(houses, generations, population_size)
            history = []
        print('Hybrid optimization: running MOEA/D stage')
        moead = self.optimize_moead(houses, population_size)
        final_front = self.find_pareto_front(nsga_front + moead)
        return (final_front, history) if track else final_front

    def export_front_points(self, front: list, path: str = 'pareto_points.json', plot: bool = True) -> list:
        """保存帕累托前沿解集各目标取值，并可绘制前两目标散点图"""
        points = []
        for h in front:
            pt = []
            for k in self.objective_keys:
                pt.append(h.get(k, float('inf')))
            points.append(pt)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'objectives': self.objective_keys, 'points': points}, f, ensure_ascii=False, indent=4)
        if plot and len(self.objective_keys) >= 2 and points:
            x, y = zip(*[(p[0], p[1]) for p in points])
            plt.figure()
            plt.scatter(x, y)
            plt.xlabel(self.objective_keys[0])
            plt.ylabel(self.objective_keys[1])
            plt.tight_layout()
            img_path = path.replace('.json', '.png')
            plt.savefig(img_path)
        return points


def optimize_with_hybrid(houses: list, criteria: dict, generations: int = 50, population_size: int = 20,
                         save_pareto_path: str | None = None, track: bool = False):
    """运行混合进化算法，可选地保存帕累托前沿数据/图并返回超体积历史"""
    optimizer = EvolutionaryOptimizer(criteria)
    result = optimizer.optimize_hybrid(houses, generations, population_size, track=track)
    if track:
        front, history = result
    else:
        front, history = result, []
    if save_pareto_path:
        optimizer.export_front_points(front, save_pareto_path)
    return (front, history) if track else front


def rank_by_pareto_layers(houses: list, criteria: dict, weights: dict, top_n: int = 10) -> list:
    """按照帕累托层级进行排序, 每层内部按加权得分排序"""
    optimizer = EvolutionaryOptimizer(criteria)
    remaining = houses.copy()
    ranked_list = []
    while remaining and len(ranked_list) < top_n:
        front = optimizer.find_pareto_front(remaining)
        scored_front = rank_properties(front, criteria, weights)
        for house, _ in scored_front:
            ranked_list.append(house)
            if len(ranked_list) >= top_n:
                break
        remaining = [h for h in remaining if h not in front]
    return ranked_list

