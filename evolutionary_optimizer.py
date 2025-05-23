import random


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

    def optimize_nsga2(self, houses: list, generations: int = 50, population_size: int = 20) -> list:
        if not houses:
            return []
        population = random.sample(houses, min(population_size, len(houses)))
        for _ in range(generations):
            new_pop = population.copy()
            pareto = self.find_pareto_front(population)
            while len(new_pop) < population_size:
                new_pop.append(random.choice(pareto))
            if random.random() < 0.3:
                remaining = [h for h in houses if h not in new_pop]
                if remaining:
                    new_pop[random.randrange(len(new_pop))] = random.choice(remaining)
            population = new_pop
        final_front = []
        seen = set()
        for h in self.find_pareto_front(population):
            hid = h.get('id')
            if hid not in seen:
                final_front.append(h)
                seen.add(hid)
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

    def optimize_hybrid(self, houses: list, generations: int = 50, population_size: int = 20) -> list:
        nsga = self.optimize_nsga2(houses, generations, population_size)
        moead = self.optimize_moead(houses, population_size)
        return self.find_pareto_front(nsga + moead)


def optimize_with_hybrid(houses: list, criteria: dict, generations: int = 50, population_size: int = 20) -> list:
    optimizer = EvolutionaryOptimizer(criteria)
    return optimizer.optimize_hybrid(houses, generations, population_size)

