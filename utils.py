import json
import re


def compute_statistics(houses: list) -> dict:
    stats = {}
    if not houses:
        return stats
    numeric_fields = ['price', 'bedrooms', 'bathrooms', 'area']
    for field in numeric_fields:
        values = []
        for h in houses:
            raw = h.get(field)
            if raw is None:
                continue
            try:
                num = float(raw)
                if field in ('bedrooms', 'bathrooms'):
                    num = int(num)
                h[field] = num
                values.append(num)
            except (ValueError, TypeError):
                continue
        if values:
            stats[field] = (min(values), max(values))
    return stats


def normalize_features(houses: list, stats: dict) -> None:
    for h in houses:
        for field, (min_val, max_val) in stats.items():
            val = h.get(field)
            if val is None:
                h[f'{field}_norm'] = 0.0
            else:
                if max_val > min_val:
                    h[f'{field}_norm'] = (val - min_val) / (max_val - min_val)
                else:
                    h[f'{field}_norm'] = 0.0


def parse_natural_language(text: str) -> dict:
    text = text.lower()
    adjust = {
        'price': {'weight': 1, 'size': 1},
        'bedrooms': {'weight': 1, 'size': 1},
        'bathrooms': {'weight': 1, 'size': 1},
        'area': {'weight': 1, 'size': 1},
        'house_type': {'weight': 1},
        'hospital_nearby': {'weight': 1},
        'school_nearby': {'weight': 1}
    }
    if '价格' in text or '预算' in text:
        if any(w in text for w in ['低', '便宜', '降']):
            adjust['price']['weight'] = 0
            adjust['price']['size'] = 0
        elif any(w in text for w in ['高', '贵', '增加']):
            adjust['price']['weight'] = 2
            adjust['price']['size'] = 2
    if '卧室' in text:
        if any(w in text for w in ['多', '增']):
            adjust['bedrooms']['weight'] = 2
            adjust['bedrooms']['size'] = 2
        elif any(w in text for w in ['少']):
            adjust['bedrooms']['weight'] = 0
            adjust['bedrooms']['size'] = 0
    family_match = re.search(r"(\d+)口人", text)
    if family_match:
        adjust['bedrooms']['weight'] = 2
        adjust['bedrooms']['size'] = 2 if int(family_match.group(1)) > 0 else 1
    if '卫生间' in text or '浴室' in text:
        if any(w in text for w in ['多', '增']):
            adjust['bathrooms']['weight'] = 2
            adjust['bathrooms']['size'] = 2
        elif any(w in text for w in ['少']):
            adjust['bathrooms']['weight'] = 0
            adjust['bathrooms']['size'] = 0
    if '面积' in text or '大小' in text:
        if '大' in text:
            adjust['area']['weight'] = 2
            adjust['area']['size'] = 2
        elif '小' in text:
            adjust['area']['weight'] = 0
            adjust['area']['size'] = 0
    if '别墅' in text:
        adjust['house_type']['weight'] = 2
    if '公寓' in text:
        adjust['house_type']['weight'] = 2
    if '学校' in text or '学区' in text:
        adjust['school_nearby']['weight'] = 2
    if '医院' in text or '医疗' in text:
        adjust['hospital_nearby']['weight'] = 2
    return adjust


def apply_adjustments(criteria: dict, preferences: dict, adjust: dict, save: bool = True) -> None:
    weights = preferences.get('weights', {})
    for key, adj in adjust.items():
        if key in weights and 'weight' in adj:
            if adj['weight'] == 2:
                weights[key] = weights.get(key, 1.0) * 1.25
            elif adj['weight'] == 0:
                weights[key] = weights.get(key, 1.0) * 0.75
        if key in criteria:
            if isinstance(criteria[key], list) and 'size' in adj:
                mi, ma = criteria[key]
                if adj['size'] == 2:
                    criteria[key] = [mi * 1.2, ma * 1.2]
                elif adj['size'] == 0:
                    criteria[key] = [mi * 0.8, ma * 0.8]
            elif isinstance(criteria[key], bool) and 'weight' in adj:
                criteria[key] = adj['weight'] == 2
    preferences['weights'] = weights
    if save:
        with open('user_preferences.json', 'w', encoding='utf-8') as f:
            json.dump(preferences, f, ensure_ascii=False, indent=4)
        with open('updated_criteria.json', 'w', encoding='utf-8') as f:
            json.dump(criteria, f, ensure_ascii=False, indent=4)


def calculate_score(house: dict, criteria: dict, weights: dict) -> float:
    score = 0.0
    for key, weight in weights.items():
        if not weight:
            continue
        if key in ['price', 'bedrooms', 'bathrooms', 'area']:
            value = house.get(key)
            if value is None:
                continue
            if key in criteria and isinstance(criteria[key], list):
                mi, ma = criteria[key]
            else:
                mi = ma = value
            if mi <= value <= ma:
                attr_score = 100.0
            else:
                diff = (mi - value) if value < mi else (value - ma)
                span = ma - mi if ma != mi else (ma if ma != 0 else 1)
                attr_score = max(0.0, 1 - diff / span) * 100.0
            score += attr_score * weight
        elif key == 'house_type':
            desired = criteria.get('house_type')
            actual = house.get('house_type')
            if desired and actual and desired.lower() in str(actual).lower():
                score += 100.0 * weight
            else:
                score += 50.0 * weight
        elif key in ['hospital_nearby', 'school_nearby']:
            if not criteria.get(key, False):
                continue
            dist = house.get(
                'distance_to_nearest_hospital' if key == 'hospital_nearby' else 'distance_to_nearest_school',
                float('inf'))
            max_d = criteria.get(f'{key}_max_distance', 10.0)
            if dist == float('inf'):
                attr_score = 0.0
            else:
                attr_score = max(0.0, 1 - dist / max_d) * 100.0
            score += attr_score * weight
    return score


def rank_properties(houses: list, criteria: dict, weights: dict, top_n: int = None) -> list:
    scored = []
    for h in houses:
        s = calculate_score(h, criteria, weights)
        scored.append((h, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    if top_n is not None:
        return scored[:top_n]
    return scored

