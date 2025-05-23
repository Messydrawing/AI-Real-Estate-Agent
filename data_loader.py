import json
from geopy.distance import geodesic
from utils import compute_statistics, normalize_features

DEFAULT_DISTANCE_THRESHOLD = 5.0  # 公里


def load_user_preferences(preferences_file: str) -> dict:
    """加载用户偏好设置"""
    with open(preferences_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_criteria(criteria_file: str) -> dict:
    """加载筛选条件"""
    with open(criteria_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_houses(houses_file: str) -> list:
    """加载房源列表并赋予唯一 id"""
    with open(houses_file, 'r', encoding='utf-8') as f:
        houses = json.load(f)
    for idx, house in enumerate(houses):
        house['id'] = idx
        if 'house_size' in house:
            house['area'] = house.pop('house_size')
    return houses


def load_facilities(facilities_file: str) -> tuple:
    """加载学校和医院坐标"""
    with open(facilities_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    schools, hospitals = [], []
    for feat in data.get('features', []):
        props = feat.get('properties', {})
        amenity = str(props.get('amenity', '')).lower()
        coords = feat.get('geometry', {}).get('coordinates', [])
        if len(coords) == 2:
            lon, lat = coords
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                if amenity == 'school':
                    schools.append((lat, lon))
                elif amenity == 'hospital':
                    hospitals.append((lat, lon))
    return schools, hospitals


def filter_houses(houses: list, criteria: dict, schools: list = None,
                  hospitals: list = None,
                  distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD) -> list:
    """按照筛选条件过滤房源并计算设施距离"""
    filtered = []
    for house in houses:
        try:
            if house.get('price') is not None:
                house['price'] = float(house['price'])
            if house.get('bedrooms') is not None:
                house['bedrooms'] = int(float(house['bedrooms']))
            if house.get('bathrooms') is not None:
                house['bathrooms'] = int(float(house['bathrooms']))
            if house.get('area') is not None:
                house['area'] = float(house['area'])
        except (ValueError, TypeError):
            continue
        if 'price' in criteria and isinstance(criteria['price'], list):
            p = house.get('price')
            if p is None or p < criteria['price'][0] or p > criteria['price'][1]:
                continue
        if 'bedrooms' in criteria and isinstance(criteria['bedrooms'], list):
            br = house.get('bedrooms')
            if br is None or br < criteria['bedrooms'][0] or br > criteria['bedrooms'][1]:
                continue
        if 'bathrooms' in criteria and isinstance(criteria['bathrooms'], list):
            ba = house.get('bathrooms')
            if ba is None or ba < criteria['bathrooms'][0] or ba > criteria['bathrooms'][1]:
                continue
        if 'area' in criteria and isinstance(criteria['area'], list):
            a = house.get('area')
            if a is None or a < criteria['area'][0] or a > criteria['area'][1]:
                continue
        if criteria.get('house_type'):
            if criteria['house_type'].lower() not in str(house.get('house_type', '')).lower():
                continue
        filtered.append(house)

    schools = schools or []
    hospitals = hospitals or []
    for house in filtered:
        coords = house.get('coordinates')
        if not coords or len(coords) != 2:
            continue
        lat, lon = coords
        if hospitals:
            d_h = min((geodesic((lat, lon), s).kilometers for s in hospitals), default=float('inf'))
            house['distance_to_nearest_hospital'] = d_h
        if schools:
            d_s = min((geodesic((lat, lon), s).kilometers for s in schools), default=float('inf'))
            house['distance_to_nearest_school'] = d_s
    return filtered


def load_data(houses_file: str, facilities_file: str, preferences_file: str,
              criteria_file: str) -> tuple:
    """综合加载并预处理数据"""
    preferences = load_user_preferences(preferences_file)
    criteria = load_criteria(criteria_file)
    houses = load_houses(houses_file)
    schools, hospitals = load_facilities(facilities_file)
    houses = filter_houses(houses, criteria, schools, hospitals)
    stats = compute_statistics(houses)
    normalize_features(houses, stats)
    return houses, schools, hospitals, preferences, criteria

