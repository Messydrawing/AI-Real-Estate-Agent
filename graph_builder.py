import networkx as nx
from geopy.distance import geodesic


def build_graph(houses: list, schools: list, hospitals: list, distance_threshold: float = 20.0) -> nx.Graph:
    """构建房产-设施异构图，边权反映距离远近"""
    G = nx.Graph()
    for house in houses:
        node_id = f"house_{house['id']}"
        features = {
            'node_type': 'house',
            'price': house.get('price'),
            'bedrooms': house.get('bedrooms'),
            'bathrooms': house.get('bathrooms'),
            'area': house.get('area'),
            'price_norm': house.get('price_norm', 0.0),
            'bedrooms_norm': house.get('bedrooms_norm', 0.0),
            'bathrooms_norm': house.get('bathrooms_norm', 0.0),
            'area_norm': house.get('area_norm', 0.0),
            'house_type': house.get('house_type'),
            'coordinates': house.get('coordinates')
        }
        G.add_node(node_id, **features)
    for idx, (lat, lon) in enumerate(schools):
        node_id = f"school_{idx}"
        G.add_node(node_id, node_type='facility', facility_type='school', coordinates=(lat, lon))
    for idx, (lat, lon) in enumerate(hospitals):
        node_id = f"hospital_{idx}"
        G.add_node(node_id, node_type='facility', facility_type='hospital', coordinates=(lat, lon))
    for house in houses:
        if not house.get('coordinates'):
            continue
        house_node = f"house_{house['id']}"
        lat, lon = house['coordinates']
        for idx, (s_lat, s_lon) in enumerate(schools):
            dist = geodesic((lat, lon), (s_lat, s_lon)).kilometers
            if dist <= distance_threshold:
                w = max(0.0, 1 - dist / distance_threshold)
                G.add_edge(house_node, f"school_{idx}", distance=dist, weight=w)
        for idx, (h_lat, h_lon) in enumerate(hospitals):
            dist = geodesic((lat, lon), (h_lat, h_lon)).kilometers
            if dist <= distance_threshold:
                w = max(0.0, 1 - dist / distance_threshold)
                G.add_edge(house_node, f"hospital_{idx}", distance=dist, weight=w)
    return G

