import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv


class HouseGCN(nn.Module):
    """两层GCN提取房产节点嵌入"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x


def train_gnn(graph, embedding_dim: int = 16, epochs: int = 100) -> list:
    """在异构图上训练GCN模型并返回房产嵌入"""
    house_ids = [int(n.split('_')[1]) for n, d in graph.nodes(data=True) if d.get('node_type') == 'house']
    house_count = max(house_ids) + 1 if house_ids else 0
    school_nodes = [n for n, d in graph.nodes(data=True) if d.get('facility_type') == 'school']
    hospital_nodes = [n for n, d in graph.nodes(data=True) if d.get('facility_type') == 'hospital']
    school_count = len(school_nodes)
    hospital_count = len(hospital_nodes)
    total_nodes = house_count + school_count + hospital_count

    features = torch.zeros((total_nodes, 6), dtype=torch.float)
    for n, data in graph.nodes(data=True):
        if data.get('node_type') == 'house':
            hid = int(n.split('_')[1])
            features[hid, 0] = float(data.get('price_norm', 0.0))
            features[hid, 1] = float(data.get('bedrooms_norm', 0.0))
            features[hid, 2] = float(data.get('bathrooms_norm', 0.0))
            features[hid, 3] = float(data.get('area_norm', 0.0))
        else:
            if data.get('facility_type') == 'school':
                idx = house_count + int(n.split('_')[1])
                features[idx, 4] = 1.0
            elif data.get('facility_type') == 'hospital':
                idx = house_count + school_count + int(n.split('_')[1])
                features[idx, 5] = 1.0

    edges = []
    weights = []
    for u, v, data in graph.edges(data=True):
        if u.startswith('house') and v.startswith('house'):
            continue
        if u.startswith('house'):
            house_node, fac_node = u, v
        elif v.startswith('house'):
            house_node, fac_node = v, u
        else:
            continue
        hid = int(house_node.split('_')[1])
        if fac_node.startswith('school'):
            fid = house_count + int(fac_node.split('_')[1])
        else:
            fid = house_count + school_count + int(fac_node.split('_')[1])
        w = float(data.get('weight', 1.0))
        edges.append([hid, fid])
        weights.append(w)
        edges.append([fid, hid])
        weights.append(w)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float) if edges else torch.empty((0,), dtype=torch.float)

    input_dim = features.size(1)
    model = HouseGCN(input_dim, 16, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        embed = model(features, edge_index, edge_weight)
        pos, neg = [], []
        for u, v in graph.edges():
            if u.startswith('house'):
                hid, fid_node = int(u.split('_')[1]), v
            else:
                hid, fid_node = int(v.split('_')[1]), u
            if fid_node.startswith('school'):
                fid = house_count + int(fid_node.split('_')[1])
            else:
                fid = house_count + school_count + int(fid_node.split('_')[1])
            pos.append(torch.sigmoid((embed[hid] * embed[fid]).sum()))
        import random
        house_indices = [int(n.split('_')[1]) for n in graph.nodes() if n.startswith('house')]
        for _ in range(len(graph.edges())):
            if not house_indices or (school_count + hospital_count) == 0:
                break
            hid = random.choice(house_indices)
            fid = house_count + random.randrange(school_count + hospital_count)
            if not ((f'house_{hid}', f'school_{fid - house_count}') in graph.edges() or (f'house_{hid}', f'hospital_{fid - house_count - school_count}') in graph.edges()):
                neg.append(torch.sigmoid((embed[hid] * embed[fid]).sum()))
        if not pos:
            break
        loss_pos = -torch.stack(pos).log().mean()
        loss = loss_pos
        if neg:
            loss_neg = -(1 - torch.stack(neg)).log().mean()
            loss += loss_neg
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_embed = model(features, edge_index, edge_weight).cpu().numpy()
    house_embeddings = [final_embed[i].tolist() if i < final_embed.shape[0] else [0.0] * embedding_dim for i in range(house_count)]
    return house_embeddings

