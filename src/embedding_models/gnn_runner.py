# embedding/gnn_runner.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import HeteroConv, SAGEConv
from torch.nn import LazyLinear


class HeteroGNN(nn.Module):
    def __init__(self, metadata, hidden_channels: int, out_channels: int):
        super().__init__()
        node_types, edge_types = metadata

        # Primo layer: SAGEConv per ogni relazione
        self.conv1 = HeteroConv(
            {edge_type: SAGEConv((-1, -1), hidden_channels)
             for edge_type in edge_types},
            aggr='sum'
        )
        # Skip‐linear per ogni tipo di nodo
        self.lin1 = nn.ModuleDict({
            node: LazyLinear(hidden_channels)
            for node in node_types
        })

        # Secondo layer
        self.conv2 = HeteroConv(
            {edge_type: SAGEConv((-1, -1), out_channels)
             for edge_type in edge_types},
            aggr='sum'
        )
        self.lin2 = nn.ModuleDict({
            node: LazyLinear(out_channels)
            for node in node_types
        })

    def forward(self, x_dict, edge_index_dict):
        # Layer1 + skip + ReLU
        h1 = self.conv1(x_dict, edge_index_dict)
        for node, x0 in x_dict.items():
            h1_node = h1.get(node, 0)
            h1[node] = F.relu(h1_node + self.lin1[node](x0))

        # Layer2 + skip
        h2 = self.conv2(h1, edge_index_dict)
        out = {}
        for node, h1_node in h1.items():
            h2_node = h2.get(node, 0)
            out[node] = h2_node + self.lin2[node](h1_node)

        return out


class MultiTaskHeads(nn.Module):
    def __init__(self, embedding_dim: int, num_categories: int, num_skus: int):
        super().__init__()
        # 1) Churn prediction (binaria)
        self.churn_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        # 2) Category propensity (multi‐label)
        self.category_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_categories),
            nn.Sigmoid()
        )
        # 3) SKU propensity (multi‐label)
        self.sku_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_skus),
            nn.Sigmoid()
        )

    def forward(self, user_emb: torch.Tensor):
        churn    = self.churn_head(user_emb)
        category = self.category_head(user_emb)
        sku      = self.sku_head(user_emb)
        return churn, category, sku


class FullModel(nn.Module):
    def __init__(self,
                 metadata,
                 hidden_channels: int,
                 out_channels: int,
                 num_categories: int,
                 num_skus: int):
        super().__init__()
        self.gnn   = HeteroGNN(metadata, hidden_channels, out_channels)
        self.heads = MultiTaskHeads(out_channels, num_categories, num_skus)

    def forward(self, data: HeteroData):
        # Normalizza gli embeddings prima di passarli al modello
        x_dict = {key: F.normalize(val, p=2, dim=-1) for key, val in data.x_dict.items()}

        # Passa gli embeddings normalizzati attraverso il GNN
        x_dict = self.gnn(x_dict, data.edge_index_dict)

        # Estrai l'embedding dell'utente
        user_emb = x_dict['client']

        # Passa l'embedding dell'utente attraverso i vari task
        churn_pred, cat_pred, sku_pred = self.heads(user_emb)
        
        return churn_pred, cat_pred, sku_pred, user_emb



def train_hetero_gnn(
    data: HeteroData,
    y_churn: torch.Tensor,
    y_category: torch.Tensor,
    y_sku: torch.Tensor,
    cfg: dict
) -> torch.Tensor:
   
    device = torch.device(cfg.get("device", "cpu"))
    data   = ToUndirected()(data).to(device)

    # Porta i target su device
    y_churn    = y_churn.to(device)
    y_category = y_category.to(device)
    y_sku      = y_sku.to(device)

    # Inizializza modello, ottimizzatore e loss
    model = FullModel(
        metadata=data.metadata(),
        hidden_channels=cfg["hidden_channels"],
        out_channels=cfg["out_channels"],
        num_categories=y_category.size(1),
        num_skus=y_sku.size(1)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]))
    loss_fn   = nn.BCELoss()

    # Training loop
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        churn_pred, cat_pred, sku_pred, _ = model(data)

        loss = (
            loss_fn(churn_pred.squeeze(),   y_churn.squeeze()) +
            loss_fn(cat_pred,               y_category) +
            loss_fn(sku_pred,               y_sku)
        )

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"[Epoch {epoch}/{cfg['epochs']}] Loss: {loss.item():.4f}")

        torch.cuda.empty_cache()

    # Estrai gli embedding finali
    model.eval()
    with torch.no_grad():
        _, _, _, user_emb = model(data)

    torch.cuda.empty_cache()

    return user_emb.cpu()

"""import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import HeteroConv, SAGEConv
from torch.nn import LazyLinear
import random


class HeteroGNN(nn.Module):
    def __init__(self, metadata, hidden_channels: int, out_channels: int):
        super().__init__()
        node_types, edge_types = metadata
        self.conv1 = HeteroConv(
            {etype: SAGEConv((-1, -1), hidden_channels) for etype in edge_types}, aggr='sum')
        self.lin1 = nn.ModuleDict({ntype: LazyLinear(hidden_channels) for ntype in node_types})

        self.conv2 = HeteroConv(
            {etype: SAGEConv((-1, -1), out_channels) for etype in edge_types}, aggr='sum')
        self.lin2 = nn.ModuleDict({ntype: LazyLinear(out_channels) for ntype in node_types})

    def forward(self, x_dict, edge_index_dict):
        h1 = self.conv1(x_dict, edge_index_dict)
        for node, x0 in x_dict.items():
            h1_node = h1.get(node, torch.zeros_like(x0))
            h1[node] = F.relu(h1_node + self.lin1[node](x0))

        h2 = self.conv2(h1, edge_index_dict)
        return {node: h2.get(node, 0) + self.lin2[node](h1_node) for node, h1_node in h1.items()}


class MultiTaskHeads(nn.Module):
    def __init__(self, embedding_dim, num_categories, num_skus):
        super().__init__()
        self.churn_head = nn.Sequential(
            nn.Linear(embedding_dim, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid())
        self.category_head = nn.Sequential(
            nn.Linear(embedding_dim, 128), nn.ReLU(),
            nn.Linear(128, num_categories), nn.Sigmoid())
        self.sku_head = nn.Sequential(
            nn.Linear(embedding_dim, 128), nn.ReLU(),
            nn.Linear(128, num_skus), nn.Sigmoid())

    def forward(self, user_emb):
        return (self.churn_head(user_emb),
                self.category_head(user_emb),
                self.sku_head(user_emb))


class FullModel(nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_categories, num_skus):
        super().__init__()
        self.gnn = HeteroGNN(metadata, hidden_channels, out_channels)
        self.heads = MultiTaskHeads(out_channels, num_categories, num_skus)

    def forward(self, data: HeteroData):
        x_dict = self.gnn(data.x_dict, data.edge_index_dict)
        return (*self.heads(x_dict['client']), x_dict['client'])


def get_batch_data(full_data: HeteroData, client_idx_batch):
    batch_data = HeteroData()
    device = full_data['client'].x.device
    client_idx = torch.tensor(client_idx_batch, device=device)

    # Client nodes features subset
    batch_data['client'].x = full_data['client'].x[client_idx]

    # Copy other node types features entirely (optional: optimize if needed)
    for ntype in full_data.node_types:
        if ntype != 'client':
            batch_data[ntype].x = full_data[ntype].x

    # Filter edges where source OR target node is in client batch
    for etype in full_data.edge_types:
        edge_idx = full_data[etype].edge_index
        src_mask = torch.isin(edge_idx[0], client_idx)
        dst_mask = torch.isin(edge_idx[1], client_idx)
        mask = src_mask | dst_mask
        batch_data[etype].edge_index = edge_idx[:, mask]

    return batch_data


def train_hetero_gnn(data: HeteroData, y_churn, y_category, y_sku, cfg):
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    data = ToUndirected()(data)  # keep on CPU until batching
    model = FullModel(
        metadata=data.metadata(),
        hidden_channels=cfg["hidden_channels"],
        out_channels=cfg["out_channels"],
        num_categories=y_category.size(1),
        num_skus=y_sku.size(1)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]))
    loss_fn = nn.BCELoss()
    batch_size = cfg.get("batch_size", 1)
    client_indices = list(range(data['client'].num_nodes))

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        random.shuffle(client_indices)
        total_loss = 0.0

        for i in range(0, len(client_indices), batch_size):
            batch_ids = client_indices[i:i + batch_size]
            batch_data = get_batch_data(data, batch_ids).to(device)

            b_y_churn = y_churn[batch_ids].to(device)
            b_y_category = y_category[batch_ids].to(device)
            b_y_sku = y_sku[batch_ids].to(device)

            optimizer.zero_grad()
            churn_pred, cat_pred, sku_pred, _ = model(batch_data)
            loss = (
                loss_fn(churn_pred.view(-1), b_y_churn.view(-1)) +
                loss_fn(cat_pred, b_y_category) +
                loss_fn(sku_pred, b_y_sku)
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        torch.cuda.empty_cache()

        avg_loss = total_loss / (len(client_indices) // batch_size + 1)
        if epoch == 1 or epoch % 10 == 0:
            print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        data = data.to(device)  # Load entire graph for final embeddings
        _, _, _, user_emb = model(data)
        user_emb = user_emb.detach().cpu()

    return user_emb"""
