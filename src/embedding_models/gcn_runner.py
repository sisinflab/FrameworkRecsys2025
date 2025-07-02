# src/embedding_models/gcn_runner.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import HeteroConv, LayerNorm, GraphConv
from torch.nn import LazyLinear

from typing import Dict
from torch import Tensor


class HeteroGCN(nn.Module):
    """
    HeteroGCN: rete eterogenea che applica num_layers di HeteroConv con GCNConv
    - Ogni GCNConv in_channels = out_channels = 'channels'
    - Dopo ogni layer, LayerNorm + ReLU per ogni tipo di nodo
    """
    def __init__(
        self,
        node_types: list[str],
        edge_types: list[tuple[str, str, str]],
        channels: int,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        # Creiamo num_layers istanze di HeteroConv({edge_type: GCNConv(channels→channels)})
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: GraphConv(
                        in_channels=channels,
                        out_channels=channels,
                        aggr="mean",
                        bias=True
                    )
                    for edge_type in edge_types
                },
                aggr="sum"  # aggregazione tra diverse relazioni
            )
            self.convs.append(conv)

        # LayerNorm + ReLU dopo ogni HeteroConv
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = nn.ModuleDict({
                node_type: LayerNorm(channels, mode="node")
                for node_type in node_types
            })
            self.norms.append(norm_dict)

        self.channels = channels
        self.num_layers = num_layers

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        # Applico in sequenza i layer GCN eterogenei + LayerNorm + ReLU
        for conv, norm_dict in zip(self.convs, self.norms):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {nt: norm_dict[nt](feat) for nt, feat in x_dict.items()}
            x_dict = {nt: feat.relu() for nt, feat in x_dict.items()}

        return x_dict


class MultiTaskHeads(nn.Module):
    """
    Le tre testine sullo stesso embedding utente:
      - churn (binario)
      - category propensity (multi-label)
      - sku propensity (multi-label)
    """
    def __init__(self, embedding_dim: int, num_categories: int, num_skus: int):
        super().__init__()
        # 1) Churn: Linear -> ReLU -> Linear(->1) -> Sigmoid
        self.churn_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        # 2) Category propensity: Linear -> ReLU -> Linear(num_categories) -> Sigmoid
        self.category_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_categories),
            nn.Sigmoid()
        )
        # 3) SKU propensity: Linear -> ReLU -> Linear(num_skus) -> Sigmoid
        self.sku_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_skus),
            nn.Sigmoid()
        )

    def forward(self, user_emb: Tensor):
        churn    = self.churn_head(user_emb)
        category = self.category_head(user_emb)
        sku      = self.sku_head(user_emb)
        return churn, category, sku


class FullGCNModel(nn.Module):
    """
    Modello completo con:
      1) Layer iniziale di proiezione (LazyLinear) per ogni tipo di nodo
      2) HeteroGCN (più layer di GCNConv+LayerNorm+ReLU)
      3) Testine multitask che operano solo sull'embedding 'client'
    """
    def __init__(
        self,
        metadata,
        channels: int,
        num_layers: int,
        num_categories: int,
        num_skus: int,
    ):
        super().__init__()

        node_types, edge_types = metadata

        # Proiezione iniziale per ciascun tipo di nodo: x_dict[node] → [*, channels]
        self.lin_initial = nn.ModuleDict({
            node: LazyLinear(channels)
            for node in node_types
        })

        # Rete GCN eterogenea
        self.gcn = HeteroGCN(
            node_types=node_types,
            edge_types=edge_types,
            channels=channels,
            num_layers=num_layers
        )

        # Testine multitask
        self.heads = MultiTaskHeads(channels, num_categories, num_skus)

        self.channels = channels

    def forward(self, data: HeteroData):
        # 1) Normalizzo L2 le feature iniziali (opzionale)
        x_dict = {nt: F.normalize(feat, p=2, dim=-1) for nt, feat in data.x_dict.items()}

        # 2) Proietto tutte le feature in dimensione 'channels'
        x_dict = {nt: self.lin_initial[nt](feat) for nt, feat in x_dict.items()}

        # 3) Passo x_dict + edge_index al GCN eterogeneo
        x_dict = self.gcn(x_dict, data.edge_index_dict)

        # 4) Estraggo l'embedding 'client'
        user_emb = x_dict['client']  # (num_users, channels)

        # 5) (Opzionale) Normalizzo di nuovo user_emb L2
        user_emb = F.normalize(user_emb, p=2, dim=-1)

        # 6) Passo user_emb alle tre testine multitask
        churn_pred, cat_pred, sku_pred = self.heads(user_emb)

        return churn_pred, cat_pred, sku_pred, user_emb


def train_hetero_gcn(
    data: HeteroData,
    y_churn: Tensor,
    y_category: Tensor,
    y_sku: Tensor,
    cfg: dict
) -> Tensor:
    """
    Funzione di training per la GCN eterogenea:
      - data: HeteroData (già non-diretto)
      - y_churn: [num_users, 1]
      - y_category: [num_users, num_categories]
      - y_sku: [num_users, num_skus]
      - cfg: {
          "channels": int,
          "num_layers": int,
          "lr": float,
          "epochs": int,
          "device": str
        }
    Restituisce user_emb [num_users, channels] su CPU.
    """
    device = torch.device(cfg.get("device", "cpu"))
    data = ToUndirected()(data).to(device)

    # Sposto i target sul device
    y_churn    = y_churn.to(device)
    y_category = y_category.to(device)
    y_sku      = y_sku.to(device)

    # Inizializzo il modello FullGCNModel
    model = FullGCNModel(
        metadata=data.metadata(),
        channels=cfg["channels"],
        num_layers=cfg["num_layers"],
        num_categories=y_category.size(1),
        num_skus=y_sku.size(1)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]))
    loss_fn = nn.BCELoss()

    # Training loop
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad()

        churn_pred, cat_pred, sku_pred, _ = model(data)
        loss = (
            loss_fn(churn_pred.view(-1),   y_churn.view(-1)) +
            loss_fn(cat_pred,               y_category) +
            loss_fn(sku_pred,               y_sku)
        )
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0:
            print(f"[GCN Epoch {epoch}/{cfg['epochs']}] Loss: {loss.item():.4f}")

        torch.cuda.empty_cache()

    # Estrai embedding finale utenti
    model.eval()
    with torch.no_grad():
        _, _, _, user_emb = model(data)
        user_emb = user_emb.cpu()

    torch.cuda.empty_cache()
    return user_emb
