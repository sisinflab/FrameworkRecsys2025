# src/embedding_models/gin_runner.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import HeteroConv, GINConv, LayerNorm
from torch.nn import LazyLinear
from collections import OrderedDict

from typing import Dict
from torch import Tensor


class HeteroGIN(nn.Module):
    """
    Versione migliorata di GIN eterogenea:
    - Proiezione iniziale delle feature in 'channels' via LazyLinear
    - 'num_layers' di GINConv con MLP interno di 'num_layers_nn' Linear(channels→channels)+ReLU
    - Dopo ogni layer, LayerNorm per ogni tipo di nodo + ReLU
    """
    def __init__(
        self,
        metadata,
        channels: int,
        num_layers: int = 2,
        num_layers_nn: int = 2,
        aggr: str = "sum",
        eps: float = 1e-5,
        train_eps: bool = True,
    ) -> None:
        super().__init__()
        node_types, edge_types = metadata

        # 0) Proiezione iniziale per ciascuna x_dict[node] → [*, channels]
        self.lin_initial = nn.ModuleDict({
            node: LazyLinear(channels)
            for node in node_types
        })

        # 1) Layer GINConv multipli
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            # Costruisco l'MLP interno di ciascun GINConv:
            # num_layers_nn strati di Linear(channels→channels)+ReLU
            mlp_list = []
            for layer_nn in range(num_layers_nn):
                mlp_list.append((f"nn_{layer}_{layer_nn}", nn.Linear(channels, channels, bias=True)))
                mlp_list.append((f"relu_{layer}_{layer_nn}", nn.ReLU()))
            # (Facoltativo: mantengo anche la ReLU finale)
            mlp = nn.Sequential(OrderedDict(mlp_list))

            # Converto aggr in stringa (ma di solito cfg["aggr"] lo è già)
            aggr_str = str(aggr)

            # Creo HeteroConv con GINConv:
            conv = HeteroConv(
                {
                    edge_type: GINConv(
                        nn=mlp,
                        eps=eps,             # sarà già float
                        train_eps=train_eps, # sarà già bool
                        aggr=aggr_str
                    )
                    for edge_type in edge_types
                },
                aggr="sum"  # sommo i risultati tra relazioni diverse
            )
            self.convs.append(conv)

        # 2) LayerNorm dopo ogni GINConv, per ciascun tipo di nodo
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = nn.ModuleDict({
                node: LayerNorm(channels, mode="node")
                for node in node_types
            })
            self.norms.append(norm_dict)

        self.channels = channels

    def reset_parameters(self) -> None:
        for lin in self.lin_initial.values():
            lin.reset_parameters()
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
        # 1) Proietto le feature iniziali in dimensione 'channels'
        x_dict = {node: self.lin_initial[node](feat) for node, feat in x_dict.items()}

        # 2) Ciclo su ciascun GINConv + LayerNorm + ReLU
        for conv, norm_dict in zip(self.convs, self.norms):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {node: norm_dict[node](feat) for node, feat in x_dict.items()}
            x_dict = {node: feat.relu() for node, feat in x_dict.items()}

        return x_dict


class MultiTaskHeads(nn.Module):
    """
    Identica al tuo precedente: 3 testine su embedding 'client':
    - churn (binaria)
    - category propensity (multi-label)
    - sku propensity (multi-label)
    """
    def __init__(self, embedding_dim: int, num_categories: int, num_skus: int):
        super().__init__()
        # Churn (binaria)
        self.churn_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        # Category propensity (multi‐label)
        self.category_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_categories),
            nn.Sigmoid()
        )
        # SKU propensity (multi‐label)
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


class FullGINModel(nn.Module):
    """
    Modello completo che incapsula HeteroGIN + MultiTaskHeads.
    """
    def __init__(
        self,
        metadata,
        channels: int,
        num_layers: int,
        num_layers_nn: int,
        num_categories: int,
        num_skus: int,
        aggr: str = "sum",
        eps: float = 1e-5,
        train_eps: bool = True
    ):
        super().__init__()
        # Modulo GIN eterogeneo
        self.gin = HeteroGIN(
            metadata=metadata,
            channels=int(channels),
            num_layers=int(num_layers),
            num_layers_nn=int(num_layers_nn),
            aggr=str(aggr),
            eps=float(eps),
            train_eps=bool(train_eps)
        )
        # Testine multitask
        self.heads = MultiTaskHeads(channels, num_categories, num_skus)

    def forward(self, data: HeteroData):
        # 1) Normalizzo L2 le feature iniziali (opzionale; il GIN usa già LayerNorm nei layer)
        x_dict = {node: F.normalize(feat, p=2, dim=-1) for node, feat in data.x_dict.items()}

        # 2) Passo x_dict proiettate nel GIN
        x_dict = self.gin(x_dict, data.edge_index_dict)

        # 3) Estraggo embedding 'client'
        user_emb = x_dict['client']  # [num_users, channels]

        # 4) (Opzionale) Normalizzo di nuovo user_emb L2
        user_emb = F.normalize(user_emb, p=2, dim=-1)

        # 5) Passo user_emb alle testine multitask
        churn_pred, cat_pred, sku_pred = self.heads(user_emb)

        return churn_pred, cat_pred, sku_pred, user_emb


def train_hetero_gin(
    data: HeteroData,
    y_churn: torch.Tensor,
    y_category: torch.Tensor,
    y_sku: torch.Tensor,
    cfg: dict
) -> torch.Tensor:
    """
    Training loop per GIN eterogeneo.
    - data: HeteroData già non-diretto
    - y_churn: [num_users, 1]
    - y_category: [num_users, num_categories]
    - y_sku: [num_users, num_skus]
    - cfg: {
          "channels": int or str,
          "num_layers": int or str,
          "num_layers_nn": int or str,
          "lr": float or str,
          "epochs": int or str,
          "device": str,
          "aggr": str,
          "eps": float or str,
          "train_eps": bool or str
      }
    Restituisce: user_emb [num_users, channels] su CPU.
    """
    # Cast espliciti per sicurezza
    channels    = int(cfg["channels"])
    num_layers  = int(cfg.get("num_layers", 2))
    num_layers_nn = int(cfg.get("num_layers_nn", 2))
    aggr        = str(cfg.get("aggr", "sum"))
    eps         = float(cfg.get("eps", 1e-5))
    train_eps   = bool(cfg.get("train_eps", True))
    lr          = float(cfg["lr"])
    epochs      = int(cfg["epochs"])
    device      = torch.device(cfg.get("device", "cpu"))

    data = ToUndirected()(data).to(device)

    # Sposto i target su device
    y_churn    = y_churn.to(device)
    y_category = y_category.to(device)
    y_sku      = y_sku.to(device)

    # Crea il modello FullGINModel
    model = FullGINModel(
        metadata=data.metadata(),
        channels=channels,
        num_layers=num_layers,
        num_layers_nn=num_layers_nn,
        num_categories=y_category.size(1),
        num_skus=y_sku.size(1),
        aggr=aggr,
        eps=eps,
        train_eps=train_eps
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    for epoch in range(1, epochs + 1):
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
            print(f"[GIN Epoch {epoch}/{epochs}] Loss: {loss.item():.4f}")

        torch.cuda.empty_cache()

    # Estrai embedding finale utenti
    model.eval()
    with torch.no_grad():
        _, _, _, user_emb = model(data)
        user_emb = user_emb.cpu()

    torch.cuda.empty_cache()
    return user_emb
