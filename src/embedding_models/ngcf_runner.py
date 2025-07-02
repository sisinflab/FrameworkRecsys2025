import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import HeteroConv, MessagePassing, LayerNorm
from torch.nn import LazyLinear
from typing import Dict
from torch import Tensor

class NGCFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__(aggr='add')
        self.linear = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU()

    def forward(self, x, edge_index):
        # x può essere Tensor o Tuple[Tensor, Tensor]
        if isinstance(x, tuple):
            x_i, x_j = x
            x_i = self.dropout(x_i)
            x_j = self.dropout(x_j)
        else:
            x_i = x_j = self.dropout(x)

        return self.propagate(edge_index=edge_index, x=(x_i, x_j))

    def message(self, x_j, x_i):
        return self.act(self.linear(x_j * x_i))

    def update(self, aggr_out):
        return aggr_out



class HeteroNGCF(nn.Module):
    def __init__(
        self,
        node_types: list[str],
        edge_types: list[tuple[str, str, str]],
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: NGCFConv(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        dropout=dropout
                    )
                    for edge_type in edge_types
                },
                aggr="sum"
            )
            self.convs.append(conv)

        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = nn.ModuleDict({
                node_type: LayerNorm(out_channels, mode="node")
                for node_type in node_types
            })
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            for module in conv.convs.values():
                module.linear.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(self, x_dict: Dict[str, Tensor], edge_index_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        for conv, norm_dict in zip(self.convs, self.norms):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {nt: norm_dict[nt](feat) for nt, feat in x_dict.items()}
            x_dict = {nt: feat.relu() for nt, feat in x_dict.items()}
        return x_dict


class MultiTaskHeads(nn.Module):
    def __init__(self, embedding_dim: int, num_categories: int, num_skus: int):
        super().__init__()
        self.churn_head = nn.Sequential(nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.category_head = nn.Sequential(nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, num_categories), nn.Sigmoid())
        self.sku_head = nn.Sequential(nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, num_skus), nn.Sigmoid())

    def forward(self, user_emb: torch.Tensor):
        churn = self.churn_head(user_emb)
        category = self.category_head(user_emb)
        sku = self.sku_head(user_emb)
        return churn, category, sku


class FullNGCFModel(nn.Module):
    def __init__(
        self,
        metadata,
        channels: int,
        num_layers: int,
        num_categories: int,
        num_skus: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        node_types, edge_types = metadata

        self.lin_initial = nn.ModuleDict({
            node: LazyLinear(channels)
            for node in node_types
        })

        self.encoder = HeteroNGCF(
            node_types=node_types,
            edge_types=edge_types,
            in_channels=channels,
            out_channels=channels,
            num_layers=num_layers,
            dropout=dropout
        )

        self.heads = MultiTaskHeads(channels, num_categories, num_skus)

    def forward(self, data: HeteroData):
        x_dict = {nt: F.normalize(feat, p=2, dim=-1) for nt, feat in data.x_dict.items()}
        x_dict = {nt: self.lin_initial[nt](feat) for nt, feat in x_dict.items()}
        x_dict = self.encoder(x_dict, data.edge_index_dict)
        user_emb = F.normalize(x_dict['client'], p=2, dim=-1)
        churn_pred, cat_pred, sku_pred = self.heads(user_emb)
        return churn_pred, cat_pred, sku_pred, user_emb


def train_hetero_ngcf(
    data: HeteroData,
    y_churn: Tensor,
    y_category: Tensor,
    y_sku: Tensor,
    cfg: dict
) -> Tensor:
    device = torch.device(cfg.get("device", "cpu"))
    data = ToUndirected()(data).to(device)
    y_churn, y_category, y_sku = y_churn.to(device), y_category.to(device), y_sku.to(device)

    model = FullNGCFModel(
        metadata=data.metadata(),
        channels=cfg["channels"],
        num_layers=cfg["num_layers"],
        num_categories=y_category.size(1),
        num_skus=y_sku.size(1),
        dropout=cfg.get("dropout", 0.2)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]))
    loss_fn = nn.BCELoss()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        churn_pred, cat_pred, sku_pred, _ = model(data)
        loss = (
            loss_fn(churn_pred.view(-1), y_churn.view(-1)) +
            loss_fn(cat_pred, y_category) +
            loss_fn(sku_pred, y_sku)
        )
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0:
            print(f"[NGCF Epoch {epoch}/{cfg['epochs']}] Loss: {loss.item():.4f}")
        torch.cuda.empty_cache()

    model.eval()
    with torch.no_grad():
        _, _, _, user_emb = model(data)
        user_emb = user_emb.cpu()
    torch.cuda.empty_cache()
    return user_emb