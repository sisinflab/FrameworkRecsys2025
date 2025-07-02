import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import HeteroConv, GATConv, LayerNorm
from torch.nn import LazyLinear
from typing import Dict
from torch import Tensor


class HeteroGAT(nn.Module):
    def __init__(
        self,
        node_types: list[str],
        edge_types: list[tuple[str, str, str]],
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        heads: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: GATConv(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        heads=heads,
                        concat=False,
                        dropout=dropout,
                        add_self_loops=False,
                        bias=True,
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
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        for conv, norm_dict in zip(self.convs, self.norms):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {nt: norm_dict[nt](feat) for nt, feat in x_dict.items()}
            x_dict = {nt: feat.relu() for nt, feat in x_dict.items()}
        return x_dict


class MultiTaskHeads(nn.Module):
    def __init__(self, embedding_dim: int, num_categories: int, num_skus: int):
        super().__init__()
        # 1) Churn prediction (binary)
        self.churn_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        # 2) Category propensity (multi-label)
        self.category_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_categories),
            nn.Sigmoid()
        )
        # 3) SKU propensity (multi-label)
        self.sku_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_skus),
            nn.Sigmoid()
        )

    def forward(self, user_emb: torch.Tensor):
        churn    = self.churn_head(user_emb)
        category = self.category_head(user_emb)
        sku      = self.sku_head(user_emb)
        return churn, category, sku


class FullGATModel(nn.Module):
    """
    Full GAT Model consisting of:
      1) Initial projection layer for each node type (LazyLinear or nn.Linear)
      2) HeteroGAT (Multiple layers of GATConv + LayerNorm + ReLU)
      3) Multitask heads that operate only on the 'client' embedding
    """
    def __init__(
        self,
        metadata,
        channels: int,
        num_layers: int,
        num_categories: int,
        num_skus: int,
        heads: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        node_types, edge_types = metadata

        # Initial projection for each node type: x_dict[node] → [*, channels]
        self.lin_initial = nn.ModuleDict({
            node: LazyLinear(channels)  # Use LazyLinear if the input feature size is dynamic
            for node in node_types
        })

        # Define HeteroGAT (Heterogeneous Graph Attention Network)
        self.gat = HeteroGAT(
    node_types=node_types,
    edge_types=edge_types,
    in_channels=channels,      # input features (dopo projection)
    out_channels=channels,     # output embedding size per nodo
    num_layers=num_layers,
    heads=heads,
    dropout=dropout
)

        # Multitask heads for final predictions
        self.heads = MultiTaskHeads(channels, num_categories, num_skus)

    def forward(self, data: HeteroData):
        # 1) Normalize the input features (optional)
        x_dict = {nt: F.normalize(feat, p=2, dim=-1) for nt, feat in data.x_dict.items()}

        # 2) Project initial features for each node type
        x_dict = {nt: self.lin_initial[nt](feat) for nt, feat in x_dict.items()}

        # 3) Pass through the GAT layers
        x_dict = self.gat(x_dict, data.edge_index_dict)

        # 4) Extract the 'client' node embedding (user embeddings)
        user_emb = x_dict['client']  # Shape: [num_users, channels]

        # 5) Optionally normalize the user embedding (L2 normalization)
        user_emb = F.normalize(user_emb, p=2, dim=-1)

        # 6) Pass the user embedding through multitask heads
        churn_pred, cat_pred, sku_pred = self.heads(user_emb)

        # 7) Return predictions and final user embedding
        return churn_pred, cat_pred, sku_pred, user_emb


def train_hetero_gat(
    data: HeteroData,
    y_churn: Tensor,
    y_category: Tensor,
    y_sku: Tensor,
    cfg: dict
) -> Tensor:
    device = torch.device(cfg.get("device", "cpu"))
    data = ToUndirected()(data).to(device)

    model = FullGATModel(
        metadata=data.metadata(),
        channels=cfg["channels"],
        num_layers=cfg["num_layers"],
        num_categories=y_category.size(1),
        num_skus=y_sku.size(1),
        heads=cfg["heads"],
        dropout=cfg.get("dropout", 0.2)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]))
    loss_fn = nn.BCELoss()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad()

        # Sposta target sul device solo ora
        y_churn_device = y_churn.to(device)
        y_category_device = y_category.to(device)
        y_sku_device = y_sku.to(device)

        churn_pred, cat_pred, sku_pred, _ = model(data)
        loss = (
            loss_fn(churn_pred.view(-1), y_churn_device.view(-1)) +
            loss_fn(cat_pred, y_category_device) +
            loss_fn(sku_pred, y_sku_device)
        )
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0:
            print(f"[GAT Epoch {epoch}/{cfg['epochs']}] Loss: {loss.item():.4f}")

        # Libera memoria intermedia
        del churn_pred, cat_pred, sku_pred, loss
        torch.cuda.empty_cache()

    # Eval
    model.eval()
    with torch.no_grad():
        _, _, _, user_emb = model(data)
        user_emb = user_emb.cpu()
    torch.cuda.empty_cache()

    return user_emb
