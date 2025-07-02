import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected


class LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int, num_layers: int):
      
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_nodes = num_users + num_items
        self.num_layers = num_layers

        # Embedding iniziale: utenti da 0 a num_users-1, prodotti da num_users a num_users+num_items-1
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

        # Xavier initialization
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, edge_index: torch.Tensor):
       
        # Concateno i pesi di utente+prodotto in un unico tensor di shape [num_nodes, dim]
        embeddings = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)  # (N, D)
        all_embs = [embeddings]

        # Costruisco edge_index “undirected”:
        # - la parte originale è client→product: [u_indices; p_indices]
        # - per farlo simmetrico, faccio un flip(0): [p_indices; u_indices]
        # Poi con `torch.cat` ottengo un edge_index_full di shape [2, 2E]
        edge_index_full = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        # Pre‐calcolo grado normalizzato: deg_inv_sqrt per ciascun nodo
        row, col = edge_index_full  # entrambi shape [2E]
        deg = torch.bincount(row, minlength=self.num_nodes).float()  # (num_nodes,)
        # evito divisione su zero: se deg[i]==0 => deg_inv_sqrt=0
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg == 0] = 0.0

        # Layer‐by‐layer propagation
        for _ in range(self.num_layers):
            # calcolo norm: [2E] scalare moltiplicativo per ogni messaggio
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # (2E,)
            # init output embeddings con zeri
            out = torch.zeros_like(embeddings)  # (num_nodes, dim)
            # aggrego i valori dei vicini: out[i] += norm[e] * embeddings[j], per ogni (i=row[e], j=col[e])
            out = out.index_add_(0, row, embeddings[col] * norm.unsqueeze(1))
            embeddings = out
            all_embs.append(embeddings)

        # Media vettoriale su tutti gli strati (include layer‐0 iniziale)
        all_embs = torch.stack(all_embs, dim=1)       # (num_nodes, num_layers+1, dim)
        final_emb = all_embs.mean(dim=1)              # (num_nodes, dim)

        # Separo utenti e prodotti
        user_final = final_emb[: self.num_users, :]   # (num_users, dim)
        item_final = final_emb[self.num_users :, :]    # (num_items, dim)
        return user_final, item_final


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


class FullLightGCN(nn.Module):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 embedding_dim: int,
                 num_layers: int,
                 num_categories: int,
                 num_skus: int):
       
        super().__init__()
        self.lightgcn = LightGCN(num_users, num_items, embedding_dim, num_layers)
        self.heads = MultiTaskHeads(embedding_dim, num_categories, num_skus)

    def forward(self, edge_index: torch.Tensor):
       
        user_emb, _ = self.lightgcn(edge_index)   # (num_users, dim), (num_items, dim)
        # **Normalizzo L₂ l'embedding utente** prima di passarlo alle teste multitask
        # (p=2, dim=-1 significa "norma lungo l'ultima dimensione")
        #user_emb = F.normalize(user_emb, p=2, dim=-1)
        churn_pred, cat_pred, sku_pred = self.heads(user_emb)
        return churn_pred, cat_pred, sku_pred, user_emb


def train_lightgcn(
    data: HeteroData,
    y_churn: torch.Tensor,
    y_category: torch.Tensor,
    y_sku: torch.Tensor,
    cfg: dict
) -> torch.Tensor:
    
    device = torch.device(cfg.get("device", "cpu"))
    data = ToUndirected()(data).to(device)

    # 1) Estrai numero di utenti e di prodotti dal grafo eterogeneo
    # Supponiamo che i nodi di tipo "client" vadano da 0 a N_users-1
    # e i nodi di tipo "product" vadano da 0 a N_items-1 internamente a HeteroData.
    num_users = data["client"].num_nodes
    num_items = data["product"].num_nodes

    # 2) Costruisci edge_index bipartito (user_idx, item_idx+offset)
    # Prendiamo tutte le interazioni client → product (es: "buys", "adds", "removes"), e usiamo solo quelle:
    edge_types_to_use = [
        ("client", "buys", "product"),
        ("client", "adds", "product"),
        ("client", "removes", "product"),
    ]
    # Se una relazione non esiste (edge_index mancante), consideriamo un tensor vuoto.
    all_edges = []
    for et in edge_types_to_use:
        if et in data.edge_types:
            eidx = data[et].edge_index  # shape [2, E_et]
            all_edges.append(eidx)
        else:
            all_edges.append(torch.empty((2, 0), dtype=torch.long, device=device))

    if len(all_edges) > 1:
        edge_index_cat = torch.cat(all_edges, dim=1)  # [2, sum_E]
    else:
        edge_index_cat = all_edges[0]

    # In HeteroData, per “client → product”, i valori in edge_index_cat[1] sono già
    # indici [0..num_items‐1]. Per LightGCN dobbiamo sommarvi un offset = num_users.
    # L’offset va applicato solo alla seconda riga (destinazione “product”).
    user_idx = edge_index_cat[0]                # range [0..num_users-1]
    prod_idx = edge_index_cat[1] + num_users     # range [num_users..num_users+num_items-1]
    edge_index_lgcn = torch.stack([user_idx, prod_idx], dim=0)  # [2, E_tot]

    # 3) Portare i target su device
    y_churn    = y_churn.to(device)
    y_category = y_category.to(device)
    y_sku      = y_sku.to(device)

    # 4) Costruisci modello LightGCN + multitask
    model = FullLightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=cfg["embedding_dim"],
        num_layers=cfg["num_layers"],
        num_categories=y_category.size(1),
        num_skus=y_sku.size(1)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]))
    loss_fn = nn.BCELoss()

    best_loss = float("inf")
    patience = cfg.get("patience", 10)
    patience_counter = 0
    best_state = None

    # 5) Loop di training
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        churn_pred, cat_pred, sku_pred, _ = model(edge_index_lgcn)

        # churn_pred: [num_users, 1], y_churn: [num_users, 1] o [num_users]
        loss = (
            loss_fn(churn_pred.view(-1),   y_churn.view(-1)) +
            loss_fn(cat_pred,               y_category) +
            loss_fn(sku_pred,               y_sku)
        )
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"[Epoch {epoch}/{cfg['epochs']}] Loss: {loss.item():.4f}")

        # Early stopping
        """if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break"""

        # Libera memoria intermedia
        del churn_pred, cat_pred, sku_pred, loss
        torch.cuda.empty_cache()


    # 6) Estrai embedding finale utenti
    model.eval()
    with torch.no_grad():
        user_emb, _ = model.lightgcn(edge_index_lgcn)
        # user_emb: [num_users, embedding_dim] su device
    user_emb = user_emb.cpu()
    torch.cuda.empty_cache()

    return user_emb

"""import os
import sys
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

# ----------------------------------
# 1) LightGCN with baseline projection
# ----------------------------------
class LightGCN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        num_layers: int,
        baseline_embs: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_nodes = num_users + num_items
        self.num_layers = num_layers

        # target embedding space
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

        # if baseline given, insert a projection layer to init user_emb
        if baseline_embs is not None:
           print(f"[LightGCN] Initializing from baseline_embs of shape {baseline_embs.shape}")
    # 1) Create projection on same device as baseline_embs
           self.user_proj = nn.Linear(baseline_embs.size(1), embedding_dim, bias=False).to(baseline_embs.device)
           nn.init.xavier_uniform_(self.user_proj.weight)

    # 2) Project the baseline vectors and seed user_emb
        with torch.no_grad():
         projected = self.user_proj(baseline_embs)        # (num_users, embedding_dim), on baseline_embs.device
        # Copy back to CPU user_emb weights
         self.user_emb.weight.copy_(projected.cpu())
         print("[LightGCN] user_emb weights initialized from baseline projection")


        # random init for items
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, edge_index: torch.Tensor):
        # stack user+item
        embeddings = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        all_embs = [embeddings]

        # make undirected
        edge_index_full = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        row, col = edge_index_full
        deg = torch.bincount(row, minlength=self.num_nodes).float()
        deg_inv_sqrt = deg.pow(-0.5); deg_inv_sqrt[deg==0]=0.0

        for _ in range(self.num_layers):
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            out = torch.zeros_like(embeddings)
            out = out.index_add_(0, row, embeddings[col] * norm.unsqueeze(1))
            embeddings = out
            all_embs.append(embeddings)

        final_emb = torch.stack(all_embs, dim=1).mean(dim=1)
        return final_emb[:self.num_users], final_emb[self.num_users:]

# ----------------------------------
# 2) Multitask heads
# ----------------------------------
class MultiTaskHeads(nn.Module):
    def __init__(self, embedding_dim: int, num_categories: int, num_skus: int):
        super().__init__()
        self.churn_head = nn.Sequential(
            nn.Linear(embedding_dim, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
        self.category_head = nn.Sequential(
            nn.Linear(embedding_dim, 128), nn.ReLU(),
            nn.Linear(128, num_categories), nn.Sigmoid()
        )
        self.sku_head = nn.Sequential(
            nn.Linear(embedding_dim, 128), nn.ReLU(),
            nn.Linear(128, num_skus), nn.Sigmoid()
        )

    def forward(self, user_emb: torch.Tensor):
        return (
            self.churn_head(user_emb),
            self.category_head(user_emb),
            self.sku_head(user_emb),
        )

# ----------------------------------
# 3) Full model
# ----------------------------------
class FullLightGCN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        num_layers: int,
        num_categories: int,
        num_skus: int,
        baseline_embs: torch.Tensor | None = None,
    ):
        super().__init__()
        self.lightgcn = LightGCN(
            num_users, num_items, embedding_dim, num_layers, baseline_embs
        )
        self.heads = MultiTaskHeads(embedding_dim, num_categories, num_skus)

    def forward(self, edge_index: torch.Tensor):
        user_emb, _ = self.lightgcn(edge_index)
        churn, cat, sku = self.heads(user_emb)
        return churn, cat, sku, user_emb

# ----------------------------------
# 4) Training function
# ----------------------------------
def train_lightgcn(
    data: HeteroData,
    y_churn: torch.Tensor,
    y_category: torch.Tensor,
    y_sku: torch.Tensor,
    cfg: dict,
    baseline_embs: torch.Tensor,
) -> torch.Tensor:
    device = torch.device(cfg.get("device", "cpu"))
    data = ToUndirected()(data).to(device)

    # build bipartite edge_index
    num_users = data["client"].num_nodes
    num_items = data["product"].num_nodes
    edge_list = []
    for et in [("client","buys","product"),("client","adds","product"),("client","removes","product")]:
        e = data[et].edge_index if et in data.edge_types else torch.empty((2,0),dtype=torch.long,device=device)
        edge_list.append(e)
    edge_index_cat = torch.cat(edge_list, dim=1)
    edge_index_lgcn = torch.stack([
        edge_index_cat[0],
        edge_index_cat[1] + num_users
    ], dim=0)

    # move targets
    y_churn    = y_churn.to(device)
    y_category = y_category.to(device)
    y_sku      = y_sku.to(device)

    # build model with baseline
    model = FullLightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=cfg["embedding_dim"],
        num_layers=cfg["num_layers"],
        num_categories=y_category.size(1),
        num_skus=y_sku.size(1),
        baseline_embs=baseline_embs.to(device).float(),
    ).to(device)

    # optimizer only for projection + heads
    params_to_opt = list(model.lightgcn.user_proj.parameters()) + list(model.heads.parameters())
    optimizer = torch.optim.Adagrad(params_to_opt, lr=float(cfg["lr"]))
    loss_fn = nn.BCELoss()

    best_loss, patience_counter = float("inf"), 0
    #patience = cfg.get("patience", 10)

    for epoch in range(1, cfg["epochs"]+1):
        model.train(); optimizer.zero_grad()
        churn_pred, cat_pred, sku_pred, _ = model(edge_index_lgcn)
        loss = (
            loss_fn(churn_pred.view(-1), y_churn.view(-1)) +
            loss_fn(cat_pred, y_category) +
            loss_fn(sku_pred, y_sku)
        )
        loss.backward(); optimizer.step()

        if epoch % 10 == 0:
            print(f"[{epoch}/{cfg['epochs']}] Loss: {loss.item():.4f}")

        # early stop
         if loss.item() < best_loss:
            best_loss = loss.item(); patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stop at epoch {epoch}")
                break

    # final user embeddings
    model.eval()
    with torch.no_grad():
        user_emb, _ = model.lightgcn(edge_index_lgcn)
    return user_emb.cpu()
"""
