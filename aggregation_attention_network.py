# attention_train.py
import os
import glob
from pathlib import Path
import argparse
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Validation import
from validator.validate import validate_and_load_embeddings
# Data splitting imports
from data_utils.data_dir import DataDir
from data_utils.constants import DAYS_IN_TARGET
from data_utils.split_data import DataSplitter
# Constants import
from training_pipeline.constants import MAX_EMBEDDING_DIM

# Target calculators
from src.preprocessing.target_calculators import (
    ChurnTargetCalculator,
    PropensityTargetCalculator,
    SKUPropensityTargetCalculator,
    PropensityTasks as TC_PropensityTasks,
)

# -----------------------------
# 1) Teste multitask 
# -----------------------------
class MultiTaskHeads(nn.Module):
    def __init__(self, embedding_dim: int, num_categories: int, num_skus: int):
        super().__init__()
        self.churn_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.category_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_categories),
            nn.Sigmoid()
        )
        self.sku_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_skus),
            nn.Sigmoid()
        )

    def forward(self, user_emb: Tensor):
        churn    = self.churn_head(user_emb)
        category = self.category_head(user_emb)
        sku      = self.sku_head(user_emb)
        return churn, category, sku

# -----------------------------
# 2) AttentionAggregator
# -----------------------------
class AttentionAggregator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.proj(x)                  # [B, M, H]
        attn_out, _ = self.attn(h, h, h)  # [B, M, H]
        return attn_out.mean(dim=1)       # [B, H]

# -----------------------------
# 3) Modello completo
# -----------------------------
class AttentionMultiTaskModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_sources: int,
        attention_dim: int,
        attn_heads: int,
        num_categories: int,
        num_skus: int,
    ):
        super().__init__()
        self.aggregator = AttentionAggregator(
            input_dim=input_dim,
            hidden_dim=attention_dim,
            num_heads=attn_heads
        )
        self.heads = MultiTaskHeads(
            embedding_dim=attention_dim,
            num_categories=num_categories,
            num_skus=num_skus
        )

    def forward(self, x: Tensor):
        uemb = self.aggregator(x)                    # [B, H]
        churn, cat, sku = self.heads(uemb)           # predizioni
        return churn, cat, sku, uemb

# -----------------------------
# 4) Funzione di training
# -----------------------------
def train_attention_model(
    embeddings: Tensor,
    y_churn: Tensor,
    y_category: Tensor,
    y_sku: Tensor,
    cfg: dict
) -> Tensor:
    device = torch.device(cfg.get("device", "cpu"))
    embeddings = embeddings.to(device)
    y_churn    = y_churn.to(device)
    y_category = y_category.to(device)
    y_sku      = y_sku.to(device)

    model = AttentionMultiTaskModel(
        input_dim=embeddings.size(-1),
        num_sources=embeddings.size(1),
        attention_dim=cfg["attention_dim"],
        attn_heads=cfg["attn_heads"],
        num_categories=y_category.size(1),
        num_skus=y_sku.size(1),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    loss_fn   = nn.BCELoss()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        churn_pred, cat_pred, sku_pred, _ = model(embeddings)
        loss = (
            loss_fn(churn_pred,    y_churn)  +
            loss_fn(cat_pred,      y_category) +
            loss_fn(sku_pred,      y_sku)
        )
        loss.backward()
        optimizer.step()

        if epoch == 3:
            for p in model.heads.parameters():
                p.requires_grad = False
            print(f">>> Freeze delle teste multitask dopo epoca {epoch}")

        if epoch == 1 or epoch % 10 == 0:
            print(f"[Epoch {epoch}/{cfg['epochs']}] loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        _, _, _, user_emb = model(embeddings)
    return user_emb.cpu()

# -----------------------------
# 5) Loader e concatenazione orizzontale di più file npy
# -----------------------------
def load_and_concat_embeddings(folder: str):
    """
    Cerca in `folder` file embedding_*.npy e client_ids_*.npy,
    allinea per id e restituisce:
    - merged: Tensor [N_common, total_dim] con concatenazione orizzontale
    - common_ids: list of ids length N_common
    """
    emb_files = sorted(glob.glob(os.path.join(folder, 'embedding*.npy')))
    id_files  = sorted(glob.glob(os.path.join(folder, 'client_ids*.npy')))
    assert len(emb_files) == len(id_files), "Numero embedding != client_ids"

    # Carica arrays
    emb_arrs = [np.load(f) for f in emb_files]    # list of (Ni, Di)
    id_arrs  = [np.load(f) for f in id_files]     # list of (Ni,)

    # Verifica dimensioni embedding coerenti per ciascuna fonte
    dims = [arr.shape[1] for arr in emb_arrs]
    if len(set(dims)) != len(dims):
        print(f"Embedding dimensions per fonte: {dims}")
    # Trova intersezione e ordina
    common = set(id_arrs[0])
    for ids in id_arrs[1:]: common &= set(ids)
    common_ids = sorted(common)
    N = len(common_ids)

    # Crea mappature id->indice e allinea
    aligned = []  # lista di array [N, Di]
    for arr, ids in zip(emb_arrs, id_arrs):
        id2idx = {cid: idx for idx, cid in enumerate(ids)}
        aligned_i = np.stack([arr[id2idx[cid]] for cid in common_ids], axis=0)
        aligned.append(aligned_i)

    # Concatenazione orizzontale
    merged = np.concatenate(aligned, axis=1)  # [N, sum(Di)]
    print(f"Merging {len(common_ids)} clients: dims {dims} -> total {merged.shape[1]}")

    return torch.from_numpy(merged.astype(np.float32)), common_ids

# -----------------------------
# 6) Main adattato con validazione
# -----------------------------
def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Train attention model on aligned embeddings and compute targets"
    )
    parser.add_argument(
        "--challenge-dir", type=str, required=True,
        help="Cartella radice dei dati di competizione (deve contenere 'input')"
    )
    parser.add_argument(
        "--embeddings-dir", type=str, required=True,
        help="Cartella contenente i file embedding_*.npy e client_ids_*.npy"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Dove salvare client_embeddings_128.npy e client_ids_128.npy"
    )
    args = parser.parse_args()

    # Directory di input dati di sfida e embeddings
    challenge_dir = Path(args.challenge_dir)
    embeddings_dir = Path(args.embeddings_dir)
    out_dir = Path(args.output_dir or args.embeddings_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 6.1) Allinea embeddings dai file separati
    embeddings, client_ids = load_and_concat_embeddings(str(embeddings_dir))

    # 6.2) Split per calcolare target su eventi
    data_dir_obj = DataDir(data_dir=challenge_dir)

    # 6.2) End date dai dati di purchase nel folder input
    input_dir = challenge_dir / 'input'
    df_buy = pd.read_parquet(input_dir / 'product_buy.parquet')
    end_date = pd.to_datetime(df_buy['timestamp'].max())

    splitter = DataSplitter(
        challenge_data_dir=data_dir_obj,
        days_in_target=DAYS_IN_TARGET,
        end_date=end_date
    )
    splitter.split()
    train_target = splitter.target_events['train_target']

    # 6.3) Top-k categories e skus
    top_categories = train_target['category'].value_counts().nlargest(100).index.tolist()
    top_skus       = train_target['sku'].value_counts().nlargest(100).index.tolist()

    # 6.4) Init calculators
    churn_calc = ChurnTargetCalculator()
    cat_calc   = PropensityTargetCalculator(
        TC_PropensityTasks.CATEGORY_PROPENSITY,
        top_categories
    )
    sku_calc   = SKUPropensityTargetCalculator(top_skus)

    # 6.5) Compute target tensors allineati
    logger.info("Calcolo target per ogni client...")
    y_churn = torch.tensor(
        [churn_calc.compute_target(cid, train_target)[0] for cid in client_ids],
        dtype=torch.float32
    ).unsqueeze(1)
    y_category = torch.stack([
        torch.from_numpy(cat_calc.compute_target(cid, train_target))
        for cid in client_ids
    ]).float()
    y_sku = torch.stack([
        torch.from_numpy(sku_calc.compute_target(cid, train_target))
        for cid in client_ids
    ]).float()

    # 6.6) Config training
    cfg = {
        'device':        'cuda' if torch.cuda.is_available() else 'cpu',
        'lr':            1e-3,
        'epochs':        10,
        'attention_dim': 128,
        'attn_heads':    4,
    }

    # 6.7) Train e salva risultati
    final_emb = train_attention_model(
        embeddings, y_churn, y_category, y_sku, cfg
    )

    np.save(out_dir / 'client_embeddings_128.npy', final_emb.numpy())
    np.save(out_dir / 'client_ids_128.npy', np.array(client_ids))
    logger.info(f"Salvati {final_emb.shape[0]} embeddings in {out_dir}")

    # 6.8) Validazione embeddings
    input_dir = challenge_dir / 'input'
    validate_and_load_embeddings(
        input_dir=input_dir,
        embeddings_dir=out_dir,
        max_embedding_dim=MAX_EMBEDDING_DIM,
        disable_relevant_clients_check=False,
    )
    logger.info("Embedding validation completed successfully")

if __name__ == "__main__":
    main()

"""

# attention_train.py - versione aggiornata con supporto per embedding di dimensioni diverse
import os
import glob
from pathlib import Path
import argparse
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor

# Validation import
from validator.validate import validate_and_load_embeddings
# Data splitting imports
from data_utils.data_dir import DataDir
from data_utils.constants import DAYS_IN_TARGET
from data_utils.split_data import DataSplitter
# Constants import
from training_pipeline.constants import MAX_EMBEDDING_DIM

# Target calculators
from src.preprocessing.target_calculators import (
    ChurnTargetCalculator,
    PropensityTargetCalculator,
    SKUPropensityTargetCalculator,
    PropensityTasks as TC_PropensityTasks,
)

# -----------------------------
# 1) Teste multitask 
# -----------------------------
class MultiTaskHeads(nn.Module):
    def __init__(self, embedding_dim: int, num_categories: int, num_skus: int):
        super().__init__()
        self.churn_head = nn.Sequential(
            nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.category_head = nn.Sequential(
            nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, num_categories), nn.Sigmoid()
        )
        self.sku_head = nn.Sequential(
            nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, num_skus), nn.Sigmoid()
        )

    def forward(self, user_emb: Tensor):
        return self.churn_head(user_emb), self.category_head(user_emb), self.sku_head(user_emb)

# -----------------------------
# 2) AttentionAggregator con input embedding di dimensioni diverse
# -----------------------------
class AttentionAggregator(nn.Module):
    def __init__(self, dims, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(d, hidden_dim) for d in dims])
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, M, ?] (non usato direttamente)
        parts = [proj(x[:, i]) for i, proj in enumerate(self.projs)]  # list of [B, H]
        stacked = torch.stack(parts, dim=1)  # [B, M, H]
        attn_out, _ = self.attn(stacked, stacked, stacked)
        return attn_out.mean(dim=1)

# -----------------------------
# 3) Modello completo
# -----------------------------
class AttentionMultiTaskModel(nn.Module):
    def __init__(
        self,
        dims,
        attention_dim: int,
        attn_heads: int,
        num_categories: int,
        num_skus: int,
    ):
        super().__init__()
        self.aggregator = AttentionAggregator(dims, attention_dim, attn_heads)
        self.heads = MultiTaskHeads(attention_dim, num_categories, num_skus)

    def forward(self, x: Tensor):
        uemb = self.aggregator(x)
        return *self.heads(uemb), uemb

# -----------------------------
# 4) Training
# -----------------------------
def train_attention_model(embeddings: Tensor, y_churn: Tensor, y_category: Tensor, y_sku: Tensor, cfg: dict, dims: list) -> Tensor:
    device = torch.device(cfg.get("device", "cpu"))
    embeddings = [e.to(device) for e in embeddings]
    y_churn = y_churn.to(device)
    y_category = y_category.to(device)
    y_sku = y_sku.to(device)

    model = AttentionMultiTaskModel(
        dims=dims,
        attention_dim=cfg["attention_dim"],
        attn_heads=cfg["attn_heads"],
        num_categories=y_category.size(1),
        num_skus=y_sku.size(1),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.BCELoss()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        churn_pred, cat_pred, sku_pred, _ = model(embeddings)
        loss = loss_fn(churn_pred, y_churn) + loss_fn(cat_pred, y_category) + loss_fn(sku_pred, y_sku)
        loss.backward()
        optimizer.step()

        if epoch == 3:
            for p in model.heads.parameters():
                p.requires_grad = False
            print(f">>> Freeze delle teste multitask dopo epoca {epoch}")

        if epoch == 1 or epoch % 10 == 0:
            print(f"[Epoch {epoch}/{cfg['epochs']}] loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        _, _, _, user_emb = model(embeddings)
    return user_emb.cpu()

# -----------------------------
# 5) Caricamento embedding con reshape in 3D
# -----------------------------
def load_and_stack_embeddings(folder: str):
    emb_files = sorted(glob.glob(os.path.join(folder, 'embedding*.npy')))
    id_files  = sorted(glob.glob(os.path.join(folder, 'client_ids*.npy')))
    assert len(emb_files) == len(id_files), "Numero embedding != client_ids"

    emb_arrs = [np.load(f) for f in emb_files]
    id_arrs  = [np.load(f) for f in id_files]

    dims = [arr.shape[1] for arr in emb_arrs]
    common = set(id_arrs[0])
    for ids in id_arrs[1:]:
        common &= set(ids)
    common_ids = sorted(common)

    aligned = []
    for arr, ids in zip(emb_arrs, id_arrs):
        id2idx = {cid: idx for idx, cid in enumerate(ids)}
        aligned_i = np.stack([arr[id2idx[cid]] for cid in common_ids], axis=0)
        aligned.append(aligned_i)

    merged = np.concatenate(aligned, axis=1)  # [N, sum(Di)]
    splits = np.split(merged, np.cumsum(dims)[:-1], axis=1)
    splits = [torch.tensor(s, dtype=torch.float32) for s in splits]
    return splits, common_ids, dims

# -----------------------------
# 6) Main
# -----------------------------
def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-dir", type=str, required=True)
    parser.add_argument("--embeddings-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    challenge_dir = Path(args.challenge_dir)
    embeddings_dir = Path(args.embeddings_dir)
    out_dir = Path(args.output_dir or args.embeddings_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings, client_ids, dims = load_and_stack_embeddings(str(embeddings_dir))

    input_dir = challenge_dir / 'input'
    df_buy = pd.read_parquet(input_dir / 'product_buy.parquet')
    end_date = pd.to_datetime(df_buy['timestamp'].max())

    splitter = DataSplitter(
        challenge_data_dir=DataDir(data_dir=challenge_dir),
        days_in_target=DAYS_IN_TARGET,
        end_date=end_date
    )
    splitter.split()
    train_target = splitter.target_events['train_target']

    top_categories = train_target['category'].value_counts().nlargest(100).index.tolist()
    top_skus = train_target['sku'].value_counts().nlargest(100).index.tolist()

    churn_calc = ChurnTargetCalculator()
    cat_calc = PropensityTargetCalculator(TC_PropensityTasks.CATEGORY_PROPENSITY, top_categories)
    sku_calc = SKUPropensityTargetCalculator(top_skus)

    logger.info("Calcolo target per ogni client...")
    y_churn = torch.tensor([
        churn_calc.compute_target(cid, train_target)[0] for cid in client_ids
    ], dtype=torch.float32).unsqueeze(1)
    y_category = torch.stack([
        torch.from_numpy(cat_calc.compute_target(cid, train_target)) for cid in client_ids
    ]).float()
    y_sku = torch.stack([
        torch.from_numpy(sku_calc.compute_target(cid, train_target)) for cid in client_ids
    ]).float()

    cfg = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr': 1e-3,
        'epochs': 10,
        'attention_dim': 128,
        'attn_heads': 4,
    }
    

    final_emb = train_attention_model(embeddings, y_churn, y_category, y_sku, cfg, dims)
    np.save(out_dir / 'client_embeddings_comb.npy', final_emb.numpy())
    np.save(out_dir / 'client_ids_comb.npy', np.array(client_ids))
    logger.info(f"Salvati {final_emb.shape[0]} embeddings in {out_dir}")

    validate_and_load_embeddings(
        input_dir=input_dir,
        embeddings_dir=out_dir,
        max_embedding_dim=MAX_EMBEDDING_DIM,
        disable_relevant_clients_check=False
    )
    logger.info("Embedding validation completed successfully")

if __name__ == "__main__":
    main()"""

"""# attention_train.py - versione aggiornata con supporto per embedding di dimensioni diverse
import os
import glob
from pathlib import Path
import argparse
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor

# Validation import
from validator.validate import validate_and_load_embeddings
# Data splitting imports
from data_utils.data_dir import DataDir
from data_utils.constants import DAYS_IN_TARGET
from data_utils.split_data import DataSplitter
# Constants import
from training_pipeline.constants import MAX_EMBEDDING_DIM

# Target calculators
from src.preprocessing.target_calculators import (
    ChurnTargetCalculator,
    PropensityTargetCalculator,
    SKUPropensityTargetCalculator,
    PropensityTasks as TC_PropensityTasks,
)

# -----------------------------
# 1) Teste multitask 
# -----------------------------
class MultiTaskHeads(nn.Module):
    def __init__(self, embedding_dim: int, num_categories: int, num_skus: int):
        super().__init__()
        self.churn_head = nn.Sequential(
            nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.category_head = nn.Sequential(
            nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, num_categories), nn.Sigmoid()
        )
        self.sku_head = nn.Sequential(
            nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, num_skus), nn.Sigmoid()
        )

    def forward(self, user_emb: Tensor):
        return self.churn_head(user_emb), self.category_head(user_emb), self.sku_head(user_emb)

# -----------------------------
# 2) AttentionAggregator con input embedding di dimensioni diverse
# -----------------------------
class AttentionAggregator(nn.Module):
    def __init__(self, dims, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(d, hidden_dim) for d in dims])
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, M, ?] (non usato direttamente)
        parts = [proj(x[:, i]) for i, proj in enumerate(self.projs)]  # list of [B, H]
        stacked = torch.stack(parts, dim=1)  # [B, M, H]
        attn_out, _ = self.attn(stacked, stacked, stacked)
        return attn_out.mean(dim=1)

# -----------------------------
# 3) Modello completo
# -----------------------------
class AttentionMultiTaskModel(nn.Module):
    def __init__(
        self,
        dims,
        attention_dim: int,
        attn_heads: int,
        num_categories: int,
        num_skus: int,
    ):
        super().__init__()
        self.aggregator = AttentionAggregator(dims, attention_dim, attn_heads)
        self.heads = MultiTaskHeads(attention_dim, num_categories, num_skus)

    def forward(self, x: Tensor):
        uemb = self.aggregator(x)
        return *self.heads(uemb), uemb

# -----------------------------
# 4) Training
# -----------------------------
def train_attention_model(embeddings: Tensor, y_churn: Tensor, y_category: Tensor, y_sku: Tensor, cfg: dict, dims: list) -> Tensor:
    device = torch.device(cfg.get("device", "cpu"))
    embeddings = embeddings.to(device)
    y_churn = y_churn.to(device)
    y_category = y_category.to(device)
    y_sku = y_sku.to(device)

    model = AttentionMultiTaskModel(
        dims=dims,
        attention_dim=cfg["attention_dim"],
        attn_heads=cfg["attn_heads"],
        num_categories=y_category.size(1),
        num_skus=y_sku.size(1),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.BCELoss()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        churn_pred, cat_pred, sku_pred, _ = model(embeddings)
        loss = loss_fn(churn_pred, y_churn) + loss_fn(cat_pred, y_category) + loss_fn(sku_pred, y_sku)
        loss.backward()
        optimizer.step()

        if epoch == 3:
            for p in model.heads.parameters():
                p.requires_grad = False
            print(f">>> Freeze delle teste multitask dopo epoca {epoch}")

        if epoch == 1 or epoch % 10 == 0:
            print(f"[Epoch {epoch}/{cfg['epochs']}] loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        _, _, _, user_emb = model(embeddings)
    return user_emb.cpu()

# -----------------------------
# 5) Caricamento embedding con reshape in 3D
# -----------------------------
def load_and_stack_embeddings(folder: str):
    emb_files = sorted(glob.glob(os.path.join(folder, 'embedding*.npy')))
    id_files  = sorted(glob.glob(os.path.join(folder, 'client_ids*.npy')))
    assert len(emb_files) == len(id_files), "Numero embedding != client_ids"

    emb_arrs = [np.load(f) for f in emb_files]
    id_arrs  = [np.load(f) for f in id_files]

    dims = [arr.shape[1] for arr in emb_arrs]
    common = set(id_arrs[0])
    for ids in id_arrs[1:]:
        common &= set(ids)
    common_ids = sorted(common)

    aligned = []
    for arr, ids in zip(emb_arrs, id_arrs):
        id2idx = {cid: idx for idx, cid in enumerate(ids)}
        aligned_i = np.stack([arr[id2idx[cid]] for cid in common_ids], axis=0)
        aligned.append(aligned_i)

    merged = np.concatenate(aligned, axis=1)  # [N, sum(Di)]
    splits = np.split(merged, np.cumsum(dims)[:-1], axis=1)
    stacked = np.stack(splits, axis=1)  # [N, M, D_var]
    print(f"Merging {len(common_ids)} clients: dims {dims} → stacked shape {stacked.shape}")
    return torch.from_numpy(stacked.astype(np.float32)), common_ids, dims

# -----------------------------
# 6) Main
# -----------------------------
def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-dir", type=str, required=True)
    parser.add_argument("--embeddings-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    challenge_dir = Path(args.challenge_dir)
    embeddings_dir = Path(args.embeddings_dir)
    out_dir = Path(args.output_dir or args.embeddings_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings, client_ids, dims = load_and_stack_embeddings(str(embeddings_dir))

    input_dir = challenge_dir / 'input'
    df_buy = pd.read_parquet(input_dir / 'product_buy.parquet')
    end_date = pd.to_datetime(df_buy['timestamp'].max())

    splitter = DataSplitter(
        challenge_data_dir=DataDir(data_dir=challenge_dir),
        days_in_target=DAYS_IN_TARGET,
        end_date=end_date
    )
    splitter.split()
    train_target = splitter.target_events['train_target']

    top_categories = train_target['category'].value_counts().nlargest(100).index.tolist()
    top_skus = train_target['sku'].value_counts().nlargest(100).index.tolist()

    churn_calc = ChurnTargetCalculator()
    cat_calc = PropensityTargetCalculator(TC_PropensityTasks.CATEGORY_PROPENSITY, top_categories)
    sku_calc = SKUPropensityTargetCalculator(top_skus)

    logger.info("Calcolo target per ogni client...")
    y_churn = torch.tensor([
        churn_calc.compute_target(cid, train_target)[0] for cid in client_ids
    ], dtype=torch.float32).unsqueeze(1)
    y_category = torch.stack([
        torch.from_numpy(cat_calc.compute_target(cid, train_target)) for cid in client_ids
    ]).float()
    y_sku = torch.stack([
        torch.from_numpy(sku_calc.compute_target(cid, train_target)) for cid in client_ids
    ]).float()

    cfg = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr': 1e-3,
        'epochs': 10,
        'attention_dim': 128,
        'attn_heads': 4,
    }

    final_emb = train_attention_model(embeddings, y_churn, y_category, y_sku, cfg, dims)
    np.save(out_dir / 'client_embeddings_128.npy', final_emb.numpy())
    np.save(out_dir / 'client_ids_128.npy', np.array(client_ids))
    logger.info(f"Salvati {final_emb.shape[0]} embeddings in {out_dir}")

    validate_and_load_embeddings(
        input_dir=input_dir,
        embeddings_dir=out_dir,
        max_embedding_dim=MAX_EMBEDDING_DIM,
        disable_relevant_clients_check=False
    )
    logger.info("Embedding validation completed successfully")

if __name__ == "__main__":
    main()
# attention_train.py - versione aggiornata con supporto per embedding di dimensioni diverse
import os
import glob
from pathlib import Path
import argparse
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor

# Validation import
from validator.validate import validate_and_load_embeddings
# Data splitting imports
from data_utils.data_dir import DataDir
from data_utils.constants import DAYS_IN_TARGET
from data_utils.split_data import DataSplitter
# Constants import
from training_pipeline.constants import MAX_EMBEDDING_DIM

# Target calculators
from src.preprocessing.target_calculators import (
    ChurnTargetCalculator,
    PropensityTargetCalculator,
    SKUPropensityTargetCalculator,
    PropensityTasks as TC_PropensityTasks,
)

# -----------------------------
# 1) Teste multitask 
# -----------------------------
class MultiTaskHeads(nn.Module):
    def __init__(self, embedding_dim: int, num_categories: int, num_skus: int):
        super().__init__()
        self.churn_head = nn.Sequential(
            nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.category_head = nn.Sequential(
            nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, num_categories), nn.Sigmoid()
        )
        self.sku_head = nn.Sequential(
            nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, num_skus), nn.Sigmoid()
        )

    def forward(self, user_emb: Tensor):
        return self.churn_head(user_emb), self.category_head(user_emb), self.sku_head(user_emb)

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionAggregator(nn.Module):
    def __init__(self, input_dims, hidden_dim, num_sources):
        super().__init__()
        self.num_sources = num_sources
        assert len(input_dims) == num_sources, "input_dims deve avere num_sources elementi"
        self.projs = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in input_dims
        ])
        self.att_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_list):
     
        parts = [proj(x_list[i]) for i, proj in enumerate(self.projs)]  # [B, H] per ognuno
        stacked = torch.stack(parts, dim=1)  # [B, M, H]
        scores = self.att_mlp(stacked)  # [B, M, 1]
        weights = torch.softmax(scores, dim=1)  # [B, M, 1]
        output = torch.sum(weights * stacked, dim=1)  # [B, H]
        return output



# -----------------------------
# 3) Modello completo
# -----------------------------
class AttentionMultiTaskModel(nn.Module):
    def __init__(
        self,
        dims,
        attention_dim: int,
        attn_heads: int,
        num_categories: int,
        num_skus: int,
    ):
        super().__init__()
        self.aggregator = AttentionAggregator(input_dims=dims, hidden_dim=attention_dim, num_sources=attn_heads)
        self.heads = MultiTaskHeads(attention_dim, num_categories, num_skus)

    def forward(self, x):
        uemb = self.aggregator(x)
        return *self.heads(uemb), uemb


# -----------------------------
# 4) Training
# -----------------------------
def train_attention_model(embeddings: Tensor, y_churn: Tensor, y_category: Tensor, y_sku: Tensor, cfg: dict, dims: list) -> Tensor:
    device = torch.device(cfg.get("device", "cpu"))
    embeddings = [e.to(device) for e in embeddings]
    y_churn = y_churn.to(device)
    y_category = y_category.to(device)
    y_sku = y_sku.to(device)

    model = AttentionMultiTaskModel(
        dims=dims,
        attention_dim=cfg["attention_dim"],
        attn_heads=cfg["attn_heads"],
        num_categories=y_category.size(1),
        num_skus=y_sku.size(1),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.BCELoss()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        churn_pred, cat_pred, sku_pred, _ = model(embeddings)
        loss = loss_fn(churn_pred, y_churn) + loss_fn(cat_pred, y_category) + loss_fn(sku_pred, y_sku)
        loss.backward()
        optimizer.step()

        if epoch == 3:
            for p in model.heads.parameters():
                p.requires_grad = False
            print(f">>> Freeze delle teste multitask dopo epoca {epoch}")

        if epoch == 1 or epoch % 10 == 0:
            print(f"[Epoch {epoch}/{cfg['epochs']}] loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        _, _, _, user_emb = model(embeddings)
    return user_emb.cpu()

# -----------------------------
# 5) Caricamento embedding con reshape in 3D
# -----------------------------
def load_and_stack_embeddings(folder: str):
    emb_files = sorted(glob.glob(os.path.join(folder, 'embedding*.npy')))
    id_files  = sorted(glob.glob(os.path.join(folder, 'client_ids*.npy')))
    assert len(emb_files) == len(id_files), "Numero embedding != client_ids"

    emb_arrs = [np.load(f) for f in emb_files]
    id_arrs  = [np.load(f) for f in id_files]

    dims = [arr.shape[1] for arr in emb_arrs]
    common = set(id_arrs[0])
    for ids in id_arrs[1:]:
        common &= set(ids)
    common_ids = sorted(common)

    aligned = []
    for arr, ids in zip(emb_arrs, id_arrs):
        id2idx = {cid: idx for idx, cid in enumerate(ids)}
        aligned_i = np.stack([arr[id2idx[cid]] for cid in common_ids], axis=0)
        aligned.append(aligned_i)

    merged = np.concatenate(aligned, axis=1)  # [N, sum(Di)]
    splits = np.split(merged, np.cumsum(dims)[:-1], axis=1)
    splits = [torch.tensor(s, dtype=torch.float32) for s in splits]
    return splits, common_ids, dims

# -----------------------------
# 6) Main
# -----------------------------
def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-dir", type=str, required=True)
    parser.add_argument("--embeddings-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    challenge_dir = Path(args.challenge_dir)
    embeddings_dir = Path(args.embeddings_dir)
    out_dir = Path(args.output_dir or args.embeddings_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings, client_ids, dims = load_and_stack_embeddings(str(embeddings_dir))

    input_dir = challenge_dir / 'input'
    df_buy = pd.read_parquet(input_dir / 'product_buy.parquet')
    end_date = pd.to_datetime(df_buy['timestamp'].max())

    splitter = DataSplitter(
        challenge_data_dir=DataDir(data_dir=challenge_dir),
        days_in_target=DAYS_IN_TARGET,
        end_date=end_date
    )
    splitter.split()
    train_target = splitter.target_events['train_target']

    top_categories = train_target['category'].value_counts().nlargest(100).index.tolist()
    top_skus = train_target['sku'].value_counts().nlargest(100).index.tolist()

    churn_calc = ChurnTargetCalculator()
    cat_calc = PropensityTargetCalculator(TC_PropensityTasks.CATEGORY_PROPENSITY, top_categories)
    sku_calc = SKUPropensityTargetCalculator(top_skus)

    logger.info("Calcolo target per ogni client...")
    y_churn = torch.tensor([
        churn_calc.compute_target(cid, train_target)[0] for cid in client_ids
    ], dtype=torch.float32).unsqueeze(1)
    y_category = torch.stack([
        torch.from_numpy(cat_calc.compute_target(cid, train_target)) for cid in client_ids
    ]).float()
    y_sku = torch.stack([
        torch.from_numpy(sku_calc.compute_target(cid, train_target)) for cid in client_ids
    ]).float()

    cfg = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr': 1e-3,
        'epochs': 10,
        'attention_dim': 128,
        'attn_heads': 4,
    }
    

    final_emb = train_attention_model(embeddings, y_churn, y_category, y_sku, cfg, dims)
    np.save(out_dir / 'client_embeddings_comb.npy', final_emb.numpy())
    np.save(out_dir / 'client_ids_comb.npy', np.array(client_ids))
    logger.info(f"Salvati {final_emb.shape[0]} embeddings in {out_dir}")

    validate_and_load_embeddings(
        input_dir=input_dir,
        embeddings_dir=out_dir,
        max_embedding_dim=MAX_EMBEDDING_DIM,
        disable_relevant_clients_check=False
    )
    logger.info("Embedding validation completed successfully")

if __name__ == "__main__":
    main() """
