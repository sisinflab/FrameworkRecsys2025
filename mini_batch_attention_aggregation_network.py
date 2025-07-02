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
from torch.utils.data import TensorDataset, DataLoader # Added for mini-batching

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
        # If x is [B, Features], proj makes it [B, H].
        # MHA expects [B, SeqLen, H] or [SeqLen, B, H].
        # Assuming we want to attend over a "sequence" of 1 item per client from the input.
        # This means the current setup applies attention trivially if input x is 2D.
        # If x were [B, M, Features_in], proj would be [B, M, H_out]
        
        # Original code implies x is [B, M, input_dim] for MHA
        # However, if load_and_concat_embeddings gives [N_common, total_dim],
        # and it's passed directly (or in batches), then M (sequence length) for MHA is effectively 1
        # after potential unsqueezing if not already 3D.
        # Let's assume the input x to this module will be [Batch, Features]
        # and we make it [Batch, 1, Features] for MHA.

        if x.ndim == 2: # Input is [Batch, Features]
            x = x.unsqueeze(1) # Make it [Batch, 1, Features] for sequence length of 1

        h = self.proj(x)                  # [B, 1, H] if x was [B,1,input_dim], or [B,M,H] if x was [B,M,input_dim]
        attn_out, _ = self.attn(h, h, h)  # [B, 1, H] or [B,M,H]
        return attn_out.mean(dim=1)       # [B, H] (aggregates over sequence dimension M)

# -----------------------------
# 3) Modello completo
# -----------------------------
class AttentionMultiTaskModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        # num_sources: int, # This parameter is not used by the model currently
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
    full_embeddings: Tensor, # Renamed to signify it's the full dataset
    full_y_churn: Tensor,
    full_y_category: Tensor,
    full_y_sku: Tensor,
    cfg: dict
) -> Tensor:
    device = torch.device(cfg.get("device", "cpu"))
    batch_size = cfg["batch_size"]

    # Create TensorDataset and DataLoader for mini-batch training
    dataset = TensorDataset(full_embeddings, full_y_churn, full_y_category, full_y_sku)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())

    # Determine input_dim from the first batch or full_embeddings
    # It's safer to get it from full_embeddings before it's batched if it's consistent
    input_dim = full_embeddings.size(-1)
    num_categories = full_y_category.size(1)
    num_skus = full_y_sku.size(1)

    model = AttentionMultiTaskModel(
        input_dim=input_dim,
        # num_sources=full_embeddings.size(1), # This was likely an error if full_embeddings is 2D [N, D]
                                              # If it was meant to be sequence length, it should come from 3D data.
                                              # Not critical as it's unused in AttentionMultiTaskModel.
        attention_dim=cfg["attention_dim"],
        attn_heads=cfg["attn_heads"],
        num_categories=num_categories,
        num_skus=num_skus,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    loss_fn   = nn.BCELoss()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        num_batches_processed = 0

        for batch_embeddings, batch_y_churn, batch_y_category, batch_y_sku in train_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_y_churn    = batch_y_churn.to(device)
            batch_y_category = batch_y_category.to(device)
            batch_y_sku      = batch_y_sku.to(device)

            optimizer.zero_grad()
            churn_pred, cat_pred, sku_pred, _ = model(batch_embeddings)
            
            loss = (
                loss_fn(churn_pred,    batch_y_churn)  +
                loss_fn(cat_pred,      batch_y_category) +
                loss_fn(sku_pred,      batch_y_sku)
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches_processed += 1

        if epoch == 3:
            for p in model.heads.parameters():
                p.requires_grad = False
            print(f">>> Freeze delle teste multitask dopo epoca {epoch}")

        avg_epoch_loss = epoch_loss / num_batches_processed if num_batches_processed > 0 else 0
        if epoch == 1 or epoch % 10 == 0 or epoch == cfg["epochs"]:
            print(f"[Epoch {epoch}/{cfg['epochs']}] loss: {avg_epoch_loss:.4f}")

    # For final embedding extraction, process the full dataset (can also be batched if very large)
    model.eval()
    all_user_embs = []
    # Use a DataLoader for evaluation to handle potentially large datasets without OOM
    # No need to shuffle for evaluation
    eval_dataset = TensorDataset(full_embeddings) # Only need embeddings for this
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    with torch.no_grad():
        for batch_eval_embeddings_tuple in eval_loader:
            batch_eval_embeddings = batch_eval_embeddings_tuple[0].to(device)
            _, _, _, user_emb_batch = model(batch_eval_embeddings)
            all_user_embs.append(user_emb_batch.cpu())
    
    final_user_emb = torch.cat(all_user_embs, dim=0)
    return final_user_emb

# -----------------------------
# 5) Loader e concatenazione orizzontale di più file npy
# -----------------------------
def load_and_concat_embeddings(folder: str):
    emb_files = sorted(glob.glob(os.path.join(folder, 'embedding*.npy')))
    id_files  = sorted(glob.glob(os.path.join(folder, 'client_ids*.npy')))
    assert len(emb_files) == len(id_files), "Numero embedding != client_ids"

    emb_arrs = [np.load(f) for f in emb_files]
    id_arrs  = [np.load(f) for f in id_files]

    dims = [arr.shape[1] for arr in emb_arrs]
    # if len(set(dims)) != len(dims): # This check seems incorrect, should be if there are multiple *different* non-problematic dims
        # print(f"Embedding dimensions per fonte: {dims}") # This is more of an info message
    print(f"Embedding dimensions per fonte: {dims}")


    common = set(id_arrs[0])
    for ids in id_arrs[1:]: common &= set(ids)
    common_ids = sorted(list(common)) # Ensure it's a list for indexing
    N = len(common_ids)

    if N == 0:
        raise ValueError("No common client IDs found across embedding files.")

    aligned = []
    for arr, ids in zip(emb_arrs, id_arrs):
        id2idx = {cid: idx for idx, cid in enumerate(ids)}
        # Filter out ids not in common_ids to avoid KeyError if an id_arr doesn't have all common_ids
        # This shouldn't happen if common_ids is derived correctly, but good for robustness
        aligned_i = np.stack([arr[id2idx[cid]] for cid in common_ids if cid in id2idx], axis=0)
        if aligned_i.shape[0] != N:
             raise ValueError(f"Mismatch in aligned client count. Expected {N}, got {aligned_i.shape[0]}. Check ID consistency in files.")
        aligned.append(aligned_i)

    merged = np.concatenate(aligned, axis=1)
    print(f"Merging {len(common_ids)} clients: dims {dims} -> total {merged.shape[1]}")

    return torch.from_numpy(merged.astype(np.float32)), common_ids

# -----------------------------
# 6) Main 
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
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for training the attention model (default: 32)"
    )
    args = parser.parse_args()

    challenge_dir = Path(args.challenge_dir)
    embeddings_dir = Path(args.embeddings_dir)
    out_dir = Path(args.output_dir or args.embeddings_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading and concatenating embeddings...")
    embeddings, client_ids = load_and_concat_embeddings(str(embeddings_dir))
    logger.info(f"Loaded {embeddings.shape[0]} total client embeddings with dimension {embeddings.shape[1]}")


    data_dir_obj = DataDir(data_dir=challenge_dir)
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

    top_categories = train_target['category'].value_counts().nlargest(100).index.tolist()
    top_skus       = train_target['sku'].value_counts().nlargest(100).index.tolist()

    churn_calc = ChurnTargetCalculator()
    cat_calc   = PropensityTargetCalculator(
        TC_PropensityTasks.CATEGORY_PROPENSITY,
        top_categories
    )
    sku_calc   = SKUPropensityTargetCalculator(top_skus)

    logger.info("Calcolo target per ogni client...")
    y_churn_list    = []
    y_category_list = []
    y_sku_list      = []
    valid_client_ids_for_target = []

    client_ids_list = list(client_ids) if not isinstance(client_ids, list) else client_ids

    for cid_idx, cid in enumerate(client_ids_list):
        try:
            # --- MODIFICATION FOR CHURN TARGET VALUE ---
            churn_target_result = churn_calc.compute_target(cid, train_target)
            extracted_val = None
            if isinstance(churn_target_result, tuple):
                extracted_val = churn_target_result[0]
            else:
                extracted_val = churn_target_result

            # Ensure extracted_val is converted to a Python scalar float
            churn_scalar = None
            if hasattr(extracted_val, 'item'): # Works for numpy scalars, 0-dim arrays, or 1-element arrays/tensors
                churn_scalar = float(extracted_val.item())
            elif isinstance(extracted_val, (list, tuple)) and len(extracted_val) == 1: # For Python list/tuple like [0.0]
                churn_scalar = float(extracted_val[0])
            elif isinstance(extracted_val, (int, float, np.float_, np.int_)): # Handles various numpy and python scalar types
                churn_scalar = float(extracted_val)
            else:
                # If the format is still not recognized, raise an error to be caught by the except block
                raise ValueError(f"Unexpected format for churn value for client {cid}. Got: {extracted_val}, type: {type(extracted_val)}")
            # --- END MODIFICATION FOR CHURN TARGET VALUE ---

            cat_val = cat_calc.compute_target(cid, train_target) # Assuming this returns a 1D numpy array
            sku_val = sku_calc.compute_target(cid, train_target) # Assuming this returns a 1D numpy array

            y_churn_list.append(churn_scalar) # Append Python scalar
            # Convert to float32 numpy arrays before creating tensors for consistency
            y_category_list.append(torch.from_numpy(cat_val.astype(np.float32)))
            y_sku_list.append(torch.from_numpy(sku_val.astype(np.float32)))
            valid_client_ids_for_target.append(cid)
        except Exception as e:
            logger.warning(f"Could not compute target for client ID {cid} (index {cid_idx}): {e}. Skipping this client.")

    if not valid_client_ids_for_target:
        logger.error("No valid targets could be computed for any client. Aborting training.")
        logger.error("This often happens if target calculators consistently fail or return unexpected formats.")
        logger.error("Please check the warnings above for clues about issues within your target calculator(s).")
        return # Exit main if no data

    logger.info(f"Targets computed successfully for {len(valid_client_ids_for_target)} out of {len(client_ids_list)} clients.")

    if len(valid_client_ids_for_target) < len(client_ids_list):
        original_id_to_idx = {cid_val: i for i, cid_val in enumerate(client_ids_list)}
        indices_to_keep = [original_id_to_idx[cid_val] for cid_val in valid_client_ids_for_target]
        embeddings = embeddings[indices_to_keep]
        client_ids = valid_client_ids_for_target
        logger.info(f"Embeddings tensor reduced to shape: {embeddings.shape} to match clients with targets.")

    # y_churn_list now contains Python floats.
    # torch.tensor(y_churn_list) will be 1D [N_valid].
    # .unsqueeze(1) will make it [N_valid, 1]. This is the correct shape.
    y_churn = torch.tensor(y_churn_list, dtype=torch.float32).unsqueeze(1)

    # y_category_list and y_sku_list contain lists of 1D tensors.
    # torch.stack will create [N_valid, num_features] tensors.
    y_category = torch.stack(y_category_list) # .float() is not needed here if already float32 from_numpy
    y_sku = torch.stack(y_sku_list)           # .float() is not needed here

    logger.info(f"Target shapes: y_churn: {y_churn.shape}, y_category: {y_category.shape}, y_sku: {y_sku.shape}")
    logger.info(f"Embeddings shape before training: {embeddings.shape}")

    assert embeddings.shape[0] == y_churn.shape[0] == y_category.shape[0] == y_sku.shape[0], \
        "Mismatch between number of embeddings and target samples after filtering."

    cfg = {
        'device':        'cuda' if torch.cuda.is_available() else 'cpu',
        'lr':            1e-3,
        'epochs':        10,
        'attention_dim': 128,
        'attn_heads':    4,
        'batch_size':    args.batch_size
    }
    logger.info(f"Training configuration: {cfg}")

    final_emb = train_attention_model(
        embeddings, y_churn, y_category, y_sku, cfg
    )

    np.save(out_dir / 'client_embeddings_128.npy', final_emb.numpy())
    np.save(out_dir / 'client_ids_128.npy', np.array(client_ids))
    logger.info(f"Salvati {final_emb.shape[0]} embeddings in {out_dir}")

    validate_and_load_embeddings(
        input_dir=input_dir,
        embeddings_dir=out_dir,
        max_embedding_dim=MAX_EMBEDDING_DIM,
        disable_relevant_clients_check=False,
    )
    logger.info("Embedding validation completed successfully")

if __name__ == "__main__":
    main()
