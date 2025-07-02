#!/usr/bin/env python3
import numpy as np
from pathlib import Path

# 1) Imposta i path
base_dir = Path(__file__).resolve().parent / "outputs"

# Definizione dei modelli: (sottocartella, ids_file, emb_file)
models = [
    ("lightgcn13", "client_ids.npy", "embeddings.npy"),       # lightgcn13 in outputs/lightgcn13
    ("", "client_ids.npy", "embeddings.npy"),                 # baseline direttamente in outputs
]

# 2) Carica i primi IDs come reference e tutte le embeddings
ids_ref = None
emb_list = []

for model_subdir, ids_fname, emb_fname in models:
    m_dir = base_dir / model_subdir  # "" -> base_dir stesso
    ids_path = m_dir / ids_fname
    emb_path = m_dir / emb_fname

    # Caricamento
    ids = np.load(ids_path, allow_pickle=True)
    emb = np.load(emb_path, allow_pickle=True)

    # Inizializza o confronta gli IDs
    if ids_ref is None:
        ids_ref = ids
        print(f"[{model_subdir or 'baseline'}] loaded {len(ids)} IDs; emb dim = {emb.shape[1]}")
    else:
        if not np.array_equal(ids_ref, ids):
            diff1 = set(ids_ref) - set(ids)
            diff2 = set(ids) - set(ids_ref)
            raise ValueError(
                f"Client IDs mismatch in {model_subdir or 'baseline'}:\n"
                f"  only in ref: {len(diff1)}\n"
                f"  only in current: {len(diff2)}"
            )
        print(f"[{model_subdir or 'baseline'}] IDs OK; emb dim = {emb.shape[1]}")

    emb_list.append(emb)

# 3) Concatenazione orizzontale di tutte le embedding
merged_embs = np.concatenate(emb_list, axis=1)
total_dim = merged_embs.shape[1]
print(f"Merged {len(ids_ref)} clients: total embedding dim = {total_dim}")

# 4) Salvataggio in nuova cartella
out_dir = base_dir / "merged_results_finale"
out_dir.mkdir(parents=True, exist_ok=True)

np.save(out_dir / "client_ids.npy", ids_ref)
np.save(out_dir / "embeddings.npy", merged_embs.astype(np.float16))

print(f"✅ Salvati in {out_dir}:")
print(f"   • client_ids.npy")
print(f"   • embeddings.npy (dtype=float16, shape=({len(ids_ref)},{total_dim}))")
