import os
import argparse
import logging
import torch
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from itertools import chain

from config import load_config
from src.data_loader import load_data
from src.preprocessing.heterogeneous_graph import (
    build_hetero_data,
    load_hetero_graph
)
from src.embedding_models.gnn_runner import train_hetero_gnn
from src.embedding_models.gcn_runner import train_hetero_gcn
from src.embedding_models.lightgcn_runner import train_lightgcn
from src.embedding_models.gin_runner import train_hetero_gin
from src.embedding_models.gat_runner import train_hetero_gat
from src.embedding_models.ngcf_runner import train_hetero_ngcf

from validator.validate import validate_and_load_embeddings

# Data splitting imports
from data_utils.data_dir import DataDir
from data_utils.constants import  EventTypes, DAYS_IN_TARGET
from data_utils.split_data import DataSplitter

# Target calculators
from src.preprocessing.target_calculators import (
    ChurnTargetCalculator,
    PropensityTargetCalculator,
    SKUPropensityTargetCalculator,
    PropensityTasks as TC_PropensityTasks,
)

# Training imports
from training_pipeline.tasks import parse_task
from training_pipeline.task_constructor import TaskConstructor
from training_pipeline.logger_factory import NeptuneLoggerFactory
from training_pipeline.train_runner import run_tasks
from training_pipeline.train_runner import run_training

from training_pipeline.metric_aggregator import MetricsAggregator
from training_pipeline.logger_factory import (
    NeptuneLoggerFactory,
)
from training_pipeline.model import (
    UniversalModel,
)
from training_pipeline.tasks import ValidTasks
from training_pipeline.data_module import (
    BehavioralDataModule,
)
from training_pipeline.constants import (
    BATCH_SIZE,
    MAX_EMBEDDING_DIM,
    HIDDEN_SIZE_THIN,
    HIDDEN_SIZE_WIDE,
    LEARNING_RATE,
    MAX_EPOCH,
)
from training_pipeline.target_data import (
    TargetData,
)
from training_pipeline.task_constructor import (
    TaskConstructor,
    TaskSettings,
    transform_client_ids_and_embeddings,
)
from training_pipeline.metric_aggregator import (
    MetricsAggregator,
)

# Import Diversity and Novelty metrics
from training_pipeline.metrics import Diversity, Novelty  # Adjust path if needed

def parse_args():
    parser = argparse.ArgumentParser(description="Framework embedding challenge")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--tasks", nargs="+", default=[])
    parser.add_argument("--neptune_project", default=None)
    parser.add_argument("--neptune_api_token", default=None)
    parser.add_argument("--log_name", default=None)
    parser.add_argument("--score_dir", default=None)
    parser.add_argument("--devices", nargs="+", default=["auto"])
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--disable_relevant_clients_check", action='store_true')
    parser.add_argument("--embedding_filename", type=str, default="embeddings.npy", help="Nome del file .npy per salvare gli embedding")
    parser.add_argument("--client_ids_filename", type=str, default="client_ids.npy", help="Nome del file .npy per salvare la lista di client ids")
    return parser.parse_args()

def parse_devices(device_arg: List[str]) -> List[int] | int | str:
    if (len(device_arg) == 1) and (device_arg[0] == "auto"):
        return "auto"
    try:
        return [int(device) for device in device_arg]
    except ValueError:
        raise ValueError(
            f'Devices argument should be one of "auto", int or list of ints, '
            f'received: "{" ".join(device_arg)}"'
        )

if __name__ == "__main__":
    # 0) Argomenti e configurazione
    print("Parsing arguments and loading configuration...")
    args = parse_args()
    cfg = load_config(args.config)
    print("Configuration successfully loaded.")

    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 1) Caricamento dati
    print("Loading data from Parquet files...")
    data_dict = load_data(cfg['input']['data_dir'])
    print("Data successfully loaded.")

    print(f"Loading relevant clients from {cfg['input']['relevant_clients']}...")
    clients = np.load(cfg['input']['relevant_clients'], allow_pickle=True)
    print(f"Relevant clients loaded: {len(clients)} clients found.")
    print(f"Selected embedding method: {cfg['embed_method']}")
    if cfg['embed_method'] == 'baseline':
        from baseline.aggregated_features_baseline.features_aggregator import FeaturesAggregator
        from baseline.aggregated_features_baseline.constants import EVENT_TYPE_TO_COLUMNS
        from baseline.aggregated_features_baseline.constants import EventTypes
        from data_utils.utils import load_with_properties
        from data_utils.data_dir import DataDir

        baseline_cfg = cfg['baseline']
        agg = FeaturesAggregator(
            num_days=baseline_cfg["num_days"],
            top_n=baseline_cfg["top_n"],
            relevant_client_ids=clients,
        )
        data_dir_obj = DataDir(Path(cfg['input']['data_dir']))

        for event_type, cols in EVENT_TYPE_TO_COLUMNS.items():
            df = load_with_properties(data_dir=data_dir_obj, event_type=event_type.value)
            # leggi il parquet corrispondente
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            agg.generate_features(
                event_type=event_type,
                client_id_column="client_id",
                df=df,
                columns=cols,
            )

        baseline_ids, baseline_embs = agg.merge_features()
        dim = baseline_embs.shape[1]
        print(f"Baseline embedding dimension: {dim}")
        logger.info(f"Baseline embedding dimension: {dim}")
        # 3) Salvataggio embeddings
        embeddings_dir = Path(cfg['output']['embeddings_npy']).parent
        os.makedirs(embeddings_dir, exist_ok=True)
        emb_arr = baseline_embs.astype(np.float16)
        np.save(embeddings_dir / args.embedding_filename, emb_arr)
        np.save(embeddings_dir / args.client_ids_filename, baseline_ids)
        logger.info("Embeddings saved to %s", embeddings_dir)

    # 2) Branching: GNN vs LightGCN vs GIN vs Cleora
    if cfg['embed_method'] in ['gnn', 'lightgcn', 'gin', 'gcn', 'gat', 'ngcf']:
        # 2.1) Split per target
        print("Splitting data to compute targets...")
        challenge_dir = Path(cfg['splitting']['challenge_data_dir'])
        data_dir_obj  = DataDir(data_dir=challenge_dir)
        #target_data = TargetData.read_from_dir(target_dir=data_dir_obj.target_dir)
        end_date      = pd.to_datetime(data_dict['product_buy']['timestamp'].max())

        splitter = DataSplitter(
            challenge_data_dir=data_dir_obj,
            days_in_target=DAYS_IN_TARGET,
            end_date=end_date,
        )
        splitter.split()
        train_target = splitter.target_events["train_target"]

        # 2.2) Top-100 labels
        top_categories = (
            train_target['category']
            .value_counts().nlargest(100)
            .index.tolist()
        )
        top_skus = (
            train_target['sku']
            .value_counts().nlargest(100)
            .index.tolist()
        )

        # 2.3) Init calculators
        churn_calc = ChurnTargetCalculator()
        cat_calc   = PropensityTargetCalculator(
            TC_PropensityTasks.CATEGORY_PROPENSITY,
            top_categories
        )
        sku_calc   = SKUPropensityTargetCalculator(top_skus)

        # 2.4) Preparazione grafo
        print("Preparing heterogeneous graph for embedding...")
        graph_folder = "graph_scipy"
        if os.path.exists(graph_folder):
            print(f"Loading prebuilt graph from folder {graph_folder}...")
            hetero = load_hetero_graph(graph_folder)
            torch.cuda.empty_cache()

            # Load product_id_map separately if present
            product_id_map_file = os.path.join(graph_folder, "product_id_map.npy")
            if os.path.exists(product_id_map_file):
                product_id_map = np.load(product_id_map_file, allow_pickle=True).item()
                print("Loaded product_id_map from file.")
            else:
                raise FileNotFoundError(f"Missing product_id_map.npy in {graph_folder}")
        else:
            print("Building heterogeneous graph from scratch...")
            hetero, product_id_map = build_hetero_data(data_dict, clients)
            torch.cuda.empty_cache()

            # Save product_id_map separately
            os.makedirs(graph_folder, exist_ok=True)
            np.save(os.path.join(graph_folder, "product_id_map.npy"), product_id_map)
            print("Saved product_id_map to file.")

        # Ensure all expected edge types exist (create empty ones if missing)
        expected_edge_types = [
            ("client", "buys", "product"),
            ("client", "adds", "product"),
            ("client", "removes", "product"),
            ("client", "visits", "url"),
            ("client", "searches", "query"),
            ("product", "buys_rev", "client"),
            ("product", "adds_rev", "client"),
            ("product", "removes_rev", "client"),
            ("url", "visits_rev", "client"),
            ("query", "searches_rev", "client"),
        ]

        for edge_type in expected_edge_types:
            if edge_type not in hetero.edge_types:
                hetero[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)

        # 2.5) Compute target tensors
        print("Computing target tensors...")
        y_churn = torch.tensor(
            [churn_calc.compute_target(cid, train_target)[0] for cid in clients],
            dtype=torch.float32,
            device="cpu"
        )

        y_category = torch.stack([
    torch.from_numpy(cat_calc.compute_target(cid, train_target))
    for cid in clients
]).to("cpu")  # forza su CPU

        y_sku = torch.stack([
    torch.from_numpy(sku_calc.compute_target(cid, train_target))
    for cid in clients
]).to("cpu")

        torch.cuda.empty_cache()  # utile, ma non sufficiente se sopra va in GPU

        # 2.6) Train con GNN o LightGCN
        if cfg['embed_method'] == 'gnn':
            print("Training HeteroGNN...")
            user_emb = train_hetero_gnn(
                hetero,
                y_churn,
                y_category,
                y_sku,
                cfg['gnn']
            )
        elif cfg['embed_method'] == 'lightgcn':
            embeddings_dir = Path(cfg['output']['embeddings_npy']).parent

# load using the same filenames you passed in
            baseline_ids  = np.load(embeddings_dir / "client_ids.npy")
            baseline_embs = np.load(embeddings_dir / "embeddings.npy")

            device = torch.device(cfg['lightgcn'].get("device", "cpu"))
            baseline_embs = torch.from_numpy(baseline_embs).to(device).float()
            print("Training LightGCN...")
            user_emb = train_lightgcn(
                hetero,
                y_churn,
                y_category,
                y_sku,
                cfg['lightgcn'],
                baseline_embs=baseline_embs, 
            )
        elif cfg['embed_method'] == 'gcn':
            print("Training HeteroGCN...")
            user_emb = train_hetero_gcn(
                hetero,
                y_churn,
                y_category,
                y_sku,
                cfg['gcn']
            )
        elif cfg['embed_method'] == 'gat':
            print("Training HeteroGAT..")
            user_emb = train_hetero_gat(
                hetero,
                y_churn,
                y_category,
                y_sku,
                cfg['gat']
            )

        elif cfg['embed_method'] == 'ngcf':
            print("Training HeteroNGCF..")
            user_emb = train_hetero_ngcf(
                hetero,
                y_churn,
                y_category,
                y_sku,
                cfg['ngcf']
            )

        else:  # cfg['embed_method'] == 'gin'
            print("Training HeteroGIN...")
            user_emb = train_hetero_gin(
                hetero,
                y_churn,
                y_category,
                y_sku,
                cfg['gin']
            )

        emb_arr = user_emb.numpy()
        print(f"{cfg['embed_method']} training completed.")
        client_ids = clients


    # 3) Salvataggio embeddings
    embeddings_dir = Path(cfg['output']['embeddings_npy']).parent
    os.makedirs(embeddings_dir, exist_ok=True)
    emb_arr = emb_arr.astype(np.float16)
    np.save(embeddings_dir / args.embedding_filename, emb_arr)
    np.save(embeddings_dir / args.client_ids_filename, client_ids)
    logger.info("Embeddings saved to %s", embeddings_dir)

    # 4) Validazione embeddings
    input_dir = Path(cfg['input']['data_dir']) / "input"
    validate_and_load_embeddings(
        input_dir=input_dir,
        embeddings_dir=embeddings_dir,
        max_embedding_dim=MAX_EMBEDDING_DIM,
        disable_relevant_clients_check=False,
    )
    logger.info("Embedding validation completed successfully")

    # 5) Salvataggio split parquet
    challenge_data_dir = DataDir(Path(cfg['input']['data_dir']))

    # Calcolo la data di fine, prendendo l’ultima riga di product_buy
    product_buy_df = pd.read_parquet(
        challenge_data_dir.data_dir / f"{EventTypes.PRODUCT_BUY.value}.parquet"
    )
    end_date = pd.to_datetime(product_buy_df["timestamp"].max())
    # Istanzio DataSplitter con i parametri corretti
    splitter = DataSplitter(
        challenge_data_dir=challenge_data_dir,
        days_in_target=DAYS_IN_TARGET,
        end_date=end_date
    )

    # Genero gli split in memoria e salvo su disco
    splitter.split()
    splitter.save_splits()
    logger.info("Data splitting parquet salvato.")

