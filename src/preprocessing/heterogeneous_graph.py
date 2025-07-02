import numpy as np
import pandas as pd
import torch
import os
import glob
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
import scipy.sparse as sp
from tqdm import tqdm

def build_hetero_data(data_dict: dict, clients: np.ndarray):
    print("Extracting and filtering events...")
    # Estrazione dati
    pb = data_dict["product_buy"]
    ac = data_dict["add_to_cart"]
    rc = data_dict["remove_from_cart"]
    pv = data_dict["page_visit"]
    sq = data_dict["search_query"].copy()
    pp = data_dict["product_properties"]

    # Converti 'query' string -> lista int
    sq["query"] = (
        sq["query"]
          .str.strip("[]")
          .str.split()
          .apply(lambda terms: [int(i) for i in terms])
    )

    # Filtra solo clienti rilevanti
    mask = lambda df: df[df["client_id"].isin(clients)]
    pb, ac, rc, pv, sq = mask(pb), mask(ac), mask(rc), mask(pv).copy(), mask(sq).copy()
    print(f"Filtered events: {len(pb)} buys, {len(ac)} adds, {len(rc)} removes, {len(pv)} visits, {len(sq)} searches")

    # Costruzione mappature ID
    print("Building ID mappings with progress bars...")
    user_ids = np.unique(np.concatenate([
        pb.client_id.values,
        ac.client_id.values,
        rc.client_id.values,
        pv.client_id.values,
        sq.client_id.values
    ]))
    user_id_map = {u: i for i, u in enumerate(tqdm(user_ids, desc="Mapping users"))}

    product_ids = np.unique(np.concatenate([pb.sku.values, ac.sku.values, rc.sku.values]))
    product_id_map = {p: i for i, p in enumerate(tqdm(product_ids, desc="Mapping products"))}

    url_ids = pv.url.unique()
    url_id_map = {u: i for i, u in enumerate(tqdm(url_ids, desc="Mapping urls"))}

    # Unique queries con ordine preservato
    print("Extracting unique queries preserving order...")
    sq["query"] = sq["query"].map(tuple)
    unique_q = pd.DataFrame({"query": list(dict.fromkeys(sq["query"]))})
    query_id_map = {q: i for i, q in enumerate(tqdm(unique_q["query"], desc="Mapping queries"))}

    # Inizializza HeteroData
    data = HeteroData()
    data["client"].num_nodes  = len(user_id_map)
    data["product"].num_nodes = len(product_id_map)
    data["url"].num_nodes     = len(url_id_map)
    data["query"].num_nodes   = len(query_id_map)
    print("Initialized HeteroData with node types: client, product, url, query")

    # FEATURE PRODOTTO
    print("Building product node features...")
    fp = pp.loc[pp.sku.isin(product_id_map)].copy()
    fp["product_idx"] = fp.sku.replace(product_id_map)
    fp.sort_values("product_idx", inplace=True)
    fp["name"] = fp.name.apply(lambda x: [int(i) for i in x.strip("[]").split()])

    cat_price = fp[["category", "price"]].values.astype(float)
    name_emb = np.stack(fp.name.values) / 255.0
    prod_x = np.concatenate([cat_price, name_emb], axis=1)
    data["product"].x = torch.tensor(prod_x, dtype=torch.float)

    # FEATURE QUERY
    print("Building query node features...")
    qx = np.stack(unique_q["query"].tolist()) / 255.0
    data["query"].x = torch.tensor(qx, dtype=torch.float)
    print(f"Query features shape: {qx.shape}")

    # FEATURE CLIENT E URL (identity features semplici)
    N = len(user_id_map)
    data["client"].x = torch.arange(N).unsqueeze(-1).float()

    M = len(url_id_map)
    data["url"].x = torch.arange(M).unsqueeze(-1).float()

    # Funzione helper per aggiungere archi con timestamp
    def add_with_time(df, src, dst, rel, dst_map, src_key="client_id", dst_key=None):
        d = df.copy()
        d[f"{src}_idx"] = d[src_key].map(user_id_map)
        key = dst_key or dst
        d[f"{dst}_idx"] = d[key].map(dst_map)
        ei = torch.tensor(d[[f"{src}_idx", f"{dst}_idx"]].values.T, dtype=torch.long)
        ea = torch.tensor(
            pd.to_datetime(d["timestamp"]).astype(np.int64).to_numpy() // 10**9,
            dtype=torch.long
        )
        data[src, rel, dst].edge_index = ei
        data[src, rel, dst].edge_attr  = ea
        print(f"Added edge {src} -> {dst} ({rel}) with {ei.shape[1]} edges")

    print("Adding event-based edges with timestamps...")
    add_with_time(pb, "client", "product", "buys",    product_id_map, dst_key="sku")
    add_with_time(ac, "client", "product", "adds",    product_id_map, dst_key="sku")
    add_with_time(rc, "client", "product", "removes", product_id_map, dst_key="sku")
    add_with_time(pv, "client", "url",     "visits",  url_id_map,    dst_key="url")
    add_with_time(sq, "client", "query",   "searches",query_id_map,  dst_key="query")

    # Costruzione archi query -> product "matches" basati su similarità Hamming
    def build_query_product_edges(qc, pc, k=1, chunk_size=50):
        edges = []
        for start_idx in tqdm(range(0, len(qc), chunk_size), desc="Processing in chunks"):
            end_idx = min(start_idx + chunk_size, len(qc))
            qc_chunk = qc[start_idx:end_idx]

            # Distanza di Hamming
            hamming_distances = np.bitwise_xor(qc_chunk[:, None], pc).sum(axis=2)
            sims = 1 - hamming_distances / qc_chunk.shape[1]

            top_k_indices = np.argsort(sims, axis=1)[:, -k:]

            for i, indices in enumerate(top_k_indices):
                for j in indices:
                    edges.append([start_idx + i, j])
        return edges

    print("Building query -> product 'matches' edges based on Hamming similarity...")
    qc = np.array(unique_q["query"].tolist(), dtype=np.uint8)
    pc = np.array(fp["name"].tolist(), dtype=np.uint8)
    edges = build_query_product_edges(qc, pc, k=1)
    data["query", "matches", "product"].edge_index = torch.tensor(edges, dtype=torch.long).T
    print(f"Added {len(edges)} query -> product 'matches' edges")

    # Archi query -> url "leads_to" basati su vicinanza temporale
    print("Building query -> url 'leads_to' edges based on temporal proximity...")
    sq["timestamp"] = pd.to_datetime(sq["timestamp"])
    pv["timestamp"] = pd.to_datetime(pv["timestamp"])
    window = pd.Timedelta(seconds=10)
    pairs = []
    for cid in sq["client_id"].unique():
        qd = sq[sq["client_id"] == cid]
        vd = pv[pv["client_id"] == cid]
        for _, qr in qd.iterrows():
            ts, te = qr["timestamp"], qr["timestamp"] + window
            matches = vd[(vd["timestamp"] >= ts) & (vd["timestamp"] <= te)]
            for _, vr in matches.iterrows():
                qi = query_id_map[tuple(qr["query"])]
                ui = url_id_map[vr["url"]]
                pairs.append([qi, ui])
    data["query", "leads_to", "url"].edge_index = torch.tensor(pairs, dtype=torch.long).T
    print(f"Added {len(pairs)} query -> url 'leads_to' edges")

    # Converti in grafo non diretto e aggiungi archi reverse
    print("Converting to undirected graph and adding reverse edges...")
    data = ToUndirected()(data)
    for (s, r, d), ei in data.edge_index_dict.items():
        rr = f"{r}_rev"
        data[d, rr, s].edge_index = ei.flip(0)
        if hasattr(data[s, r, d], "edge_attr"):
            data[d, rr, s].edge_attr = data[s, r, d].edge_attr
    print("Graph conversion to undirected completed with reverse edges added.")

    # Prepara label per classificazione (categoria prodotto)
    y_category = torch.tensor(fp["category"].values, dtype=torch.long)

    sku_to_int = {sku: idx for idx, sku in enumerate(product_ids)}
    y_sku = torch.tensor(fp["sku"].map(sku_to_int).values, dtype=torch.long)

    # Opzionale: salva o restituisci
    # torch.save(y_category, "graph_scipy/y_category.pt")
    # torch.save(y_sku, "graph_scipy/y_sku.pt")

    # Salvataggio features come sparse (se vuoi)
    os.makedirs("graph_scipy", exist_ok=True)
    sp.save_npz("graph_scipy/client_x.npz", sp.csr_matrix(data["client"].x.numpy()))
    sp.save_npz("graph_scipy/url_x.npz", sp.csr_matrix(data["url"].x.numpy()))
    sp.save_npz("graph_scipy/product_x.npz", sp.csr_matrix(prod_x))
    sp.save_npz("graph_scipy/query_x.npz", sp.csr_matrix(qx))

    # Salvataggio matrici di adiacenza (edge_index -> sparse)
    for src, rel, dst in data.edge_types:
        edge_index = data[src, rel, dst].edge_index
        rows, cols = edge_index.numpy()
        if hasattr(data[src, rel, dst], "edge_attr"):
            weights = data[src, rel, dst].edge_attr.numpy()
        else:
            weights = np.ones(rows.shape[0], dtype=np.int64)
        adj = sp.coo_matrix((weights, (rows, cols)), shape=(data[src].num_nodes, data[dst].num_nodes))
        sp.save_npz(f"graph_scipy/{src}__{rel}__{dst}.npz", adj)

    return data, product_id_map


def load_hetero_graph(folder: str = "graph_scipy"):
    
    #Loads the saved graph via scipy: loads all the .npz matrices from the folder.
   
    data = HeteroData()

    # Load client features dense
    coo_c = sp.load_npz(os.path.join(folder, "client_x.npz"))
    data['client'].x = torch.tensor(coo_c.toarray(), dtype=torch.float)
    data['client'].num_nodes = coo_c.shape[0]

    coo_u = sp.load_npz(os.path.join(folder, "url_x.npz"))
    data['url'].x = torch.tensor(coo_u.toarray(), dtype=torch.float)
    data['url'].num_nodes = coo_u.shape[0]

    # Load product features dense
    psp = sp.load_npz(os.path.join(folder, "product_x.npz"))
    data['product'].x = torch.tensor(psp.toarray(), dtype=torch.float)
    data['product'].num_nodes = psp.shape[0]

    # Load query features dense
    qsp = sp.load_npz(os.path.join(folder, "query_x.npz"))
    data['query'].x = torch.tensor(qsp.toarray(), dtype=torch.float)
    data['query'].num_nodes = qsp.shape[0]

    # Load adjacency matrices for all relations
    for path in glob.glob(os.path.join(folder, "*__*__.npz")):
        fname = os.path.basename(path)
        src, rel, dst_ext = fname.split("__")
        dst = dst_ext.replace(".npz", "")
        mat = sp.load_npz(path)
        rows, cols = mat.row, mat.col
        data[src, rel, dst].edge_index = torch.tensor([rows, cols], dtype=torch.long)
        if mat.data is not None:
            data[src, rel, dst].edge_attr = torch.tensor(mat.data, dtype=torch.long)

    return data



"""def load_hetero_graph(folder: str = "graph_scipy"):
   
    data = HeteroData()

    # Load client features sparse
    coo_c = sp.load_npz(os.path.join(folder, "client_x.npz")).tocoo()  # Convert CSR to COO
    data['client'].x = torch.sparse_coo_tensor(
        indices=torch.tensor([coo_c.row, coo_c.col], dtype=torch.long),
        values=torch.tensor(coo_c.data, dtype=torch.float),
        size=coo_c.shape,
        dtype=torch.float
    )
    data['client'].num_nodes = coo_c.shape[0]

    # Load url features sparse
    coo_u = sp.load_npz(os.path.join(folder, "url_x.npz")).tocoo()  # Convert CSR to COO
    data['url'].x = torch.sparse_coo_tensor(
        indices=torch.tensor([coo_u.row, coo_u.col], dtype=torch.long),
        values=torch.tensor(coo_u.data, dtype=torch.float),
        size=coo_u.shape,
        dtype=torch.float
    )
    data['url'].num_nodes = coo_u.shape[0]

    # Load product features sparse
    psp = sp.load_npz(os.path.join(folder, "product_x.npz")).tocoo()  # Convert CSR to COO
    data['product'].x = torch.sparse_coo_tensor(
        indices=torch.tensor([psp.row, psp.col], dtype=torch.long),
        values=torch.tensor(psp.data, dtype=torch.float),
        size=psp.shape,
        dtype=torch.float
    )
    data['product'].num_nodes = psp.shape[0]

    # Load query features sparse
    qsp = sp.load_npz(os.path.join(folder, "query_x.npz")).tocoo()  # Convert CSR to COO
    data['query'].x = torch.sparse_coo_tensor(
        indices=torch.tensor([qsp.row, qsp.col], dtype=torch.long),
        values=torch.tensor(qsp.data, dtype=torch.float),
        size=qsp.shape,
        dtype=torch.float
    )
    data['query'].num_nodes = qsp.shape[0]

    # Load adjacency matrices for all relations as sparse
    for path in glob.glob(os.path.join(folder, "*__*__.npz")):
        fname = os.path.basename(path)
        src, rel, dst_ext = fname.split("__")
        dst = dst_ext.replace(".npz", "")
        mat = sp.load_npz(path).tocoo()  # Convert CSR to COO
        rows, cols = mat.row, mat.col
        data[src, rel, dst].edge_index = torch.sparse_coo_tensor(
            indices=torch.tensor([rows, cols], dtype=torch.long),
            values=torch.tensor(mat.data, dtype=torch.float),
            size=mat.shape,
            dtype=torch.long
        )

    return data"""
