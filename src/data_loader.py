import pandas as pd

def load_data(data_dir: str) -> dict:
    """
    Carica tutti i Parquet in un dizionario.
    """
    tables = [
        "product_buy",
        "add_to_cart",
        "remove_from_cart",
        "page_visit",
        "search_query",
        "product_properties"
    ]
    return {
        name: pd.read_parquet(f"{data_dir}/{name}.parquet", engine="pyarrow")
        for name in tables
    }
