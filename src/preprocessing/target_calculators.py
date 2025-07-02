from enum import Enum
import numpy as np
from typing import List

class PropensityTasks(Enum):
    CATEGORY_PROPENSITY = "category"

def get_propensity_column(task: PropensityTasks) -> str:
    if task == PropensityTasks.CATEGORY_PROPENSITY:
        return "category"
    raise ValueError(f"Unsupported task: {task}")

class TargetCalculator:
    def compute_target(self, client_id: int, target_df) -> np.ndarray:
        raise NotImplementedError

class ChurnTargetCalculator(TargetCalculator):
    @property
    def target_dim(self) -> int:
        return 1

    def compute_target(self, client_id: int, target_df) -> np.ndarray:
        return np.array([0.0 if client_id in target_df["client_id"].values else 1.0],
                        dtype=np.float32)

class PropensityTargetCalculator(TargetCalculator):
    def __init__(self, task: PropensityTasks, propensity_targets: np.ndarray):
        self._propensity_type = get_propensity_column(task)
        self._propensity_targets = propensity_targets

    @property
    def target_dim(self) -> int:
        return len(self._propensity_targets)

    def compute_target(self, client_id: int, target_df) -> np.ndarray:
        cats = target_df.loc[target_df["client_id"] == client_id, self._propensity_type].unique()
        mask = np.isin(self._propensity_targets, cats)
        return mask.astype(np.float32)

class SKUPropensityTargetCalculator(TargetCalculator):
    def __init__(self, top_skus: List[int]):
        self._top_skus = np.array(top_skus)

    @property
    def target_dim(self) -> int:
        return len(self._top_skus)

    def compute_target(self, client_id: int, target_df) -> np.ndarray:
        skus = target_df.loc[target_df["client_id"] == client_id, "sku"].unique()
        mask = np.isin(self._top_skus, skus)
        return mask.astype(np.float32)
