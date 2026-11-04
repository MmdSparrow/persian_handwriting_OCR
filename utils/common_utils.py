import numpy as np

__all__ = [
    "folds_mean_std_calculator",
]


def folds_mean_std_calculator(values):
    values = np.asarray(values, dtype=float)
    mean_values = np.nanmean(values) if not np.isnan(values).all() else np.nan
    std_values = np.nanstd(values, ddof=1) if not np.isnan(values).all() else np.nan
    return mean_values, std_values
