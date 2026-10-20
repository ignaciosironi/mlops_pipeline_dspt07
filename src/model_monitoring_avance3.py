"""Reusable drift calculations and prediction history for Advance 3."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp

LOG_PATH = Path(__file__).resolve().parent / 'monitoring_log.csv'
HISTORY_PATH = Path(__file__).resolve().parent / 'drift_history.csv'
META_COLUMNS = {'prediction', 'timestamp', 'batch_id'}


def normalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare records for JSON transport without modifying source data."""
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_datetime64_any_dtype(result[column]):
            result[column] = result[column].dt.strftime('%Y-%m-%dT%H:%M:%S')
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.astype(object).where(pd.notna(result), None)


def record_predictions(features: pd.DataFrame, predictions: list, path: Path = LOG_PATH) -> None:
    if len(features) != len(predictions):
        raise ValueError('The number of predictions must match the number of input rows.')
    entries = normalize_features(features)
    entries['prediction'] = predictions
    entries['timestamp'] = datetime.now(timezone.utc).isoformat()
    entries['batch_id'] = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing_columns = pd.read_csv(path, nrows=0).columns.tolist()
        if existing_columns != entries.columns.tolist():
            raise ValueError('Log columns differ from current data. Archive the old log before using a new schema.')
    entries.to_csv(path, mode='a', index=False, header=not path.exists())


def read_log(path: Path = LOG_PATH) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _numeric_metrics(reference: pd.Series, current: pd.Series, bins: int = 10) -> dict:
    a = pd.to_numeric(reference, errors='coerce').dropna().to_numpy(dtype=float)
    b = pd.to_numeric(current, errors='coerce').dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return {'ks_pvalue': np.nan, 'psi': np.nan, 'js_divergence': np.nan, 'chi2_pvalue': np.nan}
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        edges = np.linspace(lo - 0.5, hi + 0.5, bins + 1) if lo != hi else np.array([lo - 0.5, lo + 0.5])
    edges[0], edges[-1] = -np.inf, np.inf
    hist_a = np.histogram(a, bins=edges)[0].astype(float)
    hist_b = np.histogram(b, bins=edges)[0].astype(float)
    p = np.clip(hist_a / hist_a.sum(), 1e-6, None)
    q = np.clip(hist_b / hist_b.sum(), 1e-6, None)
    p, q = p / p.sum(), q / q.sum()
    return {'ks_pvalue': float(ks_2samp(a, b).pvalue),
            'psi': float(np.sum((q - p) * np.log(q / p))),
            'js_divergence': float(jensenshannon(p, q, base=2) ** 2),
            'chi2_pvalue': np.nan}


def _categorical_metrics(reference: pd.Series, current: pd.Series) -> dict:
    a = reference.fillna('__MISSING__').astype(str).value_counts()
    b = current.fillna('__MISSING__').astype(str).value_counts()
    labels = a.index.union(b.index)
    counts = np.vstack([a.reindex(labels, fill_value=0), b.reindex(labels, fill_value=0)]).astype(float)
    p = np.clip(counts[0] / counts[0].sum(), 1e-6, None)
    q = np.clip(counts[1] / counts[1].sum(), 1e-6, None)
    p, q = p / p.sum(), q / q.sum()
    return {'ks_pvalue': np.nan, 'psi': float(np.sum((q-p)*np.log(q/p))),
            'js_divergence': float(jensenshannon(p, q, base=2)**2),
            'chi2_pvalue': float(chi2_contingency(counts[:, counts.sum(axis=0)>0])[1]) if len(labels)>1 else 1.0}


def measure_drift(reference: pd.DataFrame, current: pd.DataFrame, psi_warning: float = 0.1,
                  psi_critical: float = 0.25) -> pd.DataFrame:
    """Compare common feature columns; PSI thresholds are heuristics, not statistical guarantees."""
    if psi_warning >= psi_critical:
        raise ValueError('Warning threshold must be below critical threshold.')
    if reference.empty or current.empty:
        return pd.DataFrame()
    output = []
    for column in reference.columns:
        if column in META_COLUMNS or column not in current.columns:
            continue
        a, b = reference[column], current[column]
        is_numeric = pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_bool_dtype(a)
        scores = _numeric_metrics(a, b) if is_numeric else _categorical_metrics(a, b)
        psi = scores['psi']
        severity = 'Insufficient data' if pd.isna(psi) else ('Critical' if psi >= psi_critical else 'Warning' if psi >= psi_warning else 'Stable')
        output.append({'feature': column, 'type': 'numeric' if is_numeric else 'categorical',
                       'reference_rows': int(a.notna().sum()), 'current_rows': int(b.notna().sum()),
                       **scores, 'status': severity})
    return pd.DataFrame(output)


def save_drift_snapshot(results: pd.DataFrame, path: Path = HISTORY_PATH) -> None:
    if results.empty:
        return
    snapshot = results.copy()
    snapshot.insert(0, 'checked_at', datetime.now(timezone.utc).isoformat())
    snapshot.to_csv(path, mode='a', header=not path.exists(), index=False)


def read_history(path: Path = HISTORY_PATH) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()
