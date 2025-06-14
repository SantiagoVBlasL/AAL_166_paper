#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilitario para Jupyter Notebook: Extracción de Características de Conectividad fMRI
Versión: v6.5.21_FinalFix

Cambios Principales:
- Versión actualizada a v6.5.21.
- ¡Corregido NameError definitivo!: Las variables `granger_suffix_global` y `deconv_str`
  ahora se inicializan como placeholders en el ámbito global, y se asignan sus valores
  finales DENTRO de main(args) DESPUÉS de que argparse ha poblado las variables booleanas
  (USE_GRANGER_CHANNEL, APPLY_HRF_DECONVOLUTION) y GRANGER_MAX_LAG.
- Refactorización crítica: Se movió la única invocación de _initialize_aal3_roi_processing_info()
  dentro de main(args) para asegurar que use los parámetros de argparse y evitar doble inicialización.
- Se implementó la lógica para que las advertencias CRÍTICAS iniciales no se muestren al importar el módulo,
  sino solo cuando se ejecuta como script principal (dentro de if __name__ == "__main__":).
- Corregido NameError: _get_aal3_network_mapping_and_order se define ANTES de _initialize_aal3_roi_processing_info.
- Se añadió la definición de `_calculate_mi_for_pair` que estaba ausente.
- Se eliminó la variable global `TARGET_LEN_TS` y cualquier referencia, ya que la longitud de las series
  temporales no se homogeneiza por truncado/padding en este pipeline.
- Se corrigió el uso inconsistente de `current_expected_rois_for_assembly` en `process_single_subject_pipeline`.
- Se refactorizó `main()` para usar `argparse` para la configuración de parámetros.
- Se aseguró la declaración `global N_ROIS_EXPECTED` donde corresponde.
- Se mejoró la lógica de selección de canales Pearson-OMST/Fallback para evitar duplicación innecesaria
  si dyconnmap no está disponible.
- Mantenidas todas las correcciones y mejoras de versiones anteriores, incluyendo la estructura
  para reordenamiento de ROIs y las extensas notas para tesis.

Requisitos Clave:
- dyconnmap >= 1.0.4
- networkx == 2.6.3 (debido a dyconnmap)
- scikit-learn >= 1.0
- nilearn >= 0.9
- statsmodels (para Causalidad de Granger)
- numpy, pandas, scipy, tqdm, psutil, joblib, nibabel
- (Opcional, para sugerencias avanzadas): neuroHarmonize, ruptures, hmmlearn, etc.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_regression
# Para Graphical Lasso (Sugerencia Tesis 5)
# from sklearn.covariance import GraphicalLassoCV
from nilearn.glm.first_level import spm_hrf, glover_hrf
from nilearn.datasets import fetch_atlas_yeo_2011 # Para reordenamiento de ROIs
from nilearn import image as nli_image # Para resampling
import nibabel as nib # Para cargar atlas NIfTI
from scipy.signal import butter, filtfilt, windows, cheby1 # Added cheby1
# Deconvolve se importa de scipy.signal, pero lo reemplazaremos con wiener_deconv
import os
import scipy.io as sio
from pathlib import Path
import psutil
import gc
import logging
import time
from typing import List, Tuple, Dict, Optional, Any, Union
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
import warnings
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed
import argparse
from tqdm import tqdm

# --- Configuración del Logger ---
# Declara la instancia del logger a nivel de módulo.
# Su configuración (nivel, formato) se hará solo si el script se ejecuta directamente.
logger = logging.getLogger(__name__)

# --- Importación de OMST usando dyconnmap ---
OMST_PYTHON_LOADED = False
orthogonal_minimum_spanning_tree = None
PEARSON_OMST_CHANNEL_NAME_PRIMARY = "Pearson_OMST_GCE_Signed_Weighted"
PEARSON_OMST_FALLBACK_NAME = "Pearson_Full_FisherZ_Signed"
# PEARSON_OMST_CHANNEL_NAME se definirá en _initialize_aal3_roi_processing_info

try:
    from dyconnmap.graphs.threshold import threshold_omst_global_cost_efficiency
    orthogonal_minimum_spanning_tree = threshold_omst_global_cost_efficiency
    logger.info("Successfully imported 'threshold_omst_global_cost_efficiency' from 'dyconnmap.graphs.threshold' and aliased as 'orthogonal_minimum_spanning_tree'.")
    OMST_PYTHON_LOADED = True
except ImportError:
    logger.error("ERROR: Dyconnmap module or 'threshold_omst_global_cost_efficiency' not found. "
                 f"Channel '{PEARSON_OMST_FALLBACK_NAME}' will be used as fallback if selected. "
                 "Please ensure dyconnmap is installed: pip install dyconnmap")
except Exception as e_import:
    logger.error(f"ERROR during dyconnmap import: {e_import}. "
                 f"Channel '{PEARSON_OMST_FALLBACK_NAME}' will be used as fallback if selected.")

# --- 0. Global Configuration and Constants (defaults, will be overwritten by argparse) ---
# Estas variables globales serán pobladas por la función main() al usar argparse.
# Se mantienen aquí con valores predeterminados de tipo o None para que el IDE las reconozca.
BASE_PATH_AAL3: Path = Path('.') # Valor temporal, será sobrescrito
QC_OUTPUT_DIR: Path = Path('.')
SUBJECT_METADATA_CSV_PATH_QC: Path = Path('.')
SUBJECT_METADATA_CSV_PATH: Path = Path('.')
QC_REPORT_CSV_PATH: Path = Path('.')
ROI_SIGNALS_DIR_PATH_AAL3: Path = Path('.')
ROI_FILENAME_TEMPLATE: str = ''
AAL3_META_PATH: Path = Path('.')
AAL3_NIFTI_PATH: Path = Path('.')

TR_SECONDS: float = 0.0
LOW_CUT_HZ: float = 0.0
HIGH_CUT_HZ: float = 0.0
FILTER_ORDER: int = 0
TAPER_ALPHA: float = 0.0

RAW_DATA_EXPECTED_COLUMNS: int = 0
AAL3_MISSING_INDICES_1BASED: List[int] = []
EXPECTED_ROIS_AFTER_AAL3_MISSING_REMOVAL: int = 0
SMALL_ROI_VOXEL_THRESHOLD: int = 0

N_NEIGHBORS_MI: int = 0

DFC_WIN_POINTS_SEC: float = 0.0
DFC_STEP_SEC: float = 0.0

APPLY_HRF_DECONVOLUTION: bool = False
HRF_MODEL: str = ''

USE_GRANGER_CHANNEL: bool = False
GRANGER_MAX_LAG: int = 0

# Estos dos se inicializarán con valores dummy o vacíos y se recalcularán en main()
granger_suffix_global: str = ""
deconv_str: str = ""
OUTPUT_CONNECTIVITY_DIR_NAME_BASE: str = ""

POSSIBLE_ROI_KEYS: List[str] = ["signals", "ROISignals", "roi_signals", "ROIsignals_AAL3", "AAL3_signals", "roi_ts"]

USE_PEARSON_OMST_CHANNEL: bool = False
USE_PEARSON_FULL_SIGNED_CHANNEL: bool = False
USE_MI_CHANNEL_FOR_THESIS: bool = False
USE_DFC_ABS_DIFF_MEAN_CHANNEL: bool = False
USE_DFC_STDDEV_CHANNEL: bool = False

CONNECTIVITY_CHANNEL_NAMES: List[str] = []
N_CHANNELS: int = 0

# Inicialización aquí para MAX_WORKERS y TOTAL_CPU_CORES.
# Estos se establecerán con valores predeterminados o serán sobrescritos por argparse en main.
try:
    TOTAL_CPU_CORES = multiprocessing.cpu_count()
    MAX_WORKERS = max(1, TOTAL_CPU_CORES // 2 if TOTAL_CPU_CORES > 2 else 1)
except NotImplementedError:
    logger.warning("multiprocessing.cpu_count() no está implementado en esta plataforma. Usando MAX_WORKERS = 1.")
    TOTAL_CPU_CORES = 1
    MAX_WORKERS = 1

# Variables globales que se inicializan en _initialize_aal3_roi_processing_info
VALID_AAL3_ROI_INFO_DF_166: Optional[pd.DataFrame] = None
AAL3_MISSING_INDICES_0BASED: Optional[List[int]] = None
INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166: Optional[List[int]] = None
FINAL_N_ROIS_EXPECTED: Optional[int] = None
OUTPUT_CONNECTIVITY_DIR_NAME: Optional[str] = None
AAL3_ROI_ORDER_MAPPING: Optional[Dict[str, Any]] = None
N_ROIS_EXPECTED: Optional[int] = None # Se usa para el nombre de la carpeta y se actualiza en init

YEO17_LABELS_TO_NAMES = {
    0: "Background/NonCortical",
    1: "Visual_Peripheral", 2: "Visual_Central",
    3: "Somatomotor_A", 4: "Somatomotor_B",
    5: "DorsalAttention_A", 6: "DorsalAttention_B",
    7: "Salience_VentralAttention_A", 8: "Salience_VentralAttention_B",
    9: "Limbic_A_TempPole", 10: "Limbic_B_OFC",
    11: "Control_C", 12: "Control_A", 13: "Control_B",
    14: "DefaultMode_Temp", 15: "DefaultMode_Core",
    16: "DefaultMode_DorsalMedial", 17: "DefaultMode_VentralMedial"
}

# --- Funciones auxiliares para el cálculo de conectividad ---
def _calculate_mi_for_pair(x: np.ndarray, y: np.ndarray, n_neighbors: int) -> float:
    """Calcula la información mutua entre dos series usando k-vecinos más cercanos."""
    try:
        x_reshaped = x.reshape(-1, 1)
        y_reshaped = y.flatten()

        if len(x_reshaped) != len(y_reshaped) or len(x_reshaped) < (n_neighbors + 1):
            logger.warning(f"Insufficient data points ({len(x_reshaped)}) or mismatch for MI calculation (need at least {n_neighbors + 1} points). Returning 0.")
            return 0.0
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            mi_val = mutual_info_regression(X=x_reshaped, y=y_reshaped, n_neighbors=n_neighbors, random_state=42)[0]
        return max(0.0, mi_val)
    except Exception as e:
        logger.warning(f"Error calculating MI for a pair: {e}. Returning 0.0.")
        return 0.0

def fisher_r_to_z(r_matrix: np.ndarray, n_tp: int, eps: float = 1e-7) -> np.ndarray:
    r_clean = np.nan_to_num(r_matrix.astype(np.float32), nan=0.0)
    r_clipped = np.clip(r_clean, -1.0 + eps, 1.0 - eps)
    z_matrix = np.arctanh(r_clipped)

    if n_tp > 3:
        z_matrix = z_matrix / np.sqrt(n_tp - 3)

    np.fill_diagonal(z_matrix, 0.0)
    return z_matrix.astype(np.float32)

def calculate_pearson_full_fisher_z_signed(ts_subject: np.ndarray, sid: str) -> Optional[np.ndarray]:
    n_tp = ts_subject.shape[0]
    if n_tp < 2:
        logger.warning(f"Pearson_Full_FisherZ_Signed (S {sid}): Insufficient timepoints ({n_tp} < 2).")
        return None
    try:
        corr_matrix = np.corrcoef(ts_subject, rowvar=False).astype(np.float32)
        if corr_matrix.ndim == 0 or corr_matrix.shape[0] == 0:
            logger.warning(f"Pearson_Full_FisherZ_Signed (S {sid}): Correlation resulted in a scalar or empty. Input shape: {ts_subject.shape}. Returning zero matrix.")
            num_rois = ts_subject.shape[1]
            return np.zeros((num_rois, num_rois), dtype=np.float32) if num_rois > 0 else None

        z_transformed_matrix = fisher_r_to_z(corr_matrix, n_tp)
        return z_transformed_matrix
    except Exception as e:
        logger.error(f"Error calculating Pearson_Full_FisherZ_Signed for S {sid}: {e}", exc_info=True)
        return None

def calculate_pearson_omst_signed_weighted(ts_subject: np.ndarray, sid: str) -> Optional[np.ndarray]:
    n_tp = ts_subject.shape[0]
    if not OMST_PYTHON_LOADED or orthogonal_minimum_spanning_tree is None:
        logger.error(f"Pearson_OMST_GCE_Signed_Weighted (S {sid}): Dyconnmap OMST function not available. Cannot calculate.")
        return None

    if n_tp < 2:
        logger.warning(f"Pearson_OMST_GCE_Signed_Weighted (S {sid}): Insufficient timepoints ({n_tp} < 2).")
        return None

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered in divide", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)

            corr_matrix = np.corrcoef(ts_subject, rowvar=False).astype(np.float32)

            if corr_matrix.ndim == 0 or corr_matrix.shape[0] == 0:
                logger.warning(f"Pearson_OMST_GCE_Signed_Weighted (S {sid}): Correlation resulted in a scalar or empty. Input shape: {ts_subject.shape}. Returning zero matrix.")
                num_rois = ts_subject.shape[1]
                return np.zeros((num_rois, num_rois), dtype=np.float32) if num_rois > 0 else None

            z_transformed_matrix = fisher_r_to_z(corr_matrix, n_tp)
            weights_for_omst_gce = np.abs(z_transformed_matrix)
            np.fill_diagonal(weights_for_omst_gce, 0.0)

            if np.all(np.isclose(weights_for_omst_gce, 0)):
                logger.warning(f"Pearson_OMST_GCE_Signed_Weighted (S {sid}): All input weights for OMST GCE are zero. Returning zero matrix (original Z-transformed).")
                return z_transformed_matrix.astype(np.float32)

            omst_outputs = orthogonal_minimum_spanning_tree(weights_for_omst_gce, n_msts=None)

            if isinstance(omst_outputs, tuple) and len(omst_outputs) >= 2:
                omst_adjacency_matrix_gce_weighted = np.asarray(omst_outputs[1]).astype(np.float32)
            else:
                logger.error(f"S {sid}: dyconnmap.threshold_omst_global_cost_efficiency returned an unexpected type or insufficient outputs: {type(omst_outputs)}. Cannot extract OMST matrix.")
                return None

            if not isinstance(omst_adjacency_matrix_gce_weighted, np.ndarray):
                logger.error(f"S {sid}: Extracted omst_adjacency_matrix_gce_weighted is not a numpy array (type: {type(omst_adjacency_matrix_gce_weighted)}). Cannot proceed.")
                return None

            binary_omst_mask = (omst_adjacency_matrix_gce_weighted > 0).astype(int)
            signed_weighted_omst_matrix = z_transformed_matrix * binary_omst_mask
            np.fill_diagonal(signed_weighted_omst_matrix, 0.0)

            return signed_weighted_omst_matrix.astype(np.float32)

    except AttributeError as ae:
        if 'from_numpy_matrix' in str(ae).lower() or 'from_numpy_array' in str(ae).lower():
            logger.error(f"Error calculating Pearson_OMST_GCE_Signed_Weighted (dyconnmap) for S {sid}: NetworkX version incompatibility. "
                         f"Dyconnmap (v1.0.4) may be using a deprecated NetworkX function. "
                         f"Your NetworkX version: {nx.__version__}. Consider using NetworkX 2.x. Original error: {ae}", exc_info=False)
        else:
            logger.error(f"AttributeError calculating Pearson_OMST_GCE_Signed_Weighted (dyconnmap) for S {sid}: {ae}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error calculating Pearson_OMST_GCE_Signed_Weighted (dyconnmap) connectivity for S {sid}: {e}", exc_info=True)
        return None

def calculate_mi_knn_connectivity(ts_subject: np.ndarray, n_neighbors_val: int, sid: str) -> Optional[np.ndarray]:
    n_tp, n_rois = ts_subject.shape
    if n_tp == 0:
        logger.warning(f"MI_KNN (S {sid}): 0 Timepoints provided. Cannot calculate MI.")
        return None
    
    # Apply adaptive k
    k_adaptive = min(n_neighbors_val, max(1, int(np.sqrt(n_tp)))) # Ensure k is at least 1
    if n_tp <= k_adaptive:
        logger.warning(f"MI_KNN (S {sid}): Timepoints ({n_tp}) <= adaptive k ({k_adaptive}). Skipping MI calculation.")
        return np.zeros((n_rois, n_rois), dtype=np.float32) if n_rois > 0 else None

    mi_matrix = np.zeros((n_rois, n_rois), dtype=np.float32)

    tasks = []
    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            tasks.append({'i': i, 'j': j,
                          'data_i': ts_subject[:, i].reshape(-1, 1),
                          'data_j': ts_subject[:, j],
                          'n_neighbors': k_adaptive}) # Use adaptive k

    # Use n_jobs=1 to avoid oversubscription when running in ProcessPoolExecutor
    # as parallelization is already handled at the subject level.
    n_jobs_mi = 1 

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            results_list = Parallel(n_jobs=n_jobs_mi)(
                delayed(_calculate_mi_for_pair)(task['data_i'], task['data_j'], task['n_neighbors'])
                for task in tasks
            )
            # Retaining the dual calculation for robustness as discussed
            results_list_ji = Parallel(n_jobs=n_jobs_mi)(
                delayed(_calculate_mi_for_pair)(task['data_j'].reshape(-1,1), task['data_i'].flatten(), task['n_neighbors'])
                for task in tasks
            )
    except Exception as e_parallel:
        logger.error(f"MI_KNN (S {sid}): Error during parallel MI calculation: {e_parallel}. Falling back to serial.")
        results_list = [_calculate_mi_for_pair(task['data_i'], task['data_j'], task['n_neighbors']) for task in tasks]
        results_list_ji = [_calculate_mi_for_pair(task['data_j'].reshape(-1,1), task['data_i'].flatten(), task['n_neighbors']) for task in tasks]

    for k, task in enumerate(tasks):
        i, j = task['i'], task['j']
        mi_val_ij = results_list[k]
        mi_val_ji = results_list_ji[k]
        mi_matrix[i, j] = mi_matrix[j, i] = (mi_val_ij + mi_val_ji) / 2.0

    return mi_matrix

def calculate_custom_dfc_abs_diff_mean(ts_subject: np.ndarray, win_points_sec_val: float, step_sec_val: float, sid: str, tr_seconds: float) -> Optional[np.ndarray]:
    n_tp, n_rois = ts_subject.shape
    
    # Convertir segundos a puntos temporales
    win_points_val = int(win_points_sec_val / tr_seconds)
    step_val = int(step_sec_val / tr_seconds)

    if win_points_val < 2:
        win_points_val = 2
        logger.warning(f"dFC_AbsDiffMean (S {sid}): Calculated window points ({win_points_val}) too small. Adjusted to 2.")
    if step_val < 1:
        step_val = 1
        logger.warning(f"dFC_AbsDiffMean (S {sid}): Calculated step points ({step_val}) too small. Adjusted to 1.")

    if n_tp < win_points_val:
        logger.warning(f"dFC_AbsDiffMean (S {sid}): Timepoints ({n_tp}) < window length ({win_points_val}). Skipping.")
        return np.zeros((n_rois, n_rois), dtype=np.float32) if n_rois > 0 else None

    num_windows = (n_tp - win_points_val) // step_val + 1
    if num_windows < 2:
        logger.warning(f"dFC_AbsDiffMean (S {sid}): Fewer than 2 windows ({num_windows}) can be formed. Skipping.")
        return np.zeros((n_rois, n_rois), dtype=np.float32) if n_rois > 0 else None

    sum_abs_diff_matrix = np.zeros((n_rois, n_rois), dtype=np.float64)
    n_diffs_calculated = 0
    prev_corr_matrix: Optional[np.ndarray] = None # Changed name to reflect signed correlation

    for idx in range(num_windows):
        start_idx = idx * step_val
        end_idx = start_idx + win_points_val
        window_ts = ts_subject[start_idx:end_idx, :]

        if window_ts.shape[0] < 2: continue

        try:
            corr_matrix_window = np.corrcoef(window_ts, rowvar=False)
            if corr_matrix_window.ndim < 2 or corr_matrix_window.shape != (n_rois, n_rois):
                logger.warning(f"dFC_AbsDiffMean (S {sid}), Window {idx}: corrcoef returned unexpected shape {corr_matrix_window.shape}. Using zeros for this window's contribution.")
                corr_matrix_window = np.full((n_rois, n_rois), 0.0, dtype=np.float32)
            else:
                corr_matrix_window = np.nan_to_num(corr_matrix_window.astype(np.float32), nan=0.0)

            # --- DFC AbsDiffMean change: Use signed correlation in calculations ---
            current_corr_matrix = corr_matrix_window # No np.abs here, use signed correlation
            np.fill_diagonal(current_corr_matrix, 0) # Fill diagonal with 0 for all matrices

            if prev_corr_matrix is not None:
                # Calculate element-wise absolute difference of signed correlations
                sum_abs_diff_matrix += np.abs(current_corr_matrix - prev_corr_matrix)
                n_diffs_calculated += 1
            prev_corr_matrix = current_corr_matrix
        except Exception as e:
            logger.error(f"dFC_AbsDiffMean (S {sid}), Window {idx}: Error calculating/processing correlation: {e}")

    if n_diffs_calculated == 0:
        logger.warning(f"dFC_AbsDiffMean (S {sid}): No valid differences between windowed correlations were calculated. Returning zero matrix.")
        return np.zeros((n_rois, n_rois), dtype=np.float32) if n_rois > 0 else None

    mean_abs_diff_matrix = (sum_abs_diff_matrix / n_diffs_calculated).astype(np.float32)
    np.fill_diagonal(mean_abs_diff_matrix, 0)
    return mean_abs_diff_matrix

def calculate_dfc_std_dev(ts_subject: np.ndarray, win_points_sec_val: float, step_sec_val: float, sid: str, tr_seconds: float) -> Optional[np.ndarray]:
    n_tp, n_rois = ts_subject.shape

    # Convertir segundos a puntos temporales
    win_points_val = int(win_points_sec_val / tr_seconds)
    step_val = int(step_sec_val / tr_seconds)

    if win_points_val < 2:
        win_points_val = 2
        logger.warning(f"dFC_StdDev (S {sid}): Calculated window points ({win_points_val}) too small. Adjusted to 2.")
    if step_val < 1:
        step_val = 1
        logger.warning(f"dFC_StdDev (S {sid}): Calculated step points ({step_val}) too small. Adjusted to 1.")

    if n_tp < win_points_val:
        logger.warning(f"dFC_StdDev (S {sid}): Timepoints ({n_tp}) < window length ({win_points_val}). Skipping.")
        return np.zeros((n_rois, n_rois), dtype=np.float32) if n_rois > 0 else None

    num_windows = (n_tp - win_points_val) // step_val + 1
    if num_windows < 2:
        logger.warning(f"dFC_StdDev (S {sid}): Fewer than 2 windows ({num_windows}) can be formed. StdDev would be trivial (0). Skipping and returning zero matrix.")
        return np.zeros((n_rois, n_rois), dtype=np.float32) if n_rois > 0 else None

    window_corr_matrices_list = []

    for idx in range(num_windows):
        start_idx = idx * step_val
        end_idx = start_idx + win_points_val
        window_ts = ts_subject[start_idx:end_idx, :]

        if window_ts.shape[0] < 2: continue

        try:
            corr_matrix_window = np.corrcoef(window_ts, rowvar=False)
            if corr_matrix_window.ndim < 2 or corr_matrix_window.shape != (n_rois, n_rois):
                logger.warning(f"dFC_StdDev (S {sid}), Window {idx}: corrcoef returned unexpected shape {corr_matrix_window.shape}. Skipping this window for StdDev.")
                continue
            else:
                corr_matrix_window = np.nan_to_num(corr_matrix_window.astype(np.float32), nan=0.0)

            np.fill_diagonal(corr_matrix_window, 0)
            window_corr_matrices_list.append(corr_matrix_window)
        except Exception as e:
            logger.error(f"dFC_StdDev (S {sid}), Window {idx}: Error calculating/processing correlation: {e}")

    if len(window_corr_matrices_list) < 2:
        logger.warning(f"dFC_StdDev (S {sid}): Fewer than 2 valid windowed correlation matrices were calculated ({len(window_corr_matrices_list)}). Cannot compute StdDev. Returning zero matrix.")
        return np.zeros((n_rois, n_rois), dtype=np.float32) if n_rois > 0 else None

    stacked_corr_matrices = np.stack(window_corr_matrices_list, axis=0)
    std_dev_matrix = np.std(stacked_corr_matrices, axis=0).astype(np.float32)
    np.fill_diagonal(std_dev_matrix, 0)

    return std_dev_matrix

def _granger_pair(ts1, ts2, maxlag, sid, i, j):
    f_ij, f_ji = 0.0, 0.0
    try:
        data_for_ij = np.column_stack([ts2, ts1])
        if np.any(np.std(data_for_ij, axis=0) < 1e-6):
             pass
        else:
            granger_result_ij = grangercausalitytests(data_for_ij, maxlag=[maxlag], verbose=False)
            f_ij = granger_result_ij[maxlag][0]['ssr_ftest'][0]
        
        data_for_ji = np.column_stack([ts1, ts2])
        if np.any(np.std(data_for_ji, axis=0) < 1e-6):
            pass
        else:
            granger_result_ji = grangercausalitytests(data_for_ji, maxlag=[maxlag], verbose=False)
            f_ji = granger_result_ji[maxlag][0]['ssr_ftest'][0]
            
        return f_ij, f_ji
    except Exception as e:
        return 0.0, 0.0

def calculate_granger_f_matrix(ts_subject: np.ndarray, maxlag: int, sid: str) -> Optional[np.ndarray]:
    n_tp, n_rois = ts_subject.shape

    # --- Granger Lag Change: Fixed to 2 as per suggestion ---
    actual_lag = min(maxlag, 2)
    
    if actual_lag == 0:
        logger.warning(f"Granger (S {sid}): Adapted lag became 0 due to very short TPs ({n_tp}). Cannot compute Granger. Returning zero matrix.")
        return np.zeros((n_rois, n_rois), dtype=np.float32) if n_rois > 0 else None

    min_tp_for_granger = 2 * (actual_lag + 1) + 5
    if n_tp < min_tp_for_granger:
        logger.warning(f"Granger (S {sid}): Too few TPs ({n_tp}) for adapted lag {actual_lag} and {n_rois} ROIs. Need > ~{min_tp_for_granger}. Skipping.")
        return np.zeros((n_rois, n_rois), dtype=np.float32) if n_rois > 0 else None

    gc_mat_symmetric = np.zeros((n_rois, n_rois), dtype=np.float32)

    tasks = []
    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            tasks.append({'i': i, 'j': j,
                          'ts1': ts_subject[:, i],
                          'ts2': ts_subject[:, j],
                          'maxlag': actual_lag, 'sid': sid})

    # Use n_jobs=1 to avoid oversubscription when running in ProcessPoolExecutor
    # as parallelization is already handled at the subject level.
    n_jobs_granger = 1 

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            results_pairs = Parallel(n_jobs=n_jobs_granger)(
                delayed(_granger_pair)(task['ts1'], task['ts2'], task['maxlag'], task['sid'], task['i'], task['j'])
                for task in tasks
            )
    except Exception as e_parallel_granger:
        logger.error(f"Granger (S {sid}): Error during parallel Granger calculation: {e_parallel_granger}. Falling back to serial.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            results_pairs = [_granger_pair(task['ts1'], task['ts2'], task['maxlag'], task['sid'], task['i'], task['j']) for task in tasks]

    for k, task in enumerate(tasks):
        i, j = task['i'], task['j']
        f_ij, f_ji = results_pairs[k]
        f_sym = (f_ij + f_ji) / 2.0
        gc_mat_symmetric[i, j] = gc_mat_symmetric[j, i] = f_sym

    np.fill_diagonal(gc_mat_symmetric, 0)
    return gc_mat_symmetric

# --- Add this function before the 'main' function ---
def process_single_subject_pipeline(subject_tuple: Tuple[int, pd.Series]) -> Dict[str, Any]:
    """
    Processes a single subject's fMRI time series data to extract connectivity matrices
    for various channels, normalizes them, and saves the resulting tensor.
    This function is intended to be run in parallel for each subject.
    """
    global ROI_SIGNALS_DIR_PATH_AAL3, ROI_FILENAME_TEMPLATE, POSSIBLE_ROI_KEYS, \
           TR_SECONDS, LOW_CUT_HZ, HIGH_CUT_HZ, FILTER_ORDER, TAPER_ALPHA, \
           APPLY_HRF_DECONVOLUTION, HRF_MODEL, \
           USE_GRANGER_CHANNEL, GRANGER_MAX_LAG, N_NEIGHBORS_MI, \
           DFC_WIN_POINTS_SEC, DFC_STEP_SEC, \
           CONNECTIVITY_CHANNEL_NAMES, N_CHANNELS, FINAL_N_ROIS_EXPECTED, \
           AAL3_ROI_ORDER_MAPPING, OUTPUT_CONNECTIVITY_DIR_NAME, TOTAL_CPU_CORES

    # Extract subject ID and metadata from the tuple
    _ , subject_data = subject_tuple
    subject_id = str(subject_data['SubjectID']).strip()
    timepoints_count_from_qc = int(subject_data['Timepoints']) # Get TPs from QC report

    logger.info(f"S {subject_id}: Starting processing pipeline.")

    # Initialize result dictionary for this subject
    subject_result = {
        "id": subject_id,
        "status_overall": "FAILED",
        "detail_preprocessing": "",
        "errors_connectivity_calc": {},
        "path_saved_tensor": None,
        "final_timepoints": 0,
        "num_rois_processed": 0
    }

    try:
        # 1. Load and Preprocess Time Series
        processed_ts, original_tp_count, preproc_status_msg, preproc_success = \
            load_and_preprocess_single_subject_series(
                subject_id=subject_id,
                current_roi_signals_dir_path=ROI_SIGNALS_DIR_PATH_AAL3,
                current_roi_filename_template=ROI_FILENAME_TEMPLATE,
                possible_roi_keys_list=POSSIBLE_ROI_KEYS,
                eff_conn_max_lag_val=GRANGER_MAX_LAG, # Use max lag for Granger for general min_len check
                tr_seconds_val=TR_SECONDS, low_cut_val=LOW_CUT_HZ, high_cut_val=HIGH_CUT_HZ,
                filter_order_val=FILTER_ORDER,
                apply_hrf_deconv_val=APPLY_HRF_DECONVOLUTION,
                hrf_model_type_val=HRF_MODEL,
                taper_alpha_val=TAPER_ALPHA,
                roi_order_info=AAL3_ROI_ORDER_MAPPING
            )

        subject_result["detail_preprocessing"] = preproc_status_msg
        subject_result["final_timepoints"] = processed_ts.shape[0] if processed_ts is not None else 0
        subject_result["num_rois_processed"] = processed_ts.shape[1] if processed_ts is not None else 0

        if not preproc_success or processed_ts is None:
            subject_result["status_overall"] = "FAILED_PREPROCESSING"
            logger.error(f"S {subject_id}: Preprocessing failed. Reason: {preproc_status_msg}")
            return subject_result

        current_n_rois = processed_ts.shape[1]
        if FINAL_N_ROIS_EXPECTED is not None and current_n_rois != FINAL_N_ROIS_EXPECTED:
            logger.error(f"S {subject_id}: ROI count after preprocessing ({current_n_rois}) "
                         f"does not match expected ({FINAL_N_ROIS_EXPECTED}). Skipping connectivity for this subject.")
            subject_result["status_overall"] = "FAILED_ROI_MISMATCH"
            subject_result["errors_connectivity_calc"]["ROI_count_mismatch"] = f"Expected {FINAL_N_ROIS_EXPECTED}, got {current_n_rois}"
            return subject_result

        # 2. Calculate Connectivity Matrices & Normalize (per-subject, per-channel)
        subject_connectivity_matrices = {}
        successful_channels_count = 0
        current_expected_rois_for_assembly = FINAL_N_ROIS_EXPECTED # Use the confirmed final ROI count

        for channel_name in CONNECTIVITY_CHANNEL_NAMES:
            matrix = None
            error_calc = None
            calc_start_time = time.time()

            # Call appropriate calculation function
            if channel_name == PEARSON_OMST_CHANNEL_NAME_PRIMARY:
                matrix = calculate_pearson_omst_signed_weighted(processed_ts, subject_id)
            elif channel_name == PEARSON_OMST_FALLBACK_NAME:
                matrix = calculate_pearson_full_fisher_z_signed(processed_ts, subject_id)
            elif channel_name == "MI_KNN_Symmetric":
                matrix = calculate_mi_knn_connectivity(processed_ts, N_NEIGHBORS_MI, subject_id)
            elif channel_name == "dFC_AbsDiffMean":
                matrix = calculate_custom_dfc_abs_diff_mean(processed_ts, DFC_WIN_POINTS_SEC, DFC_STEP_SEC, subject_id, TR_SECONDS)
            elif channel_name == "dFC_StdDev":
                matrix = calculate_dfc_std_dev(processed_ts, DFC_WIN_POINTS_SEC, DFC_STEP_SEC, subject_id, TR_SECONDS)
            elif channel_name == f"Granger_F_lag{GRANGER_MAX_LAG}":
                matrix = calculate_granger_f_matrix(processed_ts, GRANGER_MAX_LAG, subject_id)
            else:
                error_calc = f"Unknown channel type: {channel_name}"
                logger.error(f"S {subject_id}, Channel {channel_name}: {error_calc}")

            if matrix is None:
                if error_calc is None: # If not already set by unknown channel
                    error_calc = f"Calculation returned None or failed after {time.time() - calc_start_time:.2f}s."
                subject_result["errors_connectivity_calc"][channel_name] = error_calc
                logger.warning(f"S {subject_id}, Channel {channel_name}: Skipping due to calculation error: {error_calc}")
                # Append a zero matrix to maintain shape consistency in tensor stacking
                subject_connectivity_matrices[channel_name] = np.zeros((current_n_rois, current_n_rois), dtype=np.float32)
                continue

            if matrix.shape != (current_n_rois, current_n_rois):
                error_calc = f"Calculated matrix shape {matrix.shape} does not match expected {current_n_rois}x{current_n_rois}."
                subject_result["errors_connectivity_calc"][channel_name] = error_calc
                logger.error(f"S {subject_id}, Channel {channel_name}: {error_calc}. Returning zero matrix for this channel.")
                # Append a zero matrix to maintain shape consistency
                subject_connectivity_matrices[channel_name] = np.zeros((current_n_rois, current_n_rois), dtype=np.float32)
                continue
            
            # Apply per-subject, per-channel RobustScaler (off-diagonal)
            # This normalization is applied to each individual matrix before stacking into the subject tensor.
            # The global inter-channel normalization will be handled by the modeling script to prevent data leakage.
            try:
                off_diag_mask = ~np.eye(current_n_rois, dtype=bool)
                off_diag_values = matrix[off_diag_mask].reshape(-1, 1)

                if off_diag_values.size > 0 and np.std(off_diag_values) > 1e-9:
                    scaler = RobustScaler()
                    scaled_off_diag = scaler.fit_transform(off_diag_values)
                    normalized_matrix = matrix.copy()
                    normalized_matrix[off_diag_mask] = scaled_off_diag.flatten()
                    subject_connectivity_matrices[channel_name] = normalized_matrix.astype(np.float32)
                    logger.info(f"S {subject_id}, Channel {channel_name}: RobustScaler applied (off-diagonal).")
                else:
                    logger.warning(f"S {subject_id}, Channel {channel_name}: Skipping RobustScaler due to zero/low variance or empty off-diagonal elements. Using original matrix.")
                    subject_connectivity_matrices[channel_name] = matrix.astype(np.float32)

            except Exception as e_norm:
                error_calc = f"Normalization with RobustScaler failed: {e_norm}. Using unscaled matrix."
                subject_result["errors_connectivity_calc"][f"{channel_name}_norm"] = error_calc
                logger.error(f"S {subject_id}, Channel {channel_name}: {error_calc}")
                subject_connectivity_matrices[channel_name] = matrix.astype(np.float32) # Fallback to unscaled


            # 3. Reorder Connectivity Matrices if mapping is available
            if AAL3_ROI_ORDER_MAPPING and AAL3_ROI_ORDER_MAPPING.get("new_order_indices") is not None:
                new_indices = AAL3_ROI_ORDER_MAPPING["new_order_indices"]
                if len(new_indices) == current_n_rois: # Ensure indices match current ROI count
                    subject_connectivity_matrices[channel_name] = _reorder_connectivity_matrix_by_network(
                        subject_connectivity_matrices[channel_name], new_indices, subject_id, channel_name
                    )
                else:
                    logger.warning(f"S {subject_id}, Channel {channel_name}: ROI reordering indices length mismatch ({len(new_indices)}) with matrix ROIs ({current_n_rois}). Skipping reordering for this matrix.")
            
            successful_channels_count += 1
            logger.info(f"S {subject_id}, Channel {channel_name}: Calculated in {time.time() - calc_start_time:.2f}s.")

        if successful_channels_count == 0:
            subject_result["status_overall"] = "FAILED_NO_CHANNELS_CALCULATED"
            logger.critical(f"S {subject_id}: No connectivity channels could be calculated successfully.")
            return subject_result

        # 4. Stack Channels into a single tensor
        # Ensure that the order of channels in the stacked tensor matches CONNECTIVITY_CHANNEL_NAMES
        stacked_matrices_list = []
        for ch_name in CONNECTIVITY_CHANNEL_NAMES:
            if ch_name in subject_connectivity_matrices:
                stacked_matrices_list.append(subject_connectivity_matrices[ch_name])
            else:
                # This case should ideally not happen if logic above is correct,
                # but adding a zero matrix ensures tensor shape consistency.
                logger.warning(f"S {subject_id}: Channel '{ch_name}' not found in calculated matrices. Appending zero matrix.")
                stacked_matrices_list.append(np.zeros((current_n_rois, current_n_rois), dtype=np.float32))

        subject_connectivity_tensor = np.stack(stacked_matrices_list, axis=0).astype(np.float32)
        del stacked_matrices_list; gc.collect() # Free up memory

        if subject_connectivity_tensor.shape != (N_CHANNELS, current_n_rois, current_n_rois):
            error_msg = f"Final tensor shape mismatch: {subject_connectivity_tensor.shape}. Expected ({N_CHANNELS}, {current_n_rois}, {current_n_rois})."
            subject_result["status_overall"] = "FAILED_TENSOR_SHAPE"
            subject_result["errors_connectivity_calc"]["final_tensor_shape"] = error_msg
            logger.critical(f"S {subject_id}: {error_msg}")
            return subject_result
        
        # 5. Save Individual Subject Tensor
        output_individual_tensors_dir = BASE_PATH_AAL3 / OUTPUT_CONNECTIVITY_DIR_NAME / "individual_subject_tensors"
        output_individual_tensors_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        
        output_filename = f"subject_{subject_id}_conn_tensor.npz"
        output_filepath = output_individual_tensors_dir / output_filename

        save_start_time = time.time()
        np.savez_compressed(output_filepath, tensor_data=subject_connectivity_tensor)
        logger.info(f"S {subject_id}: Tensor saved to {output_filepath.name} in {time.time() - save_start_time:.2f}s.")
        
        subject_result["status_overall"] = "SUCCESS_ALL_PROCESSED_AND_SAVED"
        subject_result["path_saved_tensor"] = str(output_filepath)
        del processed_ts, subject_connectivity_tensor; gc.collect() # Free up more memory

    except Exception as e_overall:
        subject_result["status_overall"] = "CRITICAL_PIPELINE_ERROR"
        subject_result["detail_preprocessing"] += f" | Critical pipeline error: {str(e_overall)}"
        logger.critical(f"S {subject_id}: CRITICAL PIPELINE ERROR: {e_overall}", exc_info=True)

    return subject_result

# --- Funciones para Reordenamiento de ROIs ---
def _get_aal3_network_mapping_and_order() -> Optional[Dict[str, Any]]:
    """
    Carga/define el mapeo de ROIs AAL3 a redes funcionales Yeo-17 y el nuevo orden.
    Requiere:
        - AAL3_NIFTI_PATH: Ruta al archivo NIfTI del atlas AAL3.
        - AAL3_META_PATH: Ruta al archivo .txt con metadatos de AAL3 (para nombres y colores).
        - Variables globales: VALID_AAL3_ROI_INFO_DF_166, INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166, FINAL_N_ROIS_EXPECTED.
    SUGERENCIA (Tesis 1): Considerar también ordenar por gradientes de conectividad (e.g., Margulies et al.)
                          o usar parcelaciones multiescala (e.g., Schaefer + Yeo17).
    """
    logger.info("Attempting to map AAL3 ROIs to Yeo-17 networks and reorder.")

    if not AAL3_NIFTI_PATH.exists():
        logger.error(f"AAL3 NIfTI file NOT found at: {AAL3_NIFTI_PATH}. Cannot perform ROI reordering.")
        return None
    # FINAL_N_ROIS_EXPECTED se inicializa antes de esta función en _initialize_aal3_roi_processing_info
    if VALID_AAL3_ROI_INFO_DF_166 is None or INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166 is None or FINAL_N_ROIS_EXPECTED is None:
        logger.error("Global AAL3 processing variables (VALID_AAL3_ROI_INFO_DF_166, etc.) not initialized. Cannot perform ROI reordering.")
        return None

    try:
        logger.info("Fetching Yeo 17-network atlas...")
        yeo_atlas_obj = fetch_atlas_yeo_2011()
        yeo_img = nib.load(yeo_atlas_obj.thick_17)
        yeo_data = yeo_img.get_fdata().astype(int)
        logger.info(f"Yeo-17 atlas loaded. Shape: {yeo_data.shape}, Affine: \n{yeo_img.affine}")

        logger.info(f"Loading AAL3 NIfTI from: {AAL3_NIFTI_PATH}")
        aal_img_orig = nib.load(AAL3_NIFTI_PATH)
        logger.info(f"Original AAL3 NIfTI atlas loaded. Shape: {aal_img_orig.shape}, Affine: \n{aal_img_orig.affine}")

        if not np.allclose(aal_img_orig.affine, yeo_img.affine, atol=1e-3) or aal_img_orig.shape != yeo_img.shape:
            logger.warning("Affines or shapes of AAL3 and Yeo atlases do not match. "
                           "Attempting to resample AAL3 to Yeo space using nearest neighbor interpolation.")
            try:
                aal_img_resampled = nli_image.resample_to_img(aal_img_orig, yeo_img, interpolation='nearest')
                aal_data = aal_img_resampled.get_fdata().astype(int)
                logger.info(f"AAL3 atlas resampled. New Shape: {aal_data.shape}, New Affine: \n{aal_img_resampled.affine}")
            except Exception as e_resample:
                logger.error(f"Failed to resample AAL3 atlas: {e_resample}. ROI reordering will be skipped.")
                return None
        else:
            aal_data = aal_img_orig.get_fdata().astype(int)
            logger.info("AAL3 and Yeo atlases appear to be in the same space. No resampling performed.")

        final_aal3_rois_info_df = VALID_AAL3_ROI_INFO_DF_166.drop(INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166).reset_index(drop=True)
        
        if len(final_aal3_rois_info_df) != FINAL_N_ROIS_EXPECTED:
            logger.error(f"Mismatch in expected final ROI count. Expected {FINAL_N_ROIS_EXPECTED}, "
                         f"derived {len(final_aal3_rois_info_df)} from VALID_AAL3_ROI_INFO_DF_166. Cannot proceed with reordering.")
            return None

        original_131_aal3_colors = final_aal3_rois_info_df['color'].tolist()
        original_131_aal3_names = final_aal3_rois_info_df['nom_c'].tolist()
        
        logger.info(f"Mapping {len(original_131_aal3_colors)} AAL3 ROIs to Yeo-17 networks...")
        roi_network_mapping = []

        for aal3_idx, aal3_color in enumerate(original_131_aal3_colors):
            aal3_name = original_131_aal3_names[aal3_idx]
            aal3_roi_mask = (aal_data == aal3_color)
            
            if not np.any(aal3_roi_mask):
                logger.warning(f"AAL3 ROI color {aal3_color} ('{aal3_name}') not found in (potentially resampled) AAL3 NIfTI data. Assigning to NonCortical.")
                roi_network_mapping.append((aal3_color, aal3_name, 0, YEO17_LABELS_TO_NAMES[0], 0.0, aal3_idx))
                continue

            overlapping_yeo_voxels = yeo_data[aal3_roi_mask]
            
            if overlapping_yeo_voxels.size > 0:
                unique_yeo_labels, counts = np.unique(overlapping_yeo_voxels, return_counts=True)
                valid_overlap_mask = unique_yeo_labels != 0
                unique_yeo_labels = unique_yeo_labels[valid_overlap_mask]
                counts = counts[valid_overlap_mask]

                if len(counts) > 0:
                    winner_yeo_label_idx = np.argmax(counts)
                    winner_yeo_label = unique_yeo_labels[winner_yeo_label_idx]
                    total_roi_voxels = np.sum(aal3_roi_mask)
                    overlap_percentage = (counts[winner_yeo_label_idx] / total_roi_voxels) * 100 if total_roi_voxels > 0 else 0.0
                    yeo17_name = YEO17_LABELS_TO_NAMES.get(winner_yeo_label, f"UnknownYeo{winner_yeo_label}")
                    if overlap_percentage < 5.0 :
                        logger.debug(f"AAL3 ROI {aal3_color} ('{aal3_name}') has low overlap ({overlap_percentage:.2f}%) with Yeo-17 Label {winner_yeo_label} ('{yeo17_name}'). May be subcortical or cerebellar.")
                else:
                    winner_yeo_label = 0
                    yeo17_name = YEO17_LABELS_TO_NAMES[0]
                    overlap_percentage = 0.0
                    logger.debug(f"AAL3 ROI {aal3_color} ('{aal3_name}') has no overlap with cortical Yeo-17 networks. Assigning to NonCortical.")
            else:
                winner_yeo_label = 0
                yeo17_name = YEO17_LABELS_TO_NAMES[0]
                overlap_percentage = 0.0
                logger.warning(f"AAL3 ROI {aal3_color} ('{aal3_name}') mask is empty in AAL3 NIfTI data. Assigning to NonCortical.")
            
            roi_network_mapping.append((aal3_color, aal3_name, winner_yeo_label, yeo17_name, overlap_percentage, aal3_idx ))
        
        roi_network_mapping_sorted = sorted(roi_network_mapping, key=lambda x: (x[2] == 0, x[2], x[0]))

        new_order_indices = [item[5] for item in roi_network_mapping_sorted]
        roi_names_new_order = [item[1] for item in roi_network_mapping_sorted]
        network_labels_new_order = [item[3] for item in roi_network_mapping_sorted]
        
        if len(new_order_indices) != FINAL_N_ROIS_EXPECTED or len(set(new_order_indices)) != FINAL_N_ROIS_EXPECTED:
            logger.error("Error en la generación de new_order_indices para reordenamiento. Longitud o unicidad incorrecta.")
            return None

        logger.info("Successfully mapped AAL3 ROIs to Yeo-17 networks and determined new ROI order.")
        
        mapping_df = pd.DataFrame(roi_network_mapping_sorted, columns=['AAL3_Color', 'AAL3_Name', 'Yeo17_Label', 'Yeo17_Network', 'Overlap_Percent', 'Original_Index_0_N'])
        mapping_filename = BASE_PATH_AAL3 / f"aal3_{FINAL_N_ROIS_EXPECTED}_to_yeo17_mapping.csv"
        try:
            mapping_filename.parent.mkdir(parents=True, exist_ok=True)
            mapping_df.to_csv(mapping_filename, index=False)
            logger.info(f"AAL3 to Yeo-17 mapping saved to: {mapping_filename}")
        except Exception as e_save_map:
            logger.warning(f"Could not save AAL3 to Yeo-17 mapping CSV: {e_save_map}")

        return {
            'order_name': 'aal3_to_yeo17_overlap_sorted',
            'roi_indices_original_order': list(range(FINAL_N_ROIS_EXPECTED)),
            'roi_names_original_order': original_131_aal3_names,
            'roi_names_new_order': roi_names_new_order,
            'network_labels_new_order': network_labels_new_order,
            'new_order_indices': new_order_indices
        }

    except FileNotFoundError as e_fnf:
        logger.error(f"Atlas file not found during ROI reordering: {e_fnf}. ROI reordering will be skipped.")
        return None
    except Exception as e:
        logger.error(f"Error during ROI reordering: {e}", exc_info=True)
        return None

def _reorder_rois_by_network_for_timeseries(
    timeseries_data: np.ndarray,
    new_order_indices: List[int],
    subject_id: str) -> np.ndarray:
    if new_order_indices is None or len(new_order_indices) != timeseries_data.shape[1]:
        return timeseries_data
    
    logger.info(f"S {subject_id}: Reordenando series temporales de ROIs ({timeseries_data.shape}) según el nuevo orden de redes (longitud de índices: {len(new_order_indices)}).")
    return timeseries_data[:, new_order_indices]

def _reorder_connectivity_matrix_by_network(
    matrix: np.ndarray,
    new_order_indices: List[int],
    subject_id: str,
    channel_name: str) -> np.ndarray:
    if new_order_indices is None or len(new_order_indices) != matrix.shape[0]:
        return matrix

    logger.info(f"S {subject_id}, Canal {channel_name}: Reordenando matriz de conectividad ({matrix.shape}) según el nuevo orden de redes (longitud de índices: {len(new_order_indices)}).")
    # --- BUG FIX: Corrected np.ix_ usage (one ix_ call only) ---
    return matrix[np.ix_(new_order_indices, new_order_indices)]

# --- Funciones para Carga y Preprocesamiento de Series Temporales ---
def _load_signals_from_mat(mat_path: Path, possible_keys: List[str]) -> Optional[np.ndarray]:
    try:
        data = sio.loadmat(mat_path)
    except Exception as e_load:
        logger.error(f"Could not load .mat file: {mat_path}. Error: {e_load}")
        return None
    
    for key in possible_keys:
        if key in data and isinstance(data[key], np.ndarray) and data[key].ndim >= 2:
            return data[key].astype(np.float64)
            
    logger.warning(f"No valid signal keys {possible_keys} found in {mat_path.name}. Keys present: {list(data.keys())}")
    return None

def _orient_and_reduce_rois(
    raw_sigs: np.ndarray,
    subject_id: str,
    initial_expected_cols: int,
    aal3_missing_0based: Optional[List[int]],
    small_rois_indices_from_166: Optional[List[int]],
    final_expected_rois: Optional[int]
) -> Optional[np.ndarray]:
    if raw_sigs.ndim != 2:
        logger.warning(f"S {subject_id}: Raw signal matrix has incorrect dimensions {raw_sigs.ndim} (expected 2). Skipping.")
        return None
    
    oriented_sigs = raw_sigs.copy()
    if oriented_sigs.shape[0] == initial_expected_cols and oriented_sigs.shape[1] != initial_expected_cols:
        logger.info(f"S {subject_id}: Transposing raw matrix from {oriented_sigs.shape} to ({oriented_sigs.shape[1]}, {oriented_sigs.shape[0]}) to match (TPs, ROIs_initial).")
        oriented_sigs = oriented_sigs.T
    elif oriented_sigs.shape[1] == initial_expected_cols and oriented_sigs.shape[0] != initial_expected_cols:
        pass
    elif oriented_sigs.shape[0] == initial_expected_cols and oriented_sigs.shape[1] == initial_expected_cols:
        logger.warning(f"S {subject_id}: Raw signal matrix is square ({oriented_sigs.shape}) and matches initial_expected_cols. Assuming [Timepoints, ROIs_initial]. Careful if TPs also equals initial_expected_cols.")
    else:
        logger.warning(f"S {subject_id}: Neither dimension of raw signal matrix ({oriented_sigs.shape}) matches initial_expected_cols ({initial_expected_cols}). Skipping.")
        return None

    if oriented_sigs.shape[1] != initial_expected_cols:
        logger.warning(f"S {subject_id}: After orientation, raw ROI count ({oriented_sigs.shape[1]}) != initial_expected_cols ({initial_expected_cols}). Skipping.")
        return None
    
    if aal3_missing_0based is None:
        logger.warning(f"S {subject_id}: AAL3 missing ROI indices (0-based) not available. Skipping AAL3 known missing ROI removal. Using {oriented_sigs.shape[1]} ROIs for next step.")
        sigs_after_known_missing_removed = oriented_sigs
    else:
        try:
            sigs_after_known_missing_removed = np.delete(oriented_sigs, aal3_missing_0based, axis=1)
            if sigs_after_known_missing_removed.shape[1] != EXPECTED_ROIS_AFTER_AAL3_MISSING_REMOVAL:
                logger.warning(f"S {subject_id}: After removing known missing ROIs, shape is {sigs_after_known_missing_removed.shape}, but expected (..., {EXPECTED_ROIS_AFTER_AAL3_MISSING_REMOVAL}).")
        except IndexError as e:
            logger.error(f"S {subject_id}: IndexError removing known missing AAL3 ROIs (indices: {aal3_missing_0based}) from matrix of shape {oriented_sigs.shape}. Error: {e}. Using original {oriented_sigs.shape[1]} ROIs for next step.")
            sigs_after_known_missing_removed = oriented_sigs
            
    if small_rois_indices_from_166 is None:
        logger.warning(f"S {subject_id}: Small ROI indices (from 166-set) not available. Skipping small ROI removal. Using {sigs_after_known_missing_removed.shape[1]} ROIs.")
        sigs_final_rois = sigs_after_known_missing_removed
    elif sigs_after_known_missing_removed.shape[1] != EXPECTED_ROIS_AFTER_AAL3_MISSING_REMOVAL:
        logger.warning(f"S {subject_id}: Cannot remove small ROIs because the matrix (shape {sigs_after_known_missing_removed.shape}) does not have the expected {EXPECTED_ROIS_AFTER_AAL3_MISSING_REMOVAL} columns after first reduction step. Using current ROIs ({sigs_after_known_missing_removed.shape[1]}).")
        sigs_final_rois = sigs_after_known_missing_removed
    else:
        try:
            sigs_final_rois = np.delete(sigs_after_known_missing_removed, small_rois_indices_from_166, axis=1)
        except IndexError as e:
            logger.error(f"S {subject_id}: IndexError removing small ROIs (indices: {small_rois_indices_from_166}) from matrix of shape {sigs_after_known_missing_removed.shape}. Error: {e}. Using {sigs_after_known_missing_removed.shape[1]} ROIs.")
            sigs_final_rois = sigs_after_known_missing_removed

    if final_expected_rois is not None and sigs_final_rois.shape[1] != final_expected_rois:
        logger.warning(f"S {subject_id}: Final ROI count ({sigs_final_rois.shape[1]}) != FINAL_N_ROIS_EXPECTED ({final_expected_rois}). "
                       "This may indicate issues in AAL3 metadata or reduction logic. Proceeding with current matrix.")
    elif final_expected_rois is None:
        logger.warning(f"S {subject_id}: FINAL_N_ROIS_EXPECTED is None. Cannot validate final ROI count. Proceeding with {sigs_final_rois.shape[1]} ROIs.")
        
    return sigs_final_rois

def _bandpass_filter_signals(sigs: np.ndarray, lowcut: float, highcut: float, fs: float, order: int, subject_id: str, taper_alpha: float = 0.1) -> np.ndarray:
    nyquist_freq = 0.5 * fs
    low_norm = lowcut / nyquist_freq
    high_norm = highcut / nyquist_freq

    if not (0 < low_norm < 1 and 0 < high_norm < 1 and low_norm < high_norm):
        logger.error(f"S {subject_id}: Invalid critical frequencies for filter (low_norm={low_norm}, high_norm={high_norm}). Nyquist={nyquist_freq}. Skipping filtering.")
        return sigs
    try:
        # --- Filter Improvement: Use Chebyshev Type I filter ---
        # b, a = butter(order, [low_norm, high_norm], btype='band', analog=False) # Original
        b, a = cheby1(order, 0.5, [low_norm, high_norm], btype='band', analog=False) # Chebyshev I, 0.5 dB ripple
        
        filtered_sigs = np.zeros_like(sigs)
        padlen_required = 3 * order + 1
        
        for i in range(sigs.shape[1]):
            roi_signal = sigs[:, i].copy()
            
            # --- Tukey Window: Applied if taper_alpha > 0 ---
            if taper_alpha > 0 and len(roi_signal) > 0:
                try:
                    tukey_window = windows.tukey(len(roi_signal), alpha=taper_alpha)
                    roi_signal_tapered = roi_signal * tukey_window
                except Exception as e_taper:
                    logger.warning(f"S {subject_id}, ROI {i}: Error applying Tukey window: {e_taper}. Proceeding without taper.")
                    roi_signal_tapered = roi_signal
            else:
                roi_signal_tapered = roi_signal

            if np.all(np.isclose(roi_signal_tapered, roi_signal_tapered[0] if len(roi_signal_tapered)>0 else 0.0)):
                # --- Micro-observation: Log when signal is flat ---
                logger.debug(f"S {subject_id}, ROI {i}: Signal is constant/flat. Skipping filtering and using original (tapered) signal.")
                filtered_sigs[:, i] = roi_signal_tapered
            elif len(roi_signal_tapered) <= padlen_required :
                logger.warning(f"S {subject_id}, ROI {i}: Signal too short ({len(roi_signal_tapered)} pts, need > {padlen_required}) for filtfilt. Skipping filter for this ROI.")
                filtered_sigs[:, i] = roi_signal_tapered
            else:
                filtered_sigs[:, i] = filtfilt(b, a, roi_signal_tapered)
        return filtered_sigs
    except Exception as e:
        logger.error(f"S {subject_id}: Error during bandpass filtering: {e}. Returning original signals.", exc_info=False)
        return sigs

# --- Wiener Deconvolution Function ---
def wiener_deconv(y: np.ndarray, h: np.ndarray, lam: float) -> np.ndarray:
    """
    Performs Wiener deconvolution.
    y: observed signal
    h: impulse response (HRF kernel)
    lam: regularization parameter (noise-to-signal ratio)
    """
    if len(y) == 0 or len(h) == 0:
        return np.zeros_like(y)
    
    # Pad h to the same length as y for FFT
    h_padded = np.zeros_like(y, dtype=float)
    h_padded[:len(h)] = h

    H = np.fft.rfft(h_padded)
    Y = np.fft.rfft(y)

    # Wiener filter formula: S_hat(omega) = H*(omega) / (|H(omega)|^2 + lambda) * Y(omega)
    # Adding a small epsilon to lambda to prevent division by zero if lam is truly zero.
    denom = (np.abs(H)**2 + lam + 1e-12) # Add small epsilon to lambda
    S = np.conj(H) / denom * Y
    
    s_hat = np.fft.irfft(S, n=len(y)) # n=len(y) ensures output is same length as input
    return s_hat

def _hrf_deconvolution(sigs: np.ndarray, tr: float, hrf_model_type: str, subject_id: str) -> np.ndarray:
    logger.info(f"S {subject_id}: Attempting HRF deconvolution (Model: {hrf_model_type}, TR: {tr}s).")
    if hrf_model_type == 'glover':
        hrf_kernel = glover_hrf(tr, oversampling=1)
    elif hrf_model_type == 'spm':
        hrf_kernel = spm_hrf(tr, oversampling=1)
    else:
        logger.error(f"S {subject_id}: Unknown HRF model type '{hrf_model_type}'. Skipping deconvolution.")
        return sigs

    if len(hrf_kernel) == 0 or np.all(np.isclose(hrf_kernel, 0)):
        logger.error(f"S {subject_id}: HRF kernel is empty or all zeros for model '{hrf_model_type}'. Skipping deconvolution.")
        return sigs

    deconvolved_sigs = np.zeros_like(sigs)
    for i in range(sigs.shape[1]):
        signal_roi = sigs[:, i]
        if len(signal_roi) < len(hrf_kernel): # Still a relevant check for basic deconvolve functions
            logger.warning(f"S {subject_id}, ROI {i}: Signal length ({len(signal_roi)}) is shorter than HRF kernel length ({len(hrf_kernel)}). Skipping deconvolution for this ROI.")
            deconvolved_sigs[:, i] = signal_roi
            continue
        try:
            # --- HRF Deconvolution Improvement: Use Wiener Deconvolution ---
            # Estimate lambda based on signal variance, plus a small constant
            lam_val = 0.01 * np.var(signal_roi) + 1e-9 if np.var(signal_roi) > 0 else 1e-9
            deconvolved_roi = wiener_deconv(signal_roi, hrf_kernel, lam=lam_val)
            
            # Ensure output length matches original signal length
            if len(deconvolved_roi) < sigs.shape[0]:
                deconvolved_sigs[:, i] = np.concatenate([deconvolved_roi, np.zeros(sigs.shape[0] - len(deconvolved_roi))])
            else:
                deconvolved_sigs[:, i] = deconvolved_roi[:sigs.shape[0]]

        except Exception as e_deconv:
            logger.error(f"S {subject_id}, ROI {i}: Deconvolution failed: {e_deconv}. Using original signal for this ROI.", exc_info=False)
            deconvolved_sigs[:, i] = signal_roi
            
    logger.info(f"S {subject_id}: HRF deconvolution attempt finished.")
    return deconvolved_sigs

def _preprocess_time_series(
    sigs: np.ndarray,
    subject_id: str,
    eff_conn_max_lag_val: int,
    tr_seconds_val: float, low_cut_val: float, high_cut_val: float, filter_order_val: int,
    apply_hrf_deconv_val: bool, hrf_model_type_val: str,
    taper_alpha_val: float
) -> Optional[np.ndarray]:
    original_length, current_n_rois = sigs.shape
    fs = 1.0 / tr_seconds_val

    logger.info(f"S {subject_id}: Preprocessing. Input TPs: {original_length}, ROIs: {current_n_rois} (should be {FINAL_N_ROIS_EXPECTED}), TR: {tr_seconds_val}s.")

    sigs_processed = _bandpass_filter_signals(sigs, low_cut_val, high_cut_val, fs, filter_order_val, subject_id, taper_alpha=taper_alpha_val)

    if apply_hrf_deconv_val:
        sigs_processed = _hrf_deconvolution(sigs_processed, tr_seconds_val, hrf_model_type_val, subject_id)
        if np.isnan(sigs_processed).any() or np.isinf(sigs_processed).any():
            logger.warning(f"S {subject_id}: NaNs/Infs detected after HRF deconvolution. Cleaning by replacing with 0.0.")
            sigs_processed = np.nan_to_num(sigs_processed, nan=0.0, posinf=0.0, neginf=0.0)

    min_len_for_granger_var = eff_conn_max_lag_val + 10
    min_len_for_dfc = int(DFC_WIN_POINTS_SEC / tr_seconds_val) if DFC_WIN_POINTS_SEC > 0 else 5
    if min_len_for_dfc == 0: min_len_for_dfc = 1
    min_overall_len = max(5, min_len_for_granger_var, min_len_for_dfc)

    if sigs_processed.shape[0] < min_overall_len:
        logger.warning(f"S {subject_id}: Timepoints after processing ({sigs_processed.shape[0]}) are less than minimum required ({min_overall_len}) for all connectivity measures. Skipping subject.")
        return None

    if np.isnan(sigs_processed).any():
        logger.warning(f"S {subject_id}: NaNs detected in signals before scaling. Filling with 0.0. This might affect results.")
        sigs_processed = np.nan_to_num(sigs_processed, nan=0.0)

    try:
        scaler = StandardScaler()
        sigs_normalized = scaler.fit_transform(sigs_processed)
        if np.isnan(sigs_normalized).any():
            logger.warning(f"S {subject_id}: NaNs detected after StandardScaler. Filling with 0.0. This is unusual.")
            sigs_normalized = np.nan_to_num(sigs_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    except ValueError as e_scale:
        logger.warning(f"S {subject_id}: StandardScaler failed (e.g. all-zero data after processing): {e_scale}. Attempting column-wise scaling or zeroing.")
        sigs_normalized = np.zeros_like(sigs_processed, dtype=np.float32)
        for i in range(sigs_processed.shape[1]):
            col_data = sigs_processed[:, i].reshape(-1,1)
            if np.std(col_data) > 1e-9:
                try:
                    sigs_normalized[:, i] = StandardScaler().fit_transform(col_data).flatten()
                except Exception as e_col_scale:
                    logger.error(f"S {subject_id}, ROI {i}: Column-wise scaling failed: {e_col_scale}. Setting to zero.")
                    sigs_normalized[:, i] = 0.0
            else:
                sigs_normalized[:, i] = 0.0
        if np.isnan(sigs_normalized).any():
            sigs_normalized = np.nan_to_num(sigs_normalized, nan=0.0, posinf=0.0, neginf=0.0)

    return sigs_normalized.astype(np.float32)

def load_and_preprocess_single_subject_series(
    subject_id: str,
    current_roi_signals_dir_path: Path, current_roi_filename_template: str,
    possible_roi_keys_list: List[str],
    eff_conn_max_lag_val: int,
    tr_seconds_val: float, low_cut_val: float, high_cut_val: float, filter_order_val: int,
    apply_hrf_deconv_val: bool, hrf_model_type_val: str,
    taper_alpha_val: float,
    roi_order_info: Optional[Dict[str, Any]]
) -> Tuple[Optional[np.ndarray], int, str, bool]:
    mat_path = current_roi_signals_dir_path / current_roi_filename_template.format(subject_id=subject_id)
    if not mat_path.exists():
        return None, 0, f"MAT file not found: {mat_path.name}", False

    try:
        loaded_sigs_raw_170 = _load_signals_from_mat(mat_path, possible_roi_keys_list)
        if loaded_sigs_raw_170 is None:
            return None, 0, f"No valid signal keys or load error in {mat_path.name}", False

        sigs_reduced_rois = _orient_and_reduce_rois(
            loaded_sigs_raw_170, subject_id,
            RAW_DATA_EXPECTED_COLUMNS,
            AAL3_MISSING_INDICES_0BASED,
            INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166,
            FINAL_N_ROIS_EXPECTED
        )
        del loaded_sigs_raw_170; gc.collect()
        if sigs_reduced_rois is None:
            return None, 0, f"ROI orientation, reduction, or validation failed for S {subject_id}.", False

        if FINAL_N_ROIS_EXPECTED is not None and sigs_reduced_rois.shape[1] != FINAL_N_ROIS_EXPECTED:
            error_msg = (f"S {subject_id}: Post-reduction ROI count ({sigs_reduced_rois.shape[1]}) "
                         f"does not match FINAL_N_ROIS_EXPECTED ({FINAL_N_ROIS_EXPECTED}).")
            logger.error(error_msg)
            return None, 0, error_msg, False
        elif FINAL_N_ROIS_EXPECTED is None:
            logger.warning(f"S {subject_id}: FINAL_N_ROIS_EXPECTED is None, cannot strictly validate ROI count. Proceeding with {sigs_reduced_rois.shape[1]} ROIs.")

        if roi_order_info and roi_order_info.get("new_order_indices") is not None:
            new_indices = roi_order_info["new_order_indices"]
            if len(new_indices) == sigs_reduced_rois.shape[1]:
                sigs_reduced_rois = _reorder_rois_by_network_for_timeseries(sigs_reduced_rois, new_indices, subject_id)

        original_tp_count_after_preproc = sigs_reduced_rois.shape[0]

        sigs_processed = _preprocess_time_series(
            sigs_reduced_rois,
            subject_id, eff_conn_max_lag_val,
            tr_seconds_val, low_cut_val, high_cut_val, filter_order_val,
            apply_hrf_deconv_val, hrf_model_type_val,
            taper_alpha_val=taper_alpha_val
        )
        del sigs_reduced_rois; gc.collect()
        if sigs_processed is None:
            return None, 0, f"Preprocessing (filtering, scaling) failed for S {subject_id}. Original TPs: {original_tp_count_after_preproc}", False

        final_shape_str = f"({sigs_processed.shape[0]}, {sigs_processed.shape[1]})"
        if FINAL_N_ROIS_EXPECTED is not None and sigs_processed.shape[1] != FINAL_N_ROIS_EXPECTED:
            error_msg = (f"S {subject_id}: Processed signal ROI count ({sigs_processed.shape[1]}) "
                         f"mismatches FINAL_N_ROIS_EXPECTED ({FINAL_N_ROIS_EXPECTED}) after all preprocessing. "
                         "This could indicate an issue with ROI reordering logic if active, or prior reduction.")
            logger.error(error_msg)
            return None, 0, error_msg, False
        elif FINAL_N_ROIS_EXPECTED is None:
            logger.warning(f"S {subject_id}: FINAL_N_ROIS_EXPECTED is None, cannot strictly validate ROI count. Proceeding with {sigs_processed.shape[1]} ROIs.")

        logger.info(f"S {subject_id}: Successfully loaded and preprocessed. Original TPs: {original_tp_count_after_preproc}, Final Shape for conn: {final_shape_str}")
        return sigs_processed, original_tp_count_after_preproc, f"OK. Original TPs: {original_tp_count_after_preproc}, final shape for conn: {final_shape_str}", True

    except Exception as e:
        logger.error(f"Unhandled exception during load_and_preprocess for S {subject_id} ({mat_path.name}): {e}", exc_info=True)
        return None, 0, f"Exception processing {mat_path.name}: {str(e)}", False


# --- 1. Subject Metadata Loading and Merging ---
def load_metadata(
    subject_meta_csv_path: Path,
    qc_report_csv_path: Path) -> Optional[pd.DataFrame]:
    logger.info("--- Starting Subject Metadata Loading and QC Integration ---")
    try:
        if not subject_meta_csv_path.exists():
            logger.critical(f"Subject metadata CSV file NOT found: {subject_meta_csv_path}")
            return None
        if not qc_report_csv_path.exists():
            logger.critical(f"QC report CSV file NOT found: {qc_report_csv_path}")
            return None

        subjects_db_df = pd.read_csv(subject_meta_csv_path)
        subjects_db_df['SubjectID'] = subjects_db_df['SubjectID'].astype(str).str.strip()
        logger.info(f"Loaded main metadata from {subject_meta_csv_path}. Shape: {subjects_db_df.shape}")
        if 'SubjectID' not in subjects_db_df.columns:
            logger.critical("Column 'SubjectID' missing in main metadata CSV.")
            return None
        if 'ResearchGroup' not in subjects_db_df.columns:
            logger.warning("Column 'ResearchGroup' missing in main metadata CSV. May be needed for downstream VAE tasks.")

        qc_df = pd.read_csv(qc_report_csv_path)
        logger.info(f"Loaded QC report from {qc_report_csv_path}. Shape: {qc_df.shape}")

        if 'Subject' in qc_df.columns and 'SubjectID' not in qc_df.columns:
            logger.info("Found 'Subject' column in QC report, renaming to 'SubjectID'.")
            qc_df.rename(columns={'Subject': 'SubjectID'}, inplace=True)
            
        if 'SubjectID' in qc_df.columns:
            qc_df['SubjectID'] = qc_df['SubjectID'].astype(str).str.strip()
        else:
            logger.critical("Neither 'Subject' nor 'SubjectID' column found in QC report CSV.")
            return None
            
        essential_qc_cols = ['SubjectID', 'ToDiscard_Overall', 'TimePoints']
        if not all(col in qc_df.columns for col in essential_qc_cols):
            logger.critical(f"Essential columns ({essential_qc_cols}) missing in QC report CSV.")
            return None

        merged_df = pd.merge(subjects_db_df, qc_df, on='SubjectID', how='inner', suffixes=('_meta', '_qc'))
        
        if 'TimePoints_qc' in merged_df.columns:
            merged_df['Timepoints_final_for_script'] = merged_df['TimePoints_qc']
        elif 'TimePoints' in merged_df.columns:
            merged_df['Timepoints_final_for_script'] = merged_df['TimePoints']
        else:
            logger.critical("Definitive 'TimePoints' column from QC report could not be identified after merge.")
            return None
            
        merged_df['Timepoints_final_for_script'] = pd.to_numeric(merged_df['Timepoints_final_for_script'], errors='coerce').fillna(0).astype(int)

        initial_subject_count = len(merged_df)
        logger.info("SUGGESTION (Data Quality): Review 'ToDiscard_Overall' criteria from QC script. "
                    "Consider if stricter thresholds for subject exclusion (e.g., based on percentage of multivariate outliers, mean FD, DVARS) "
                    "would yield a cleaner dataset for modeling, even if N is slightly reduced. Ensure discard rate is similar across groups.")
        subjects_passing_qc_df = merged_df[merged_df['ToDiscard_Overall'] == False].copy()
        num_discarded = initial_subject_count - len(subjects_passing_qc_df)
        
        logger.info(f"Total subjects after merge: {initial_subject_count}")
        logger.info(f"Subjects discarded based on QC ('ToDiscard_Overall' == True): {num_discarded}")
        logger.info(f"Subjects passing QC and to be processed: {len(subjects_passing_qc_df)}")

        if subjects_passing_qc_df.empty:
            logger.warning("No subjects passed QC. Check your QC criteria and report.")
            return None
            
        min_tp_after_qc = subjects_passing_qc_df['Timepoints_final_for_script'].min()
        max_tp_after_qc = subjects_passing_qc_df['Timepoints_final_for_script'].max()
        logger.info(f"Timepoints for subjects passing QC (from QC report): Min={min_tp_after_qc}, Max={max_tp_after_qc}.")

        final_cols_to_keep = ['SubjectID']
        subjects_passing_qc_df.rename(columns={'Timepoints_final_for_script': 'Timepoints'}, inplace=True)
        final_cols_to_keep.append('Timepoints')

        if 'ResearchGroup_meta' in subjects_passing_qc_df.columns:
            subjects_passing_qc_df.rename(columns={'ResearchGroup_meta': 'ResearchGroup'}, inplace=True)
        elif 'ResearchGroup_qc' in subjects_passing_qc_df.columns and 'ResearchGroup' not in subjects_passing_qc_df.columns:
            subjects_passing_qc_df.rename(columns={'ResearchGroup_qc': 'ResearchGroup'}, inplace=True)
            
        if 'ResearchGroup' in subjects_passing_qc_df.columns:
            final_cols_to_keep.append('ResearchGroup')
        else:
            logger.warning("Creating placeholder 'ResearchGroup' column as it was not found. This is important for classification.")
            subjects_passing_qc_df['ResearchGroup'] = 'Unknown'
            final_cols_to_keep.append('ResearchGroup')
            
        final_cols_to_keep = list(dict.fromkeys(final_cols_to_keep))
        return subjects_passing_qc_df[final_cols_to_keep]

    except FileNotFoundError as e:
        logger.critical(f"CRITICAL Error loading CSV files: {e}")
        return None
    except ValueError as e:
        logger.critical(f"Value error in metadata processing: {e}")
        return None
    except Exception as e:
        logger.critical(f"Unexpected error during metadata loading/QC integration: {e}", exc_info=True)
        return None


# --- Función de inicialización de AAL3 ROI Processing Info ---
# Esta función DEBE ir después de todas las funciones de ayuda que llama
# (como _get_aal3_network_mapping_and_order)
def _initialize_aal3_roi_processing_info():
    global VALID_AAL3_ROI_INFO_DF_166, AAL3_MISSING_INDICES_0BASED, \
           INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166, FINAL_N_ROIS_EXPECTED, \
           N_ROIS_EXPECTED, OUTPUT_CONNECTIVITY_DIR_NAME, CONNECTIVITY_CHANNEL_NAMES, N_CHANNELS, \
           granger_suffix_global, AAL3_ROI_ORDER_MAPPING, deconv_str # Añadir deconv_str aquí para global

    logger.info("--- Initializing AAL3 ROI Processing Information ---")
    
    # Estos deben usar las variables globales que ya fueron asignadas desde argparse en main()
    omst_suffix_for_dir = "OMST_GCE_Signed" if OMST_PYTHON_LOADED and orthogonal_minimum_spanning_tree is not None and USE_PEARSON_OMST_CHANNEL else "PearsonFullSigned"
    
    channel_norm_suffix = "_ChNorm"
    roi_reorder_suffix = "_ROIreorderedYeo17"
    current_roi_order_suffix = ""

    N_ROIS_EXPECTED = EXPECTED_ROIS_AFTER_AAL3_MISSING_REMOVAL # Default value before full AAL3 meta parsing

    if not AAL3_META_PATH.exists():
        logger.error(f"AAL3 metadata file NOT found at: {AAL3_META_PATH}. Cannot perform ROI reduction or reordering. "
                     f"Using initial N_ROIS_EXPECTED = {N_ROIS_EXPECTED}.")
        FINAL_N_ROIS_EXPECTED = N_ROIS_EXPECTED
        AAL3_ROI_ORDER_MAPPING = None
    else:
        try:
            meta_aal3_df = pd.read_csv(AAL3_META_PATH, sep='\t')
            meta_aal3_df['color'] = pd.to_numeric(meta_aal3_df['color'], errors='coerce')
            meta_aal3_df.dropna(subset=['color'], inplace=True)
            meta_aal3_df['color'] = meta_aal3_df['color'].astype(int)
            
            if not all(col in meta_aal3_df.columns for col in ['nom_c', 'color', 'vol_vox']):
                raise ValueError("AAL3 metadata must contain 'nom_c', 'color', 'vol_vox'.")

            AAL3_MISSING_INDICES_0BASED = [idx - 1 for idx in AAL3_MISSING_INDICES_1BASED]
            VALID_AAL3_ROI_INFO_DF_166 = meta_aal3_df[~meta_aal3_df['color'].isin(AAL3_MISSING_INDICES_1BASED)].copy()
            VALID_AAL3_ROI_INFO_DF_166.sort_values(by='color', inplace=True)
            VALID_AAL3_ROI_INFO_DF_166.reset_index(drop=True, inplace=True)

            if len(VALID_AAL3_ROI_INFO_DF_166) != EXPECTED_ROIS_AFTER_AAL3_MISSING_REMOVAL:
                logger.warning(f"Expected {EXPECTED_ROIS_AFTER_AAL3_MISSING_REMOVAL} ROIs in AAL3 meta after filtering known missing, "
                               f"but found {len(VALID_AAL3_ROI_INFO_DF_166)}. Check AAL3_META_PATH content and AAL3_MISSING_INDICES_1BASED.")
            
            small_rois_mask_on_166 = VALID_AAL3_ROI_INFO_DF_166['vol_vox'] < SMALL_ROI_VOXEL_THRESHOLD
            INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166 = VALID_AAL3_ROI_INFO_DF_166[small_rois_mask_on_166].index.tolist()
            
            FINAL_N_ROIS_EXPECTED = EXPECTED_ROIS_AFTER_AAL3_MISSING_REMOVAL - len(INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166)
            N_ROIS_EXPECTED = FINAL_N_ROIS_EXPECTED # Update global N_ROIS_EXPECTED with the final count
            
            logger.info(f"AAL3 ROI processing info initialized (prior to reordering attempt):")
            logger.info(f"  Indices of 4 AAL3 systemically missing ROIs (0-based, from 170): {AAL3_MISSING_INDICES_0BASED}")
            logger.info(f"  Number of ROIs in AAL3 meta after excluding systemically missing: {len(VALID_AAL3_ROI_INFO_DF_166)} (Expected: {EXPECTED_ROIS_AFTER_AAL3_MISSING_REMOVAL})")
            logger.info(f"  Indices of small ROIs to drop (from the {len(VALID_AAL3_ROI_INFO_DF_166)} set, 0-based): {INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166}")
            logger.info(f"  Number of small ROIs to drop: {len(INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166)}")
            logger.info(f"  FINAL_N_ROIS_EXPECTED for connectivity analysis: {FINAL_N_ROIS_EXPECTED} (This should be 131 if matching QC script)")

            AAL3_ROI_ORDER_MAPPING = _get_aal3_network_mapping_and_order()

        except Exception as e:
            logger.error(f"Error initializing AAL3 ROI processing info or during reordering attempt: {e}", exc_info=True)
            # Fallback in case of error during AAL3 meta processing
            FINAL_N_ROIS_EXPECTED = EXPECTED_ROIS_AFTER_AAL3_MISSING_REMOVAL
            N_ROIS_EXPECTED = FINAL_N_ROIS_EXPECTED
            AAL3_ROI_ORDER_MAPPING = None
    
    current_roi_order_suffix = roi_reorder_suffix if AAL3_ROI_ORDER_MAPPING and AAL3_ROI_ORDER_MAPPING.get("new_order_indices") is not None else ""
    if current_roi_order_suffix:
        logger.info(f"ROI reordering WILL BE APPLIED based on '{AAL3_ROI_ORDER_MAPPING.get('order_name', 'custom') if AAL3_ROI_ORDER_MAPPING else 'unknown'}'.")
    else:
        logger.warning("ROI reordering is INACTIVE. Matrices will use default AAL3-derived order. Consider implementing for improved CNN performance and interpretability.")

    # Estas variables 'deconv_str' y 'granger_suffix_global' deben ser globales y ya configuradas por main
    OUTPUT_CONNECTIVITY_DIR_NAME = (
        f"{OUTPUT_CONNECTIVITY_DIR_NAME_BASE}_AAL3_{N_ROIS_EXPECTED if N_ROIS_EXPECTED is not None else 'Unknown'}ROIs_"
        f"{omst_suffix_for_dir}_{granger_suffix_global}{deconv_str}" # Aquí se usan
        f"{channel_norm_suffix}{current_roi_order_suffix}_ParallelTuned"
    )
    if FINAL_N_ROIS_EXPECTED is None or not AAL3_META_PATH.exists() or \
       (roi_reorder_suffix and not current_roi_order_suffix):
        OUTPUT_CONNECTIVITY_DIR_NAME = (
            f"{OUTPUT_CONNECTIVITY_DIR_NAME_BASE}_AAL3_{N_ROIS_EXPECTED if N_ROIS_EXPECTED is not None else 'Unknown'}ROIs_"
            f"{omst_suffix_for_dir}_{granger_suffix_global}{deconv_str}"
            f"{channel_norm_suffix}_ERR_INIT_OR_REORDER_FAIL"
        )
            
    temp_channels = []
    # Logic for Pearson OMST vs. Fallback
    if USE_PEARSON_OMST_CHANNEL and OMST_PYTHON_LOADED and orthogonal_minimum_spanning_tree is not None:
        temp_channels.append(PEARSON_OMST_CHANNEL_NAME_PRIMARY)
    elif USE_PEARSON_FULL_SIGNED_CHANNEL: # If OMST is not used/loaded, but full Pearson is desired
        temp_channels.append(PEARSON_OMST_FALLBACK_NAME)
        if USE_PEARSON_OMST_CHANNEL and (not OMST_PYTHON_LOADED or orthogonal_minimum_spanning_tree is None):
            logger.info(f"OMST function from dyconnmap not loaded. Using '{PEARSON_OMST_FALLBACK_NAME}' for the Pearson-based channel.")
    
    # Add other channels if enabled
    if USE_MI_CHANNEL_FOR_THESIS: temp_channels.append("MI_KNN_Symmetric")
    if USE_DFC_ABS_DIFF_MEAN_CHANNEL: temp_channels.append("dFC_AbsDiffMean")
    if USE_DFC_STDDEV_CHANNEL: temp_channels.append("dFC_StdDev")
    
    if USE_GRANGER_CHANNEL:
        granger_channel_name = f"Granger_F_lag{GRANGER_MAX_LAG}"
        temp_channels.append(granger_channel_name)
    
    # Remove duplicates while preserving order
    CONNECTIVITY_CHANNEL_NAMES[:] = list(dict.fromkeys(temp_channels))
    N_CHANNELS = len(CONNECTIVITY_CHANNEL_NAMES)
    return True

# --- Main Script Execution Flow ---
def _normalize_global_tensor_inter_channel(global_tensor: np.ndarray, train_indices: np.ndarray, method: str = 'zscore_channels_train_params') -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Normaliza el tensor global para que cada tipo de canal tenga una escala comparable,
    calculando parámetros SOLO en el conjunto de entrenamiento.
    global_tensor: (N_subjects, N_channels, N_ROIs, N_ROIs)
    train_indices: Índices de los sujetos que pertenecen al conjunto de entrenamiento.
    method: 'zscore_channels_train_params' u otro.
    
    Retorna el tensor normalizado y los parámetros de normalización (para aplicar a test/val).
    """
    logger.info(f"Applying inter-channel normalization (method: {method}) using training set parameters.")
    normalized_global_tensor = global_tensor.copy()
    norm_params = {'method': method, 'params_per_channel': []}

    if method == 'zscore_channels_train_params':
        for c_idx in range(global_tensor.shape[1]):
            channel_data_train = global_tensor[train_indices, c_idx, :, :]
            
            off_diag_mask_ch = ~np.eye(channel_data_train.shape[1], dtype=bool)
            
            all_off_diag_train_values = []
            for subj_idx_in_train in range(channel_data_train.shape[0]):
                all_off_diag_train_values.extend(channel_data_train[subj_idx_in_train][off_diag_mask_ch])
            
            if not all_off_diag_train_values:
                mean_val = 0.0
                std_val = 1.0 # Use 1.0 to avoid division by zero without modifying if all values are zero
                logger.warning(f"Global tensor: Channel {c_idx} training data off-diagonal elements are empty or constant zero. Setting mean=0, std=1 for this channel.")
            else:
                mean_val = np.mean(all_off_diag_train_values)
                std_val = np.std(all_off_diag_train_values)
            
            norm_params['params_per_channel'].append({'mean': mean_val, 'std': std_val})
            
            if std_val > 1e-9:
                for subj_glob_idx in range(global_tensor.shape[0]):
                    current_matrix = global_tensor[subj_glob_idx, c_idx, :, :]
                    scaled_matrix_ch = current_matrix.copy()
                    scaled_matrix_ch[off_diag_mask_ch] = (current_matrix[off_diag_mask_ch] - mean_val) / std_val
                    normalized_global_tensor[subj_glob_idx, c_idx, :, :] = scaled_matrix_ch
                logger.info(f"Global tensor: Channel {c_idx} off-diagonal z-scored using train_mean={mean_val:.3f}, train_std={std_val:.3f}.")
            else:
                logger.warning(f"Global tensor: Channel {c_idx} has zero/low std in training set off-diagonal elements ({std_val:.3e}). Not scaling this channel.")
        return normalized_global_tensor, norm_params
    else:
        logger.warning(f"Inter-channel normalization method '{method}' not implemented. Returning original tensor.")
        return global_tensor, None
    
def main(args):
    # Asigna los argumentos a las variables globales configurables
    global BASE_PATH_AAL3, QC_OUTPUT_DIR, SUBJECT_METADATA_CSV_PATH_QC, SUBJECT_METADATA_CSV_PATH, \
           QC_REPORT_CSV_PATH, ROI_SIGNALS_DIR_PATH_AAL3, ROI_FILENAME_TEMPLATE, AAL3_META_PATH, \
           AAL3_NIFTI_PATH, TR_SECONDS, LOW_CUT_HZ, HIGH_CUT_HZ, FILTER_ORDER, TAPER_ALPHA, \
           RAW_DATA_EXPECTED_COLUMNS, AAL3_MISSING_INDICES_1BASED, EXPECTED_ROIS_AFTER_AAL3_MISSING_REMOVAL, \
           SMALL_ROI_VOXEL_THRESHOLD, N_NEIGHBORS_MI, DFC_WIN_POINTS_SEC, DFC_STEP_SEC, \
           APPLY_HRF_DECONVOLUTION, HRF_MODEL, USE_GRANGER_CHANNEL, GRANGER_MAX_LAG, \
           OUTPUT_CONNECTIVITY_DIR_NAME_BASE, POSSIBLE_ROI_KEYS, USE_PEARSON_OMST_CHANNEL, \
           USE_PEARSON_FULL_SIGNED_CHANNEL, USE_MI_CHANNEL_FOR_THESIS, USE_DFC_ABS_DIFF_MEAN_CHANNEL, \
           USE_DFC_STDDEV_CHANNEL, MAX_WORKERS, TOTAL_CPU_CORES, granger_suffix_global, deconv_str
    
    BASE_PATH_AAL3 = Path(args.base_path_aal3)
    QC_OUTPUT_DIR = BASE_PATH_AAL3 / args.qc_output_dir_name
    SUBJECT_METADATA_CSV_PATH_QC = BASE_PATH_AAL3 / args.subj_meta_qc_csv
    SUBJECT_METADATA_CSV_PATH = BASE_PATH_AAL3 / args.subj_meta_csv
    QC_REPORT_CSV_PATH = QC_OUTPUT_DIR / args.qc_report_csv
    ROI_SIGNALS_DIR_PATH_AAL3 = BASE_PATH_AAL3 / args.roi_signals_dir
    ROI_FILENAME_TEMPLATE = args.roi_filename_template
    AAL3_META_PATH = BASE_PATH_AAL3 / args.aal3_meta_path
    AAL3_NIFTI_PATH = Path(args.aal3_nifti_path)

    TR_SECONDS = args.tr_seconds
    LOW_CUT_HZ = args.low_cut_hz
    HIGH_CUT_HZ = args.high_cut_hz
    FILTER_ORDER = args.filter_order
    TAPER_ALPHA = args.taper_alpha

    RAW_DATA_EXPECTED_COLUMNS = args.raw_data_expected_columns
    AAL3_MISSING_INDICES_1BASED = [int(i) for i in args.aal3_missing_indices_1based.split(',')]
    EXPECTED_ROIS_AFTER_AAL3_MISSING_REMOVAL = RAW_DATA_EXPECTED_COLUMNS - len(AAL3_MISSING_INDICES_1BASED)
    SMALL_ROI_VOXEL_THRESHOLD = args.small_roi_voxel_threshold

    N_NEIGHBORS_MI = args.n_neighbors_mi

    DFC_WIN_POINTS_SEC = args.dfc_win_points_sec
    DFC_STEP_SEC = args.dfc_step_sec

    APPLY_HRF_DECONVOLUTION = args.apply_hrf_deconvolution
    HRF_MODEL = args.hrf_model

    USE_GRANGER_CHANNEL = args.use_granger_channel
    GRANGER_MAX_LAG = args.granger_max_lag

    OUTPUT_CONNECTIVITY_DIR_NAME_BASE = f"AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced_v6.5.21" # Updated version name

    USE_PEARSON_OMST_CHANNEL = args.use_pearson_omst_channel
    USE_PEARSON_FULL_SIGNED_CHANNEL = args.use_pearson_full_signed_channel
    USE_MI_CHANNEL_FOR_THESIS = args.use_mi_channel_for_thesis
    USE_DFC_ABS_DIFF_MEAN_CHANNEL = args.use_dfc_abs_diff_mean_channel
    USE_DFC_STDDEV_CHANNEL = args.use_dfc_stddev_channel

    # AHORA SÍ: Recalcula estas cadenas con los valores de argparse.
    # Es crucial que estén después de la asignación de todas las variables booleanas y numéricas relevantes.
    granger_suffix_global = f"Granger_lag{GRANGER_MAX_LAG}_" if USE_GRANGER_CHANNEL else "NoEffConn_"
    deconv_str = "_HRFdeconv" if APPLY_HRF_DECONVOLUTION else ""
    
    # Set MAX_WORKERS based on args (or calculated default)
    if args.max_workers is not None:
        if args.max_workers < 1:
            logger.warning(f"max_workers cannot be less than 1. Setting to 1.")
            MAX_WORKERS = 1
        else:
            MAX_WORKERS = args.max_workers
            logger.info(f"MAX_WORKERS explicitly set to: {MAX_WORKERS} via argparse.")
    else: # If not provided, re-calculate based on system cores
        try:
            TOTAL_CPU_CORES = multiprocessing.cpu_count()
            MAX_WORKERS = max(1, TOTAL_CPU_CORES // 2 if TOTAL_CPU_CORES > 2 else 1)
        except NotImplementedError:
            logger.warning("multiprocessing.cpu_count() no está implementado en esta plataforma. Usando MAX_WORKERS = 1.")
            TOTAL_CPU_CORES = 1
            MAX_WORKERS = 1


    # --- ÚNICA LLAMADA A INICIALIZACIÓN DE VARIABLES GLOBALES AQUÍ ---
    # Esto asegura que todas las variables globales configurables por argparse estén disponibles.
    global_init_success = _initialize_aal3_roi_processing_info()
    if not global_init_success:
        logger.critical("CRITICAL: Initial configuration failed. Aborting.")
        return

    try:
        import networkx as nx_runtime
        logger.info(f"RUNTIME NetworkX version being used: {nx_runtime.__version__}")
        if nx_runtime.__version__ != '2.6.3' and OMST_PYTHON_LOADED:
            logger.warning(f"Dyconnmap's OMST typically requires networkx 2.6.3 for full compatibility. "
                           f"Current version is {nx_runtime.__version__}. This might lead to errors with OMST.")
    except ImportError:
        logger.error("RUNTIME: NetworkX is not installed or importable.")

    np.random.seed(42)

    script_start_time = time.time()
    main_process_info = psutil.Process(os.getpid())
    logger.info(f"Main process RAM at start: {main_process_info.memory_info().rss / (1024**2):.2f} MB")
    logger.info(f"--- Starting fMRI Connectivity Pipeline (Version: {OUTPUT_CONNECTIVITY_DIR_NAME_BASE}) ---")
    
    if FINAL_N_ROIS_EXPECTED is None or OUTPUT_CONNECTIVITY_DIR_NAME is None:
        logger.critical("CRITICAL: FINAL_N_ROIS_EXPECTED or OUTPUT_CONNECTIVITY_DIR_NAME was not set during initialization. Aborting.")
        return

    logger.info(f"--- Final Expected ROIs for Connectivity Matrices: {FINAL_N_ROIS_EXPECTED} ---")
    logger.info(f"--- Output Directory Name: {OUTPUT_CONNECTIVITY_DIR_NAME} ---")
    logger.info(f"--- Selected Connectivity Channels for VAE: {CONNECTIVITY_CHANNEL_NAMES} ({N_CHANNELS} channels) ---")
    logger.info(f"--- Per-subject, per-channel normalization with RobustScaler (off-diagonal) will be applied before stacking. ---")
    
    roi_reorder_status = "ACTIVE (Yeo-17 mapping implemented)" if AAL3_ROI_ORDER_MAPPING and AAL3_ROI_ORDER_MAPPING.get("new_order_indices") is not None else "INACTIVE (default AAL3 order)"
    logger.info(f"--- ROI reordering by network is currently: {roi_reorder_status}. ---")
    if not (AAL3_ROI_ORDER_MAPPING and AAL3_ROI_ORDER_MAPPING.get("new_order_indices") is not None) :
        logger.warning("Neuroscientific/Deep Learning Recommendation: ROI reordering (e.g., by Yeo-17 networks or connectome gradients) "
                       "is highly recommended for improving CNN-based model performance and interpretability. "
                       "Current implementation of _get_aal3_network_mapping_and_order is a placeholder or failed.")


    if USE_PEARSON_OMST_CHANNEL and (not OMST_PYTHON_LOADED or orthogonal_minimum_spanning_tree is None):
        logger.warning(f"Note: OMST from dyconnmap could not be loaded. '{PEARSON_OMST_FALLBACK_NAME}' will be used instead of '{PEARSON_OMST_CHANNEL_NAME_PRIMARY}' if enabled and fallback is selected.")

    if not BASE_PATH_AAL3.exists() or not ROI_SIGNALS_DIR_PATH_AAL3.exists():
        logger.critical(f"CRITICAL: Base AAL3 path ({BASE_PATH_AAL3}) or ROI signals directory ({ROI_SIGNALS_DIR_PATH_AAL3}) not found. Aborting.")
        return

    subject_metadata_df = load_metadata(SUBJECT_METADATA_CSV_PATH, QC_REPORT_CSV_PATH)
    if subject_metadata_df is None or subject_metadata_df.empty:
        logger.critical("Metadata loading failed or no subjects passed QC to process. Aborting.")
        return

    output_main_directory = BASE_PATH_AAL3 / OUTPUT_CONNECTIVITY_DIR_NAME
    output_individual_tensors_dir = output_main_directory / "individual_subject_tensors"
    try:
        output_main_directory.mkdir(parents=True, exist_ok=True)
        output_individual_tensors_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Main output directory created/exists: {output_main_directory}")
    except OSError as e:
        logger.critical(f"Could not create output directories: {e}. Aborting."); return

    logger.info(f"Total CPU cores available: {TOTAL_CPU_CORES}. Using MAX_WORKERS = {MAX_WORKERS} for ProcessPoolExecutor.")
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    logger.warning(f"Available system RAM at start of parallel processing: {available_ram_gb:.2f} GB. Monitor usage closely.")

    subject_rows_to_process = list(subject_metadata_df.iterrows())
    num_subjects_to_process = len(subject_rows_to_process)
    if num_subjects_to_process == 0:
        logger.critical("No subjects to process after metadata loading and QC filtering. Aborting.")
        return
    logger.info(f"Starting parallel processing for {num_subjects_to_process} subjects.")
    
    all_subject_results_list = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_subject_id_map = {
            executor.submit(process_single_subject_pipeline, subject_tuple): str(subject_tuple[1]['SubjectID']).strip()
            for subject_tuple in subject_rows_to_process
        }
        for future in tqdm(as_completed(future_to_subject_id_map), total=num_subjects_to_process, desc="Processing Subjects"):
            subject_id_for_log = future_to_subject_id_map[future]
            try:
                subject_result = future.result()
                all_subject_results_list.append(subject_result)
            except Exception as exc:
                logger.critical(f"CRITICAL WORKER EXCEPTION for S {subject_id_for_log}: {exc}", exc_info=True)
                all_subject_results_list.append({
                    "id": subject_id_for_log, "status_overall": "CRITICAL_WORKER_EXCEPTION",
                    "detail_preprocessing": f"Worker process crashed: {str(exc)}",
                    "errors_connectivity_calc": {"worker_exception": str(exc)}
                })

    processing_log_df = pd.DataFrame(all_subject_results_list)
    log_file_path = output_main_directory / f"pipeline_log_{output_main_directory.name}.csv"
    try:
        processing_log_df.to_csv(log_file_path, index=False)
        logger.info(f"Detailed processing log saved to: {log_file_path}")
    except Exception as e_log_save:
        logger.error(f"Failed to save detailed processing log: {e_log_save}")

    successful_subject_entries_list = [
        res for res in all_subject_results_list
        if res.get("status_overall") == "SUCCESS_ALL_PROCESSED_AND_SAVED" and \
           res.get("path_saved_tensor") and Path(res["path_saved_tensor"]).exists()
    ]
    num_successful_subjects_for_tensor = len(successful_subject_entries_list)
    
    logger.info(f"--- Overall Processing Summary ---")
    logger.info(f"Total subjects attempted: {num_subjects_to_process}")
    logger.info(f"Successfully processed and individual tensors saved: {num_successful_subjects_for_tensor}")
    if num_successful_subjects_for_tensor < num_subjects_to_process:
        num_failed = num_subjects_to_process - num_successful_subjects_for_tensor
        logger.warning(f"{num_failed} subjects failed at some stage. Check the detailed log: {log_file_path}")

    if num_successful_subjects_for_tensor > 0:
        logger.info(f"Attempting to assemble global tensor for {num_successful_subjects_for_tensor} successfully processed subjects.")
        global_conn_tensor_list = []
        final_subject_ids_in_tensor = []
        
        current_expected_rois_for_assembly = FINAL_N_ROIS_EXPECTED
        if current_expected_rois_for_assembly is None:
            logger.critical("Cannot assemble global tensor: FINAL_N_ROIS_EXPECTED is None after all processing.")
        else:
            logger.warning("Assembling global tensor using np.stack. This may be memory-intensive for large datasets.")
            try:
                for s_entry in tqdm(successful_subject_entries_list, desc="Assembling Global Tensor"):
                    s_id = s_entry["id"]
                    tensor_path_str = s_entry["path_saved_tensor"]
                    try:
                        with np.load(tensor_path_str) as loaded_npz:
                            s_tensor_data = loaded_npz['tensor_data']
                            if s_tensor_data.shape == (N_CHANNELS, current_expected_rois_for_assembly, current_expected_rois_for_assembly):
                                global_conn_tensor_list.append(s_tensor_data)
                                final_subject_ids_in_tensor.append(s_id)
                            else:
                                logger.error(f"Tensor for S {s_id} from {tensor_path_str} has shape mismatch: {s_tensor_data.shape}. "
                                             f"Expected: ({N_CHANNELS}, {current_expected_rois_for_assembly}, {current_expected_rois_for_assembly}). Skipping.")
                        del s_tensor_data; gc.collect()
                    except Exception as e_load_ind_tensor:
                        logger.error(f"Error loading individual tensor for S {s_id} from {tensor_path_str}: {e_load_ind_tensor}. Skipping.")
                
                if global_conn_tensor_list:
                    global_conn_tensor = np.stack(global_conn_tensor_list, axis=0).astype(np.float32)
                    del global_conn_tensor_list; gc.collect()
                    
                    logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    logger.critical("!! DATA LEAKAGE ALERT & THESIS BEST PRACTICE: Inter-Channel Normalization of Global Tensor        !!")
                    logger.critical("!! La función '_normalize_global_tensor_inter_channel' es un placeholder.                         !!")
                    logger.critical("!! Si se implementa para escalar los canales del 'global_conn_tensor' entre sí (ej. Z-score):    !!")
                    logger.critical("!!   1. DEBE realizarse DENTRO de cada fold de validación cruzada en el script de MODELADO.       !!")
                    logger.critical("!!   2. Los parámetros de normalización (media, std por canal) se calculan ÚNICAMENTE sobre el    !!")
                    logger.critical("!!      CONJUNTO DE ENTRENAMIENTO (train_indices) de ESE FOLD.                                    !!")
                    logger.critical("!!   3. Estos parámetros (ej. guardados en 'norm_params') se APLICAN de forma fija a los          !!")
                    logger.critical("!!      conjuntos de validación y prueba de ESE FOLD.                                             !!")
                    logger.critical("!!   4. NO calcular estos parámetros sobre el tensor global completo ANTES de dividir en folds.   !!")
                    logger.critical("!!   5. Para la tesis: documentar este procedimiento y considerar mostrar resultados con/sin      !!")
                    logger.critical("!!      esta normalización global para demostrar su impacto (y la ausencia de 'peeking').         !!")
                    logger.critical("!!      Considerar guardar el hash del split de CV para trazabilidad.                             !!")
                    logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

                    global_tensor_fname = f"GLOBAL_TENSOR_from_{output_main_directory.name}.npz"
                    global_tensor_path = output_main_directory / global_tensor_fname
                    
                    global_save_metadata = {
                        'global_tensor_data': global_conn_tensor,
                        'subject_ids': np.array(final_subject_ids_in_tensor, dtype=str),
                        'channel_names': np.array(CONNECTIVITY_CHANNEL_NAMES, dtype=str),
                        'rois_count': current_expected_rois_for_assembly,
                        'tr_seconds': TR_SECONDS,
                        'filter_low_hz': LOW_CUT_HZ,
                        'filter_high_hz': HIGH_CUT_HZ,
                        'hrf_deconvolution_applied': APPLY_HRF_DECONVOLUTION,
                        'hrf_model': HRF_MODEL if APPLY_HRF_DECONVOLUTION else "N/A",
                        'channel_normalization_method_subject': "RobustScaler_per_channel_per_subject_off_diagonal",
                        'notes_on_further_normalization': "Inter-channel global normalization (e.g., z-scoring each of the 6 channels based on training set statistics) should be performed within the cross-validation loop of the modeling phase to prevent data leakage."
                    }
                    if AAL3_ROI_ORDER_MAPPING and AAL3_ROI_ORDER_MAPPING.get("new_order_indices") is not None:
                       global_save_metadata['roi_order_name'] = AAL3_ROI_ORDER_MAPPING.get('order_name', 'custom_network_order_placeholder')
                       if 'roi_names_new_order' in AAL3_ROI_ORDER_MAPPING:
                           global_save_metadata['roi_names_in_order'] = np.array(AAL3_ROI_ORDER_MAPPING.get('roi_names_new_order',[]), dtype=str)
                       if 'network_labels_new_order' in AAL3_ROI_ORDER_MAPPING:
                           global_save_metadata['network_labels_in_order'] = np.array(AAL3_ROI_ORDER_MAPPING.get('network_labels_new_order',[]), dtype=str)
                    else:
                       global_save_metadata['roi_order_name'] = 'aal3_original_reduced_order'

                    np.savez_compressed(global_tensor_path, **global_save_metadata)
                    logger.info(f"Global tensor successfully assembled and saved: {global_tensor_path.name}")
                    logger.info(f"Global tensor shape: {global_conn_tensor.shape} (Subjects, Channels, ROIs, ROIs)")
                    del global_conn_tensor; gc.collect()
                else:
                    logger.warning("No valid individual tensors were loaded for global assembly. Global tensor not created.")
            except MemoryError:
                logger.critical("MEMORY ERROR during global tensor assembly (np.stack). Dataset might be too large.")
            except Exception as e_global:
                logger.critical(f"An unexpected error occurred during global tensor assembly: {e_global}", exc_info=True)

    total_time_min = (time.time() - script_start_time) / 60
    logger.info(f"--- fMRI Connectivity Pipeline Finished ---")
    logger.info(f"Total execution time: {total_time_min:.2f} minutes.")
    logger.info(f"Final main process RAM: {main_process_info.memory_info().rss / (1024**2):.2f} MB")
    logger.info(f"All outputs, logs, and tensors should be in: {output_main_directory}")
    logger.info("TESIS DOCTORAL - RECORDATORIOS CLAVE (Checklist Anti-Leakage & Buenas Prácticas):")
    logger.info("  1. PREPROCESAMIENTO fMRI PREVIO (CRÍTICO): Documentar exhaustivamente todos los pasos (movimiento, confounds, scrubbing con umbrales FD/DVARS, GSR si aplica, RETROICOR/ICA-AROMA si se usaron). La calidad de estas matrices depende de ello.")
    logger.info("  2. SELECCIÓN/EXCLUSIÓN DE ROIs Y SUJETOS: Justificar umbral de volumen para ROIs (verificar ROIs clave AD). Justificar criterios de exclusión de sujetos (ej. % outliers, FD medio); asegurar que no introducen sesgo de grupo.")
    logger.info("  3. VARIABILIDAD HRF: Documentar uso de HRF canónica y considerar discusión de sus limitaciones/alternativas (HRF por sujeto, TDM).")
    logger.info("  4. PARCELACIÓN: Documentar AAL3. Considerar discutir alternativas (multi-escala, híbrida, ej. Schaefer+Yeo, o parcelaciones específicas para AD) para trabajos futuros.")
    logger.info("  5. REORDENAMIENTO DE ROIs (MUY RECOMENDADO PARA CNNs): Si se implementa, detallar el mapeo a redes funcionales (Yeo-17, gradientes de conectividad) y el nuevo orden. Guardar esta información.")
    logger.info("  6. NORMALIZACIÓN DE MATRICES:")
    logger.info("       a. Intra-Canal/Sujeto: RobustScaler (off-diagonal) (implementado) - documentar.")
    logger.info("       b. Inter-Canal Global: DEBE realizarse DENTRO de los folds de CV en el script de modelado (parámetros del set de entrenamiento aplicados a validación/test) para evitar data leakage. Documentar método (ej. Z-score) y este procedimiento. Considerar guardar hash del split de CV para trazabilidad.")
    logger.info("  7. SELECCIÓN DE CANALES DE CONECTIVIDAD: Justificar elección inicial. Planificar y documentar análisis de ablación, gating o importancia de características (SHAP, mapas de sensibilidad) para evaluar la contribución de cada canal (especialmente MI y Granger) en el modelo VAE final y decidir sobre su retención.")
    logger.info("       (Sugerencia práctica: empezar con Pearson-Full, añadir OMST, dFC-StdDev; evaluar MI y Granger con cautela).")
    logger.info("       (Considerar añadir métricas como Graphical Lasso/Correlación Parcial, Edge-Time-Series, o HMM/CAPs si el tiempo lo permite y se justifica).")
    logger.info("  8. INTERPRETACIÓN DE MEDIDAS: Ser cauto con la interpretación de Granger. Correlacionar hallazgos de dFC con clínica si es posible. Usar técnicas de decodificación (connectivity gradient decoding, BrainMap) para interpretar factores latentes del VAE.")
    logger.info("  9. VALIDACIÓN DEL MODELO VAE Y CLASIFICADOR: Describir la arquitectura del VAE (considerar GNN/VAE híbrido), función de pérdida (ej. β-VAE, aprendizaje contrastivo), y la estrategia de validación cruzada ANIDADA para el clasificador final, asegurando que no haya data leakage en ningún paso (incluida la optimización de hiperparámetros con Optuna/W&B en el bucle interno).")
    logger.info(" 10. CONFIABILIDAD Y ESTADÍSTICA: Si hay datos test-retest, calcular ICC de las matrices/canales (descartar canales con ICC < ~0.4). Para comparaciones de grupo o mapas de importancia, usar métodos estadísticos robustos (NBS, TFCE, spin-tests).")
    logger.info(" 11. HARMONIZACIÓN MULTI-SITIO (Sugerencia Tesis 7): Si los datos provienen de múltiples sitios/escáneres, aplicar harmonización (ej. ComBat, neuroHarmonize) sobre las características derivadas DENTRO de los folds de CV (parámetros del set de entrenamiento).")
    logger.info(" 12. DIMENSIONALIDAD: Reconocer que se aumenta la dimensionalidad de las features, no el N. Mitigar con regularización en el VAE (β alto, dropout de canal, gating, etc.) y early stopping basado en métricas de validación (AUROC, no solo reconstrucción). Considerar data augmentation (random masking de ROIs/canales).")
    logger.info(" 13. REPRODUCIBILIDAD (Sugerencia Tesis 10): Exportar resultados en formato BIDS-Derivatives si es posible. Publicar código (ej. Zenodo con DOI) y describir detalladamente el pipeline (versión v6.5.21).") # Updated version here
    logger.info(" 14. CITAS: Citar dyconnmap si se usó OMST, y todas las herramientas/paquetes relevantes.")


if __name__ == "__main__":
    # Necesario para compatibilidad con Windows en multiprocessing al usar ProcessPoolExecutor
    multiprocessing.freeze_support()
    
    # Configura el logger: esto solo se ejecutará si el script se corre directamente.
    # El nivel por defecto es INFO. Si se necesita depuración, cambiar a logging.DEBUG
    # NOTA: Si ya tienes un sistema de logging complejo en un proyecto, podrías comentar esta línea
    # y manejar la configuración del logger desde tu script principal que importa este módulo.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    # La instancia 'logger' ya fue obtenida a nivel de módulo al inicio del script.

    # --- Definición de los argumentos de línea de comandos ---
    parser = argparse.ArgumentParser(description="fMRI Connectivity Feature Extraction Pipeline for Doctoral Thesis.")
    
    # Paths and general config
    parser.add_argument("--base_path_aal3", type=str, default='/home/diego/Escritorio/AAL3',
                        help="Base path for AAL3 related data (ROI signals, metadata, NIfTI atlas).")
    parser.add_argument("--qc_output_dir_name", type=str, default='qc_outputs_doctoral_v3.2_aal3_shrinkage_flexible_thresh_fix',
                        help="Name of the QC output directory relative to base_path_aal3.")
    parser.add_argument("--subj_meta_qc_csv", type=str, default='SubjctsDataAndTests_Schaefer2018_400Parcels_17Networks.csv',
                        help="Subject metadata CSV path used during QC, relative to base_path_aal3.")
    parser.add_argument("--subj_meta_csv", type=str, default='SubjectsData_Schaefer2018_400ROIs.csv',
                        help="Main subject metadata CSV path, relative to base_path_aal3.")
    parser.add_argument("--qc_report_csv", type=str, default='report_qc_final_with_discard_flags_v3.2.csv',
                        help="QC report CSV path, relative to qc_output_dir.")
    parser.add_argument("--roi_signals_dir", type=str, default='ROISignals_AAL3',
                        help="Directory containing ROI signals .mat files, relative to base_path_aal3.")
    parser.add_argument("--roi_filename_template", type=str, default='ROISignals_{subject_id}.mat',
                        help="Template for ROI signal filenames.")
    parser.add_argument("--aal3_meta_path", type=str, default='ROI_MNI_V7_vol.txt',
                        help="AAL3 metadata .txt file path, relative to base_path_aal3.")
    parser.add_argument("--aal3_nifti_path", type=str, default='/home/diego/Escritorio/AAL3/AAL3v1.nii.gz',
                        help="Full path to the AAL3 NIfTI atlas file. IMPORTANT: Update this path.")

    # Preprocessing parameters
    parser.add_argument("--tr_seconds", type=float, default=3.0, help="Repetition Time (TR) in seconds.")
    parser.add_argument("--low_cut_hz", type=float, default=0.01, help="Low-cut frequency for bandpass filter (Hz).")
    parser.add_argument("--high_cut_hz", type=float, default=0.08, help="High-cut frequency for bandpass filter (Hz).")
    parser.add_argument("--filter_order", type=int, default=2, help="Order of the Butterworth bandpass filter.")
    parser.add_argument("--taper_alpha", type=float, default=0.1, help="Alpha parameter for Tukey window tapering (0.0 to 1.0).")
    parser.add_argument("--apply_hrf_deconvolution", type=lambda x: x.lower() == 'true', default=False,
                        help="Whether to apply HRF deconvolution. Use 'true' or 'false'.")
    parser.add_argument("--hrf_model", type=str, default='glover', choices=['glover', 'spm'], help="HRF model type if deconvolution is applied.")

    # ROI specific parameters
    parser.add_argument("--raw_data_expected_columns", type=int, default=170, help="Expected number of columns (ROIs) in raw .mat files.")
    parser.add_argument("--aal3_missing_indices_1based", type=str, default="35,36,81,82",
                        help="Comma-separated 1-based indices of AAL3 ROIs to be excluded (e.g., '35,36,81,82').")
    parser.add_argument("--small_roi_voxel_threshold", type=int, default=100, help="Threshold for dropping small ROIs (in voxels).")

    # Connectivity parameters
    parser.add_argument("--n_neighbors_mi", type=int, default=5, help="Number of neighbors for Mutual Information (MI) estimation.")
    parser.add_argument("--dfc_win_points_sec", type=float, default=90.0, help="Dynamic Functional Connectivity (dFC) window length in seconds.")
    parser.add_argument("--dfc_step_sec", type=float, default=15.0, help="Dynamic Functional Connectivity (dFC) step size in seconds.")
    parser.add_argument("--use_granger_channel", type=lambda x: x.lower() == 'true', default=True,
                        help="Whether to calculate Granger Causality channel. Use 'true' or 'false'.")
    parser.add_argument("--granger_max_lag", type=int, default=1, help="Maximum lag for Granger Causality calculation.")
    
    # Channel selection (Boolean flags)
    parser.add_argument("--use_pearson_omst_channel", type=lambda x: x.lower() == 'true', default=True,
                        help="Include Pearson OMST channel. Use 'true' or 'false'.")
    parser.add_argument("--use_pearson_full_signed_channel", type=lambda x: x.lower() == 'true', default=True,
                        help="Include Full Pearson Fisher Z Signed channel. Use 'true' or 'false'.")
    parser.add_argument("--use_mi_channel_for_thesis", type=lambda x: x.lower() == 'true', default=True,
                        help="Include Mutual Information KNN channel. Use 'true' or 'false'.")
    parser.add_argument("--use_dfc_abs_diff_mean_channel", type=lambda x: x.lower() == 'true', default=True,
                        help="Include dFC Absolute Difference Mean channel. Use 'true' or 'false'.")
    parser.add_argument("--use_dfc_stddev_channel", type=lambda x: x.lower() == 'true', default=True,
                        help="Include dFC Standard Deviation channel. Use 'true' or 'false'.")

    # Parallel processing
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Maximum number of worker processes for parallel processing. "
                             "Defaults to total_cpu_cores // 2 or 1 if total_cpu_cores <= 2.")

    args = parser.parse_args()

    # Llamada a la función principal con los argumentos parseados
    main(args)