#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sat_night.py

Script robusto para un pipeline de clasificación de dos etapas para la tesis doctoral.
Etapa 1: Entrena un Autoencoder Variacional (VAE) Convolucional sobre matrices de 
conectividad funcional fMRI para aprender una representación de baja dimensionalidad.
Etapa 2: Utiliza el espacio latente del VAE para entrenar y evaluar múltiples 
clasificadores para distinguir entre Controles Sanos (CN) y pacientes con 
Enfermedad de Alzheimer (AD), siguiendo las mejores prácticas para evitar data leakage.

Versión: 2.0.0 - Tesis Doctoral Edition
Cambios Principales:
- **Optimización de Hiperparámetros (HP) Robusta**:
  - Se reemplaza GridSearchCV por Optimización Bayesiana con Optuna para clasificadores
    complejos (RF, GB, XGBoost, LightGBM), lo cual es más eficiente y efectivo
    en espacios de búsqueda grandes.
  - Se mantienen grids reducidos para clasificadores más simples (SVM, LogReg).
- **Nuevos Clasificadores State-of-the-Art**:
  - Se añaden `XGBoost` y `LightGBM` como opciones, que a menudo superan a los
    clasificadores de scikit-learn en rendimiento y velocidad.
- **Calibración de Probabilidades para SVM**:
  - Se implementa `CalibratedClassifierCV` para el SVM, obteniendo probabilidades
    más fiables para el cálculo del AUC, entrenando el calibrador de forma segura
    dentro del fold para evitar data leakage.
- **Manejo Dinámico de Datos**:
  - El script ahora lee los nombres de los canales directamente de los metadatos
    del archivo del tensor `.npz`, eliminando la necesidad de listas hardcodeadas.
  - Se ha eliminado la opción de normalización con rangos fijos para reforzar
    la práctica correcta de calcular los parámetros de normalización en cada fold.
- **Refactorización y Claridad**:
  - Se ha mejorado la estructura del código para una mayor legibilidad y mantenimiento.
  - Se han añadido comentarios extensos para justificar las decisiones metodológicas.
- **Dependencias Adicionales**:
  - `optuna`, `xgboost`, `lightgbm`. (Instalar con: pip install optuna xgboost lightgbm)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time
import argparse
import gc
import subprocess
import joblib
from typing import Optional, List, Dict, Tuple, Any, Union
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    train_test_split as sk_train_test_split,
    RepeatedStratifiedKFold
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler as SklearnScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    f1_score,
    average_precision_score,
    balanced_accuracy_score
)
import matplotlib.pyplot as plt

# --- Dependencias opcionales pero recomendadas ---
# --- Dep. opcionales
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # ① << NUEVO
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# --- Configuración del Logger ---
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
#logger = logging.getLogger(__name__)

# --- Configuración del Logger ---
def init_logger(verbosity: str = "INFO") -> logging.Logger:
    numeric_level = getattr(logging, verbosity.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


# --- Modelo VAE (sin cambios, ya era robusto) ---
class ConvolutionalVAE(nn.Module):
    def __init__(self,
                 input_channels: int = 6,
                 latent_dim: int = 128,
                 image_size: int = 131,
                 final_activation: str = 'tanh',
                 intermediate_fc_dim_config: Union[int, str] = "0",
                 dropout_rate: float = 0.2,
                 use_layernorm_fc: bool = False,
                 num_conv_layers_encoder: int = 4,
                 decoder_type: str = 'convtranspose'
                 ):
        super(ConvolutionalVAE, self).__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.final_activation_name = final_activation
        self.dropout_rate = dropout_rate
        self.use_layernorm_fc = use_layernorm_fc
        self.num_conv_layers_encoder = num_conv_layers_encoder
        self.decoder_type = decoder_type

        if self.num_conv_layers_encoder not in [3, 4]:
            raise ValueError("num_conv_layers_encoder must be 3 or 4.")
        if self.decoder_type not in ['upsample_conv', 'convtranspose']:
            raise ValueError("decoder_type must be 'upsample_conv' or 'convtranspose'.")

        # --- Encoder ---
        encoder_conv_layers_list = []
        current_ch_enc = input_channels
        
        base_conv_channels = [max(16, input_channels*2), max(32, input_channels*4), max(64, input_channels*8), max(128, input_channels*16)]
        conv_channels_enc_all = [min(c, 256) for c in base_conv_channels]
        
        kernels_all = [7, 5, 5, 3]
        paddings_all = [1, 1, 1, 1]
        strides_all = [2, 2, 2, 2]

        self.encoder_kernels = kernels_all[:self.num_conv_layers_encoder]
        self.encoder_paddings = paddings_all[:self.num_conv_layers_encoder]
        self.encoder_strides = strides_all[:self.num_conv_layers_encoder]
        self.conv_channels_encoder = conv_channels_enc_all[:self.num_conv_layers_encoder]

        self.encoder_spatial_dims = [self.image_size]
        current_dim_calc = self.image_size

        for i in range(self.num_conv_layers_encoder):
            encoder_conv_layers_list.extend([
                nn.Conv2d(current_ch_enc, self.conv_channels_encoder[i],
                          kernel_size=self.encoder_kernels[i],
                          stride=self.encoder_strides[i],
                          padding=self.encoder_paddings[i]),
                nn.ReLU(),
                nn.BatchNorm2d(self.conv_channels_encoder[i]),
                nn.Dropout2d(p=self.dropout_rate)
            ])
            current_ch_enc = self.conv_channels_encoder[i]
            current_dim_calc = ((current_dim_calc + 2 * self.encoder_paddings[i] - self.encoder_kernels[i]) // self.encoder_strides[i]) + 1
            self.encoder_spatial_dims.append(current_dim_calc)
        
        self.encoder_conv = nn.Sequential(*encoder_conv_layers_list)
        
        self.final_conv_output_channels = self.conv_channels_encoder[-1]
        self.final_spatial_dim_encoder = self.encoder_spatial_dims[-1]
        self.flattened_size_after_conv = self.final_conv_output_channels * self.final_spatial_dim_encoder * self.final_spatial_dim_encoder
        
        if isinstance(intermediate_fc_dim_config, str):
            if intermediate_fc_dim_config.lower() == "half":
                self.intermediate_fc_dim = self.flattened_size_after_conv // 2
            elif intermediate_fc_dim_config.lower() == "quarter":
                self.intermediate_fc_dim = self.flattened_size_after_conv // 4
            elif intermediate_fc_dim_config.lower() == "0":
                self.intermediate_fc_dim = 0
            else:
                try:
                    self.intermediate_fc_dim = int(intermediate_fc_dim_config)
                except ValueError: self.intermediate_fc_dim = 0
        else:
            self.intermediate_fc_dim = int(intermediate_fc_dim_config)

        encoder_fc_intermediate_layers = []
        if self.intermediate_fc_dim > 0:
            encoder_fc_intermediate_layers.append(nn.Linear(self.flattened_size_after_conv, self.intermediate_fc_dim))
            if self.use_layernorm_fc:
                encoder_fc_intermediate_layers.append(nn.LayerNorm(self.intermediate_fc_dim))
            encoder_fc_intermediate_layers.extend([nn.ReLU(), nn.BatchNorm1d(self.intermediate_fc_dim), nn.Dropout(p=self.dropout_rate)])
            self.encoder_fc_intermediate = nn.Sequential(*encoder_fc_intermediate_layers)
            fc_mu_logvar_input_dim = self.intermediate_fc_dim
        else:
            self.encoder_fc_intermediate = nn.Identity()
            fc_mu_logvar_input_dim = self.flattened_size_after_conv

        self.fc_mu = nn.Linear(fc_mu_logvar_input_dim, latent_dim)
        self.fc_logvar = nn.Linear(fc_mu_logvar_input_dim, latent_dim)

        # --- Decoder ---
        decoder_fc_intermediate_layers = []
        decoder_fc_to_conv_input_dim = latent_dim
        if self.intermediate_fc_dim > 0:
            decoder_fc_intermediate_layers.append(nn.Linear(latent_dim, self.intermediate_fc_dim))
            if self.use_layernorm_fc:
                decoder_fc_intermediate_layers.append(nn.LayerNorm(self.intermediate_fc_dim))
            decoder_fc_intermediate_layers.extend([nn.ReLU(), nn.BatchNorm1d(self.intermediate_fc_dim), nn.Dropout(p=self.dropout_rate)])
            self.decoder_fc_intermediate = nn.Sequential(*decoder_fc_intermediate_layers)
            decoder_fc_to_conv_input_dim = self.intermediate_fc_dim
        else:
            self.decoder_fc_intermediate = nn.Identity()

        self.decoder_fc_to_conv = nn.Linear(decoder_fc_to_conv_input_dim, self.flattened_size_after_conv)

        decoder_conv_layers_list = []
        if self.decoder_type == 'convtranspose':
            current_ch_dec = self.final_conv_output_channels
            target_conv_t_channels = self.conv_channels_encoder[-2::-1] + [self.input_channels]
            decoder_kernels = self.encoder_kernels[::-1]
            decoder_paddings = self.encoder_paddings[::-1]
            decoder_strides = self.encoder_strides[::-1]
            
            output_paddings_calc = []
            current_spatial_dim_for_op_calc = self.final_spatial_dim_encoder
            for i in range(self.num_conv_layers_encoder):
                k, s, p = decoder_kernels[i], decoder_strides[i], decoder_paddings[i]
                target_dim_layer = self.encoder_spatial_dims[self.num_conv_layers_encoder - 1 - i]
                op = target_dim_layer - ((current_spatial_dim_for_op_calc - 1) * s - 2 * p + k)
                op = max(0, min(s - 1, op))
                output_paddings_calc.append(op)
                current_spatial_dim_for_op_calc = (current_spatial_dim_for_op_calc - 1) * s - 2 * p + k + op

            for i in range(len(target_conv_t_channels)):
                decoder_conv_layers_list.extend([
                    nn.ConvTranspose2d(current_ch_dec, target_conv_t_channels[i],
                                       kernel_size=decoder_kernels[i], stride=decoder_strides[i],
                                       padding=decoder_paddings[i], output_padding=output_paddings_calc[i]),
                    nn.ReLU() if i < len(target_conv_t_channels) - 1 else nn.Identity(),
                    nn.BatchNorm2d(target_conv_t_channels[i]) if i < len(target_conv_t_channels) - 1 else nn.Identity()
                ])
                if i < len(target_conv_t_channels) - 1:
                    decoder_conv_layers_list.append(nn.Dropout2d(p=self.dropout_rate))
                current_ch_dec = target_conv_t_channels[i]
        else:
            logger.error("Decoder type 'upsample_conv' no implementado en esta versión.")
            raise NotImplementedError

        if self.final_activation_name == 'sigmoid':
            decoder_conv_layers_list.append(nn.Sigmoid())
        elif self.final_activation_name == 'tanh':
            decoder_conv_layers_list.append(nn.Tanh())
        
        self.decoder_conv = nn.Sequential(*decoder_conv_layers_list)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        h_intermediate = self.encoder_fc_intermediate(h)
        return self.fc_mu(h_intermediate), self.fc_logvar(h_intermediate)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h_latent_intermediate = self.decoder_fc_intermediate(z)
        h_fc_out = self.decoder_fc_to_conv(h_latent_intermediate)
        h_reshaped = h_fc_out.view(h_fc_out.size(0), self.final_conv_output_channels,
                                   self.final_spatial_dim_encoder, self.final_spatial_dim_encoder)
        return self.decoder_conv(h_reshaped)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        if recon_x.shape != x.shape:
             recon_x = nn.functional.interpolate(recon_x, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return recon_x, mu, logvar, z

# --- Funciones de Pérdida y Schedules (sin cambios) ---
def vae_loss_function(recon_x, x, mu, logvar, beta):
    recon_loss_mse = nn.functional.mse_loss(recon_x, x, reduction='sum') / x.shape[0]
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld /= x.shape[0] # Average KLD per sample
    return recon_loss_mse + beta * kld

def get_cyclical_beta_schedule(current_epoch, total_epochs, beta_max, n_cycles, ratio_increase):
    if n_cycles <= 0: return beta_max
    epoch_per_cycle = total_epochs / n_cycles
    epoch_in_current_cycle = current_epoch % epoch_per_cycle
    increase_phase_duration = epoch_per_cycle * ratio_increase
    if epoch_in_current_cycle < increase_phase_duration:
        return beta_max * (epoch_in_current_cycle / increase_phase_duration)
    else:
        return beta_max

# --- Funciones de Datos (adaptadas) ---
def load_data(tensor_path: Path, metadata_path: Path) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame], Optional[List[str]]]:
    logger.info(f"Cargando tensor global desde: {tensor_path}")
    if not tensor_path.exists():
        logger.error(f"Archivo de tensor global NO encontrado: {tensor_path}")
        return None, None, None
    try:
        data_npz = np.load(tensor_path, allow_pickle=True)
        global_tensor = data_npz['global_tensor_data']
        subject_ids_tensor = data_npz['subject_ids'].astype(str)
        channel_names = [str(name) for name in data_npz.get('channel_names', [])]
        if not channel_names:
            logger.warning("Nombres de canales no encontrados en metadatos del tensor, se usarán nombres genéricos.")
        logger.info(f"Tensor global cargado. Forma: {global_tensor.shape}. Canales: {channel_names}")
    except Exception as e:
        logger.error(f"Error cargando tensor global: {e}", exc_info=True)
        return None, None, None

    logger.info(f"Cargando metadatos desde: {metadata_path}")
    if not metadata_path.exists():
        logger.error(f"Archivo de metadatos NO encontrado: {metadata_path}")
        return None, None, None
    try:
        metadata_df = pd.read_csv(metadata_path)
        metadata_df['SubjectID'] = metadata_df['SubjectID'].astype(str).str.strip()
        logger.info(f"Metadatos cargados. Forma: {metadata_df.shape}")
    except Exception as e:
        logger.error(f"Error cargando metadatos: {e}", exc_info=True)
        return None, None, None

    tensor_df = pd.DataFrame({'SubjectID': subject_ids_tensor})
    tensor_df['tensor_idx'] = np.arange(len(subject_ids_tensor))
    merged_df = pd.merge(tensor_df, metadata_df, on='SubjectID', how='left')
    
    if len(merged_df) < len(subject_ids_tensor):
        logger.warning(f"Algunos SubjectIDs del tensor no se encontraron en metadatos.")
    
    return global_tensor, merged_df, channel_names

def normalize_inter_channel_fold(
    data_tensor: np.ndarray, 
    train_indices_in_fold: np.ndarray, 
    mode: str = 'zscore_offdiag',
    selected_channel_original_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    num_subjects_total, num_selected_channels, num_rois, _ = data_tensor.shape
    logger.info(f"Aplicando normalización inter-canal (modo: {mode}) sobre {num_selected_channels} canales seleccionados.")
    logger.info(f"Parámetros de normalización se calcularán usando {len(train_indices_in_fold)} sujetos de entrenamiento.")
    
    normalized_tensor_fold = data_tensor.copy()
    norm_params_per_channel_list = []
    off_diag_mask = ~np.eye(num_rois, dtype=bool)

    for c_idx_selected in range(num_selected_channels):
        current_channel_original_name = selected_channel_original_names[c_idx_selected] if selected_channel_original_names and c_idx_selected < len(selected_channel_original_names) else f"Channel_{c_idx_selected}"
        params = {'mode': mode, 'original_name': current_channel_original_name}
        
        channel_data_train_for_norm_params = data_tensor[train_indices_in_fold, c_idx_selected, :, :]
        all_off_diag_train_values = channel_data_train_for_norm_params[:, off_diag_mask].flatten()

        if all_off_diag_train_values.size == 0:
            logger.warning(f"Canal '{current_channel_original_name}': No hay elementos fuera de la diagonal en el training set. No se escala.")
            params.update({'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 1.0, 'no_scale': True})
        elif mode == 'zscore_offdiag':
            mean_val = np.mean(all_off_diag_train_values)
            std_val = np.std(all_off_diag_train_values)
            params.update({'mean': mean_val, 'std': std_val if std_val > 1e-9 else 1.0})
            if std_val <= 1e-9: logger.warning(f"Canal '{current_channel_original_name}': STD muy bajo ({std_val:.2e}). Usando STD=1.")
        else:
            raise ValueError(f"Modo de normalización desconocido: {mode}")
        
        norm_params_per_channel_list.append(params)

        if not params.get('no_scale', False):
            current_channel_data_all_subjects = data_tensor[:, c_idx_selected, :, :]
            scaled_channel_data = current_channel_data_all_subjects.copy()
            if off_diag_mask.any():
                if mode == 'zscore_offdiag':
                    if params['std'] > 1e-9:
                        scaled_channel_data[:, off_diag_mask] = (current_channel_data_all_subjects[:, off_diag_mask] - params['mean']) / params['std']
            normalized_tensor_fold[:, c_idx_selected, :, :] = scaled_channel_data
            log_msg_params = f"Canal '{current_channel_original_name}': Off-diag {mode} (train_params: mean={params['mean']:.3f}, std={params['std']:.3f})"
            logger.info(log_msg_params)
    return normalized_tensor_fold, norm_params_per_channel_list

def apply_normalization_params(data_tensor_subset: np.ndarray, 
                               norm_params_per_channel_list: List[Dict[str, float]]
                               ) -> np.ndarray:
    num_subjects, num_selected_channels, num_rois, _ = data_tensor_subset.shape
    normalized_tensor_subset = data_tensor_subset.copy()
    off_diag_mask = ~np.eye(num_rois, dtype=bool)

    if len(norm_params_per_channel_list) != num_selected_channels:
        raise ValueError(f"Mismatch en número de canales para normalización: datos tienen {num_selected_channels}, params para {len(norm_params_per_channel_list)}")

    for c_idx_selected in range(num_selected_channels):
        params = norm_params_per_channel_list[c_idx_selected]
        mode = params.get('mode', 'zscore_offdiag')
        if params.get('no_scale', False): continue
        current_channel_data = data_tensor_subset[:, c_idx_selected, :, :]
        scaled_channel_data_subset = current_channel_data.copy()
        if off_diag_mask.any() and mode == 'zscore_offdiag' and params['std'] > 1e-9:
            scaled_channel_data_subset[:, off_diag_mask] = (current_channel_data[:, off_diag_mask] - params['mean']) / params['std']
        normalized_tensor_subset[:, c_idx_selected, :, :] = scaled_channel_data_subset
    return normalized_tensor_subset

def log_group_distributions(df: pd.DataFrame, group_cols: List[str], dataset_name: str, fold_idx_str: str):
    if df.empty:
        logger.info(f"  {fold_idx_str} {dataset_name}: DataFrame vacío.")
        return
    log_msg = f"  {fold_idx_str} {dataset_name} (N={len(df)}):\n"
    for col in group_cols:
        if col in df.columns:
            counts = df[col].value_counts().sort_index()
            log_msg += f"    {col}:\n"
            for val, count in counts.items():
                log_msg += f"      {val}: {count} ({count/len(df)*100:.1f}%)\n"
        else:
            log_msg += f"    {col}: No encontrado en el DataFrame.\n"
    logger.info(log_msg.strip())

# --- Módulo de Optimización con Optuna ---
def get_optuna_objective(X_train, y_train, X_val, y_val, classifier_type, class_weight_mode, seed):
    def objective(trial):
        if classifier_type == 'rf':
            params = {'n_estimators': trial.suggest_int('n_estimators', 50, 250),
                      'max_depth': trial.suggest_int('max_depth', 5, 30),
                      'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                      'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)}
            model = RandomForestClassifier(random_state=seed, class_weight=class_weight_mode, **params)
        elif classifier_type == 'gb':
            params = {'n_estimators': trial.suggest_int('n_estimators', 50, 250),
                      'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                      'max_depth': trial.suggest_int('max_depth', 3, 8)}
            model = GradientBoostingClassifier(random_state=seed, **params)
        elif classifier_type == 'xgb' and XGBOOST_AVAILABLE:
            scale_pos_weight = (sum(y_train == 0) / sum(y_train == 1)) if class_weight_mode else 1
            params = {'n_estimators': trial.suggest_int('n_estimators', 50, 250),
                      'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                      'max_depth': trial.suggest_int('max_depth', 3, 8),
                      'scale_pos_weight': scale_pos_weight}
            model = xgb.XGBClassifier(random_state=seed, use_label_encoder=False, eval_metric='logloss', **params)
        elif classifier_type == 'lgb' and LIGHTGBM_AVAILABLE:
            params = {'n_estimators': trial.suggest_int('n_estimators', 50, 250),
                      'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                      'num_leaves': trial.suggest_int('num_leaves', 10, 50)}
            model = lgb.LGBMClassifier(random_state=seed, class_weight=class_weight_mode, **params)
        else:
            return float('inf')

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return 1.0 - balanced_accuracy_score(y_val, preds)
    return objective

# --- Función Principal de Entrenamiento y Evaluación (Versión Doctoral Completa) ---
def train_and_evaluate_pipeline(global_tensor_all_channels: np.ndarray,
                                metadata_df_full: pd.DataFrame,
                                all_channel_names: List[str],
                                args: argparse.Namespace):
    """
    Función principal que orquesta el pipeline de entrenamiento y evaluación.
    Aplica una validación cruzada anidada para entrenar un VAE y luego
    múltiples clasificadores, asegurando que no haya fuga de datos.
    """
    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Selección de Canales de Conectividad ---
    selected_channel_indices: List[int] = []
    if args.channels_to_use:
        for ch_specifier in args.channels_to_use:
            try:
                ch_idx = int(ch_specifier)
                if 0 <= ch_idx < len(all_channel_names):
                    selected_channel_indices.append(ch_idx)
                else:
                    logger.warning(f"Índice de canal '{ch_idx}' fuera de rango. Ignorando.")
            except ValueError:
                if ch_specifier in all_channel_names:
                    selected_channel_indices.append(all_channel_names.index(ch_specifier))
                else:
                    logger.warning(f"Nombre de canal '{ch_specifier}' desconocido en el tensor. Ignorando.")
    
    if not selected_channel_indices:
        logger.info("No se especificaron canales, usando todos los disponibles en el tensor.")
        selected_channel_indices = list(range(global_tensor_all_channels.shape[1]))

    selected_channel_indices = sorted(list(set(selected_channel_indices)))
    selected_channel_names_in_tensor = [all_channel_names[i] for i in selected_channel_indices]
    current_global_tensor = global_tensor_all_channels[:, selected_channel_indices, :, :]
    num_input_channels_for_vae = current_global_tensor.shape[1]
    logger.info(f"Usando {num_input_channels_for_vae} canales: {selected_channel_names_in_tensor}")

    # --- 2. Preparación de Datos para Clasificación ---
    cn_ad_df = metadata_df_full[metadata_df_full['ResearchGroup'].isin(['CN', 'AD'])].copy()
    cn_ad_df = cn_ad_df[cn_ad_df['tensor_idx'] < current_global_tensor.shape[0]].copy()
    cn_ad_df['label'] = cn_ad_df['ResearchGroup'].map({'CN': 0, 'AD': 1})
    
    strat_cols = ['ResearchGroup']
    if args.classifier_stratify_cols:
        for col in args.classifier_stratify_cols:
            if col in cn_ad_df.columns and col not in strat_cols:
                cn_ad_df[col] = cn_ad_df[col].fillna(f"{col}_Unknown").astype(str)
                strat_cols.append(col)
    
    cn_ad_df['stratify_key_clf'] = cn_ad_df[strat_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    
    X_subject_indices = np.arange(len(cn_ad_df))
    y_stratify_clf = cn_ad_df['stratify_key_clf'].values
    logger.info(f"Total de sujetos CN/AD para clasificación: {len(cn_ad_df)}. (CN: {sum(cn_ad_df['label'] == 0)}, AD: {sum(cn_ad_df['label'] == 1)})")

    # --- 3. Bucle Principal de Validación Cruzada Externa ---
    if args.repeated_outer_folds_n_repeats > 1:
        outer_cv_clf = RepeatedStratifiedKFold(
            n_splits=args.outer_folds,
            n_repeats=args.repeated_outer_folds_n_repeats,
            random_state=args.seed)
        total_outer = args.outer_folds * args.repeated_outer_folds_n_repeats
        logger.info(f"Usando RepeatedStratifiedKFold ({total_outer} iteraciones).")
    else:
        outer_cv_clf = StratifiedKFold(
            n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
        total_outer = args.outer_folds
        logger.info(f"Usando StratifiedKFold ({total_outer} iteraciones).")

    all_folds_metrics = []
    
    for fold_idx, (train_dev_clf_idx, test_clf_idx) in enumerate(outer_cv_clf.split(X_subject_indices, y_stratify_clf)):
        fold_start_time = time.time()
        fold_idx_str = f"Fold {fold_idx + 1}/{total_outer}"
        logger.info(f"--- Iniciando {fold_idx_str} ---")
        fold_output_dir = output_base_dir / f"fold_{fold_idx + 1}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        # --- 4. Entrenamiento del VAE (dentro de cada fold) ---
        
        # 4.1. Separar datos: el VAE se entrena con TODO menos el test set del clasificador
        global_indices_clf_test = cn_ad_df.iloc[test_clf_idx]['tensor_idx'].values
        all_tensor_indices = metadata_df_full['tensor_idx'].values
        global_indices_vae_pool = np.setdiff1d(all_tensor_indices, global_indices_clf_test, assume_unique=True)
        
        vae_train_pool_df = metadata_df_full[metadata_df_full['tensor_idx'].isin(global_indices_vae_pool)]
        
        # 4.2. Crear split de train/validación para el VAE
        vae_pool_indices = np.arange(len(global_indices_vae_pool))
        vae_train_indices, vae_val_indices = sk_train_test_split(
            vae_pool_indices, test_size=args.vae_val_split_ratio, random_state=args.seed + fold_idx)
        
        # 4.3. Normalizar datos del VAE (parámetros calculados SOLO en el train set del VAE)
        vae_train_pool_tensor = current_global_tensor[global_indices_vae_pool]
        vae_train_tensor_norm, norm_params = normalize_inter_channel_fold(
            vae_train_pool_tensor[vae_train_indices], np.arange(len(vae_train_indices)),
            mode=args.norm_mode, selected_channel_original_names=selected_channel_names_in_tensor
        )
        vae_val_tensor_norm = apply_normalization_params(vae_train_pool_tensor[vae_val_indices], norm_params)

        # 4.4. Crear DataLoaders para el VAE
        vae_train_loader = DataLoader(TensorDataset(torch.from_numpy(vae_train_tensor_norm).float()), batch_size=args.batch_size, shuffle=True)
        vae_val_loader = DataLoader(TensorDataset(torch.from_numpy(vae_val_tensor_norm).float()), batch_size=args.batch_size)

        # 4.5. Bucle de entrenamiento del VAE
        vae_fold_k = ConvolutionalVAE(
            input_channels=num_input_channels_for_vae,
            latent_dim=args.latent_dim,
            image_size=current_global_tensor.shape[-1],
            num_conv_layers_encoder=args.num_conv_layers_encoder,
            decoder_type=args.decoder_type,
            dropout_rate=args.dropout_rate_vae,
            use_layernorm_fc=args.use_layernorm_vae_fc
        ).to(device)
        #vae_fold_k = ConvolutionalVAE(input_channels=num_input_channels_for_vae, latent_dim=args.latent_dim, image_size=current_global_tensor.shape[-1]).to(device)
        optimizer_vae = optim.Adam(vae_fold_k.parameters(), lr=args.lr_vae, weight_decay=args.weight_decay_vae)
        scheduler_vae = None
        if vae_val_loader and args.lr_scheduler_patience_vae > 0:
            scheduler_vae = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_vae, mode='min', patience=args.lr_scheduler_patience_vae,
                factor=0.1, verbose=False)

        # buffers de historial
        fold_vae_train_losses, fold_vae_val_losses, fold_vae_betas = [], [], []

        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(args.epochs_vae):
            epoch_train_loss = 0.0
            vae_fold_k.train()
            current_beta = get_cyclical_beta_schedule(epoch, args.epochs_vae, args.beta_vae, args.cyclical_beta_n_cycles, args.cyclical_beta_ratio_increase)
            total_recon, total_kld = 0.0, 0.0
            for (data,) in vae_train_loader:
                data = data.to(device)
                optimizer_vae.zero_grad()
                recon, mu, logvar, _ = vae_fold_k(data)
                # --- pérdidas desglosadas ---
                recon_loss = nn.functional.mse_loss(recon, data, reduction='sum') / data.size(0)
                kld_loss   = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)
                loss = recon_loss + current_beta * kld_loss
                loss.backward()
                optimizer_vae.step()
                epoch_train_loss += loss.item() * data.size(0)
                total_recon      += recon_loss.item() * data.size(0)
                total_kld        += kld_loss.item()   * data.size(0)

            avg_train_loss = epoch_train_loss / len(vae_train_loader.dataset)
            avg_recon_loss = total_recon      / len(vae_train_loader.dataset)
            avg_kld_loss   = total_kld        / len(vae_train_loader.dataset)
            fold_vae_train_losses.append(avg_train_loss)

            # Validación
            vae_fold_k.eval()
            val_loss = 0
            with torch.no_grad():
                for (data,) in vae_val_loader:
                    data = data.to(device)
                    recon, mu, logvar, _ = vae_fold_k(data)
                    val_loss += vae_loss_function(recon, data, mu, logvar, beta=current_beta).item() * data.size(0)
            
            avg_val_loss = val_loss / len(vae_val_loader.dataset)
            fold_vae_val_losses.append(avg_val_loss)
            fold_vae_betas.append(current_beta)
            if scheduler_vae:
                scheduler_vae.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(vae_fold_k.state_dict(), fold_output_dir / "best_vae_model.pt")
            else:
                epochs_no_improve += 1
            
            if epoch % args.log_interval_epochs_vae == 0:
                current_lr = optimizer_vae.param_groups[0]['lr']
                logger.info(
                    f"  {fold_idx_str} VAE Ep {epoch+1:3d} | "
                    f"LR {current_lr:.2e} | β {current_beta:.3f} | "
                    f"Train Recon {avg_recon_loss:8.2f}  KLD {avg_kld_loss:8.2f} | "
                    f"Val Loss {avg_val_loss:8.2f}"
                )
            if epochs_no_improve >= args.early_stopping_patience_vae:
                logger.info(f"  {fold_idx_str} Early stopping VAE en epoch {epoch+1}.")
                break
        
        # Cargar el mejor modelo VAE para este fold
        # -----------------------------------------------------------
        # Guardar historial y gráfica del VAE (si se solicitó)
        # -----------------------------------------------------------
        if args.save_vae_training_history:
            hist = {"train_loss": fold_vae_train_losses,
                    "val_loss": fold_vae_val_losses,
                    "beta": fold_vae_betas}
            joblib.dump(hist, fold_output_dir / f"vae_history_fold_{fold_idx+1}.joblib")
            try:
                fig, ax1 = plt.subplots(figsize=(10,5))
                ax1.plot(hist["train_loss"], label="Train Loss")
                ax1.plot(hist["val_loss"], label="Val Loss")
                ax1.set_xlabel("Epoch");  ax1.set_ylabel("Loss")
                ax1.legend(loc="upper left")
                ax2 = ax1.twinx()
                ax2.plot(hist["beta"], '--', label="Beta")
                ax2.set_ylabel("Beta");  ax2.legend(loc="upper right")
                plt.title(f"Fold {fold_idx+1} – VAE loss/beta")
                plt.savefig(fold_output_dir / f"vae_history_fold_{fold_idx+1}.png")
                plt.close(fig)
            except Exception as e:
                logger.warning(f"{fold_idx_str} No se pudo guardar la gráfica del historial VAE: {e}")

        vae_fold_k.load_state_dict(torch.load(fold_output_dir / "best_vae_model.pt"))
        
        # --- 5. Extracción de Features Latentes ---
        logger.info(f"  {fold_idx_str} Extrayendo features latentes con el VAE entrenado.")
        vae_fold_k.eval()
        with torch.no_grad():
            # Datos para el clasificador
            train_dev_global_indices = cn_ad_df.iloc[train_dev_clf_idx]['tensor_idx'].values
            test_global_indices = cn_ad_df.iloc[test_clf_idx]['tensor_idx'].values
            
            # Normalizar los datos del clasificador con los parámetros del VAE
            X_train_dev_norm = apply_normalization_params(current_global_tensor[train_dev_global_indices], norm_params)
            X_test_norm = apply_normalization_params(current_global_tensor[test_global_indices], norm_params)

            # Extraer features
            mu_train_dev, _ = vae_fold_k.encode(torch.from_numpy(X_train_dev_norm).float().to(device))
            X_latent_train_dev = mu_train_dev.cpu().numpy()
            y_train_dev = cn_ad_df.iloc[train_dev_clf_idx]['label'].values

            mu_test, _ = vae_fold_k.encode(torch.from_numpy(X_test_norm).float().to(device))
            X_latent_test = mu_test.cpu().numpy()
            y_test = cn_ad_df.iloc[test_clf_idx]['label'].values

        # --- 6. Entrenamiento y Evaluación de Clasificadores ---
        for clf_type in args.classifier_types:
            logger.info(f"  --- {fold_idx_str}, Clasificador: {clf_type} ---")
            
            X_train_primary, X_hp_tune, y_train_primary, y_hp_tune = sk_train_test_split(
                X_latent_train_dev, y_train_dev, test_size=args.classifier_hp_tune_ratio, 
                stratify=y_train_dev, random_state=args.seed)

            scaler = SklearnScaler()
            X_train_primary_scaled = scaler.fit_transform(X_train_primary)
            X_hp_tune_scaled = scaler.transform(X_hp_tune)
            
# --- Búsqueda de Hiperparámetros ---
            best_params = {}
            class_weight_mode = 'balanced' if args.classifier_use_class_weight else None
            optuna_classifiers = ['rf', 'gb', 'xgb', 'lgb']
            grid_classifiers = ['svm', 'logreg', 'mlp']

            if clf_type in optuna_classifiers and OPTUNA_AVAILABLE:
                # Comprobar si las librerías específicas están disponibles
                if (clf_type == 'xgb' and not XGBOOST_AVAILABLE) or \
                   (clf_type == 'lgb' and not LIGHTGBM_AVAILABLE):
                    logger.warning(f"La librería para el clasificador '{clf_type}' no está disponible. Saltando este clasificador.")
                    continue
                
                logger.info(f"Ajustando HPs para {clf_type} con Optuna ({args.optuna_n_trials} trials)...")
                # El pruner detiene trials que no son prometedores de forma temprana
                study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
                objective = get_optuna_objective(
                    X_train_primary_scaled, y_train_primary, X_hp_tune_scaled, y_hp_tune, 
                    clf_type, class_weight_mode, args.seed
                )
                # El timeout previene que la búsqueda se extienda indefinidamente
                study.optimize(objective, n_trials=args.optuna_n_trials, timeout=600) 
                best_params = study.best_params

            elif clf_type in grid_classifiers:
                logger.info(f"Ajustando HPs para {clf_type} con GridSearchCV...")
                model_for_grid = None
                param_grid = {}
                
                # Definir grids de búsqueda reducidos y sensatos
                if clf_type == 'svm':
                    model_for_grid = SVC(random_state=args.seed, class_weight=class_weight_mode)
                    param_grid = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 'scale'], 'kernel': ['rbf']}
                elif clf_type == 'logreg':
                    model_for_grid = LogisticRegression(random_state=args.seed, class_weight=class_weight_mode, solver='liblinear', max_iter=1000)
                    param_grid = {'C': [0.01, 0.1, 1, 10, 50]}
                elif clf_type == 'mlp':
                    model_for_grid = MLPClassifier(random_state=args.seed, max_iter=500, early_stopping=True, n_iter_no_change=15)
                    param_grid = {'hidden_layer_sizes': [(64,), (128, 64)], 'alpha': [0.001, 0.01, 0.1], 'learning_rate_init': [0.001, 0.005]}
                
                if model_for_grid:
                    # Combinar temporalmente los datos para el fit del grid search
                    X_hp_combined = np.vstack((X_train_primary_scaled, X_hp_tune_scaled))
                    y_hp_combined = np.hstack((y_train_primary, y_hp_tune))
                    
                    grid_search = GridSearchCV(model_for_grid, param_grid, cv=3, scoring='balanced_accuracy', n_jobs=args.n_jobs_gridsearch)
                    grid_search.fit(X_hp_combined, y_hp_combined)
                    best_params = grid_search.best_params_
                else:
                    logger.warning(f"No se definió un modelo para GridSearchCV para {clf_type}")

            else:
                logger.warning(f"Clasificador '{clf_type}' no soportado o su librería de optimización (Optuna) no está instalada. Se usarán HPs por defecto.")
                best_params = {} # Usará los parámetros por defecto del modelo

            logger.info(f"Mejores HPs encontrados para {clf_type}: {best_params}")

            # --- Entrenamiento del Modelo Final y Evaluación ---
            # Escalar el conjunto completo de train/dev con un nuevo scaler
            final_scaler = SklearnScaler()
            X_train_dev_scaled = final_scaler.fit_transform(X_latent_train_dev)
            X_test_scaled = final_scaler.transform(X_latent_test)
            
            # Instanciar modelo final con los mejores hiperparámetros
            final_model = None
            if clf_type == 'svm':
                # **MEJORA**: Usar CalibratedClassifierCV para probabilidades fiables
                # Nota: los HPs se encontraron en el modelo base, ahora lo envolvemos para calibrarlo
                base_svm = SVC(random_state=args.seed, class_weight=class_weight_mode, **best_params)
                # Entrenamos el calibrador en el mismo conjunto de datos de entrenamiento del clasificador
                final_model = CalibratedClassifierCV(base_svm, cv=3, method='isotonic') 
            elif clf_type == 'logreg':
                final_model = LogisticRegression(random_state=args.seed, class_weight=class_weight_mode, solver='liblinear', max_iter=2000, **best_params)
            elif clf_type == 'mlp':
                 final_model = MLPClassifier(random_state=args.seed, max_iter=750, early_stopping=True, n_iter_no_change=20, **best_params)
            elif clf_type == 'rf':
                final_model = RandomForestClassifier(random_state=args.seed, class_weight=class_weight_mode, **best_params)
            elif clf_type == 'gb':
                final_model = GradientBoostingClassifier(random_state=args.seed, **best_params)
            elif clf_type == 'xgb' and XGBOOST_AVAILABLE:
                # XGBoost usa 'scale_pos_weight' en lugar de 'class_weight'
                scale_pos_weight = (sum(y_train_dev == 0) / sum(y_train_dev == 1)) if class_weight_mode == 'balanced' else 1
                final_model = xgb.XGBClassifier(random_state=args.seed, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight, **best_params)
            elif clf_type == 'lgb' and LIGHTGBM_AVAILABLE:
                final_model = lgb.LGBMClassifier(random_state=args.seed, class_weight=class_weight_mode, **best_params)
            
            if final_model is not None:
                final_model.fit(X_train_dev_scaled, y_train_dev)
                y_pred_proba = final_model.predict_proba(X_test_scaled)[:, 1]
                y_pred = final_model.predict(X_test_scaled)

                fold_metrics = {
                    'fold': fold_idx + 1,
                    'classifier': clf_type,
                    'auc': roc_auc_score(y_test, y_pred_proba),
                    'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred),
                    'sensitivity': recall_score(y_test, y_pred, pos_label=1),
                    'specificity': recall_score(y_test, y_pred, pos_label=0),
                    'pr_auc': average_precision_score(y_test, y_pred_proba),
                    'accuracy': accuracy_score(y_test, y_pred),
                }
                all_folds_metrics.append(fold_metrics)
                logger.info(f"    Resultados {clf_type}: AUC={fold_metrics['auc']:.4f}, Bal. Acc.={fold_metrics['balanced_accuracy']:.4f}")

        logger.info(f"--- {fold_idx_str} completado en {time.time() - fold_start_time:.2f} segundos. ---")
    
    # --- 7. Reporte Final de Resultados ---
    if all_folds_metrics:
        metrics_df = pd.DataFrame(all_folds_metrics)
        logger.info("\n--- RESUMEN FINAL (Promedio +/- STD sobre Folds) ---")
        for clf_type in args.classifier_types:
            clf_df = metrics_df[metrics_df['classifier'] == clf_type]
            if not clf_df.empty:
                logger.info(f"\n  Clasificador: {clf_type}")
                for metric in ['auc', 'pr_auc', 'accuracy',
                            'balanced_accuracy', 'f1_score',
                            'sensitivity', 'specificity']:
                    mean = clf_df[metric].mean()
                    std = clf_df[metric].std()
                    logger.info(f"    {metric.capitalize():<20}: {mean:.4f} +/- {std:.4f}")
        
        results_csv_path = output_base_dir / "final_classification_metrics.csv"
        metrics_df.to_csv(results_csv_path, index=False)
        logger.info(f"\nResultados detallados de todos los folds guardados en: {results_csv_path}")

# ------------------------------------------------------------------
#  PARSER Y ARGUMENT GROUPS
# ------------------------------------------------------------------
def build_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena un VAE + clasificadores CN vs AD"
    )

    # ──────────── GENERAL ────────────
    group_general = parser.add_argument_group("General")   # ➊ CREA EL GRUPO
    group_general.add_argument(                           # ➋ AÑADE EL FLAG
        "--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO", help="Nivel global de logging (por defecto: INFO)."
    )
    group_general.add_argument("--seed", type=int, default=42)
    group_general.add_argument("--output_dir", type=str, default="outputs")
    group_general.add_argument("--channels_to_use", nargs="*", default=[])
    group_general.add_argument("--save_vae_training_history", action="store_true",
                               help="Guarda las curvas de loss/beta y el .joblib del historial.")
    group_general.add_argument("--save_fold_artefacts", action="store_true")
    group_general.add_argument("--optuna_n_trials", type=int, default=50,
                               help="Nº de trials para Optuna (clasificadores).")
    group_general.add_argument("--global_tensor_path", type=str,
                               default="data/global_tensor.npz",
                               help="Ruta al .npz con el tensor 4-D")
    group_general.add_argument("--metadata_path", type=str,
                               default="data/metadata.csv",
                               help="CSV con metadatos de sujetos")

    # --------------- VAE -------------------
    group_vae = parser.add_argument_group("VAE")
    group_vae.add_argument("--latent_dim", type=int, default=128)
    group_vae.add_argument("--epochs_vae", type=int, default=250)
    group_vae.add_argument("--batch_size", type=int, default=32)
    group_vae.add_argument("--lr_vae", type=float, default=1e-3)
    group_vae.add_argument("--weight_decay_vae", type=float, default=1e-5)
    group_vae.add_argument("--early_stopping_patience_vae", type=int, default=40)
    group_vae.add_argument("--beta_vae", type=float, default=1.0)
    group_vae.add_argument("--cyclical_beta_n_cycles", type=int, default=4)
    group_vae.add_argument("--cyclical_beta_ratio_increase", type=float, default=0.3)
    group_vae.add_argument("--vae_val_split_ratio", type=float, default=0.2)
    group_vae.add_argument("--lr_scheduler_patience_vae", type=int, default=20)
    group_vae.add_argument("--log_interval_epochs_vae", type=int, default=20,
                           help="Cada cuántos epochs imprimir LR, β, recon_loss y KLD.")
    # nuevas HPs de arquitectura / regularización
    group_vae.add_argument("--num_conv_layers_encoder", type=int, choices=[3,4],
                           default=4)
    group_vae.add_argument("--decoder_type", type=str,
                           choices=["convtranspose","upsample_conv"],
                           default="convtranspose")
    group_vae.add_argument("--dropout_rate_vae", type=float, default=0.2)
    group_vae.add_argument("--use_layernorm_vae_fc", action="store_true")

    # --------------- CROSS-VALIDATION (group_cv) ---------------
    group_cv = parser.add_argument_group("Cross-validation")
    group_cv.add_argument("--outer_folds", type=int, default=5)
    group_cv.add_argument("--repeated_outer_folds_n_repeats", type=int, default=1,
                          help=">1 activa RepeatedStratifiedKFold.")
    group_cv.add_argument("--classifier_hp_tune_ratio", type=float, default=0.2)

    # --------------- CLASIFICADORES ---------------
    group_clf = parser.add_argument_group("Clasificadores")
    group_clf.add_argument("--classifier_types", nargs="+",
                           default=["svm", "rf", "xgb", "lgb"])
    group_clf.add_argument("--classifier_use_class_weight", action="store_true")
    group_clf.add_argument("--classifier_stratify_cols", nargs="*", default=[])
    group_clf.add_argument("--n_jobs_gridsearch", type=int, default=-1)

    # --------------- NORMALIZACIÓN ---------------
    group_norm = parser.add_argument_group("Normalización")
    group_norm.add_argument("--norm_mode", type=str, default="zscore_offdiag",
                            choices=["zscore_offdiag"])

    return parser.parse_args()


# ------------------------------------------------------------------
#  MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    
    args = build_arg_parser()

    # Rutas a los datos (ajústalas o pásalas por argumentos si prefieres)
    tensor_path = Path(args.global_tensor_path)
    metadata_path = Path(args.metadata_path)
    logger = init_logger(args.log_level)

    tensor, meta_df, channel_names = load_data(tensor_path, metadata_path)
    if tensor is None:
        raise RuntimeError("No se pudieron cargar datos.")

    train_and_evaluate_pipeline(
        tensor,
        meta_df,
        channel_names,
        args,
    )
