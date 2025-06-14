#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilitario para Jupyter Notebook (Versión Adaptada):
1. Carga un tensor de conectividad (.npz) con metadatos enriquecidos.
2. Inspecciona su metadata completa.
3. Visualiza matrices de conectividad con etiquetas de redes funcionales para
   una mejor interpretación.
4. Calcula y muestra rangos de valores por canal a través de todos los sujetos.
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import random

# ------- Utilidades Auxiliares ---------

def load_npz_tensor(npz_path: Path) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Carga un archivo .npz, separa el tensor principal de la metainformación
    y devuelve ambos. Adaptado para los nuevos tensores con metadatos variables.
    """
    if not npz_path.exists():
        print(f"Error: Archivo de tensor no encontrado en {npz_path}", file=sys.stderr)
        return None, {}

    print(f"Cargando tensor desde: {npz_path.name}")
    data = np.load(npz_path, allow_pickle=True)
    
    tensor_key = None
    if 'global_tensor_data' in data:
        tensor_key = 'global_tensor_data'
    elif 'tensor_data' in data:
        tensor_key = 'tensor_data'
    else:
        print("Error: No se encontró una clave de tensor válida ('global_tensor_data' o 'tensor_data') en el archivo.", file=sys.stderr)
        return None, {}

    tensor = data[tensor_key]
    meta = {k: v for k, v in data.items() if k != tensor_key}
    
    # Estandarizar nombres de claves para consistencia
    if 'subject_id' in meta and 'subject_ids' not in meta:
        meta['subject_ids'] = np.array([meta['subject_id']])

    print("\n--- Metadatos Cargados ---")
    for key, value in meta.items():
        if isinstance(value, np.ndarray):
            # Imprime la forma y el tipo para arrays, y una muestra si no es muy grande
            print(f"  - {key}: array de forma {value.shape}, tipo {value.dtype}")
            if value.size < 10:
                print(f"    Contenido: {value}")
        else:
            print(f"  - {key}: {value}")
    print("--------------------------\n")

    return tensor, meta

def plot_matrix_with_network_labels(matrix: np.ndarray, title: str, network_labels: Optional[List[str]], save_fig: Optional[Path]):
    """
    Visualiza una matriz de conectividad, usando etiquetas de redes funcionales
    en los ejes si están disponibles.
    """
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        print(f"Advertencia: Matriz inválida para graficar (forma: {matrix.shape}). Saltando gráfico.")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap='viridis', aspect='equal')
    ax.set_title(title, fontsize=11, pad=20)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- INICIO DE LA CORRECCIÓN ---
    # Se verifica si 'network_labels' no es None antes de intentar acceder a su longitud.
    if network_labels is not None and len(network_labels) == matrix.shape[0]:
    # --- FIN DE LA CORRECCIÓN ---
        # Encontrar los límites de las redes
        unique_labels, first_indices = np.unique(network_labels, return_index=True)
        boundaries = list(first_indices) + [len(network_labels)]
        
        # Preparar etiquetas y posiciones para los ticks
        tick_labels = [label.replace("_", "\n") for label in unique_labels]
        tick_positions = [first_indices[i] + (boundaries[i+1] - first_indices[i]) / 2 for i in range(len(unique_labels))]

        # Dibujar líneas de separación de redes
        for idx in first_indices[1:]:
            ax.axhline(idx - 0.5, color='white', lw=0.8, linestyle='--')
            ax.axvline(idx - 0.5, color='white', lw=0.8, linestyle='--')

        # Configurar los ticks para que muestren los nombres de las redes
        ax.xaxis.set_major_locator(mticker.FixedLocator(tick_positions))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter(tick_labels))
        ax.yaxis.set_major_locator(mticker.FixedLocator(tick_positions))
        ax.yaxis.set_major_formatter(mticker.FixedFormatter(tick_labels))
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)
        plt.setp(ax.get_yticklabels(), fontsize=8)
    else:
        ax.set_xlabel('Índice de ROI')
        ax.set_ylabel('Índice de ROI')

    plt.tight_layout()
    if save_fig:
        save_fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig, dpi=300, bbox_inches='tight')
        print(f"Figura guardada en: {save_fig}")
    else:
        plt.show()
    plt.close(fig)

def plot_hist(values: np.ndarray, title: str, save_fig: Optional[Path]):
    """Visualiza un histograma de los valores de la matriz."""
    if values.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(values.flatten(), bins=50, color='deepskyblue', edgecolor='black', alpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Valor de Conectividad')
    ax.set_ylabel('Frecuencia')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if save_fig:
        save_fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig, dpi=300, bbox_inches='tight')
        print(f"Histograma guardado en: {save_fig}")
    else:
        plt.show()
    plt.close(fig)

def calculate_and_print_channel_ranges(tensor_global: np.ndarray, channel_names: List[str]) -> Dict[str, Dict[str, float]]:
    """Calcula y muestra el rango (mínimo y máximo) de valores para cada canal."""
    if tensor_global.ndim != 4:
        print("Cálculo de rango solo disponible para tensores globales (Sujetos, Canales, ROIs, ROIs).")
        return {}
    
    print("\n--- Rango de Valores por Canal (a través de TODOS los sujetos) ---")
    channel_value_ranges = {}
    for ch_idx, ch_name in enumerate(channel_names):
        channel_data = tensor_global[:, ch_idx, :, :]
        min_val = np.min(channel_data)
        max_val = np.max(channel_data)
        channel_value_ranges[ch_name] = {'min': float(min_val), 'max': float(max_val)}
        print(f"  - Canal '{ch_name}': Mínimo = {min_val:.4f}, Máximo = {max_val:.4f}")
    print("----------------------------------------------------------------\n")
    return channel_value_ranges

# ------- Función Principal para Notebook -------------

def visualize_tensor_data(
    tensor_path_str: str, 
    channel_to_show: Union[str, int, List[Union[str, int]], None] = "all",
    subject_id_or_index: str = "0",
    show_average: bool = False, 
    symmetrize_mode: str = 'abs', 
    save_figure_prefix: Optional[str] = None
):
    """Función principal para cargar, procesar y visualizar un tensor de conectividad."""
    tensor_path = Path(tensor_path_str)
    
    try:
        tensor, meta = load_npz_tensor(tensor_path)
        if tensor is None: return

        # Extraer nombres de canales y sujetos de la metadata
        channel_names = [str(name) for name in meta.get('channel_names', [])]
        subject_ids = [str(sid) for sid in meta.get('subject_ids', [])]
        network_labels = meta.get('network_labels_in_order') # Clave de los nuevos tensores

        # Calcular rangos si es un tensor global
        if tensor.ndim == 4:
            calculate_and_print_channel_ranges(tensor, channel_names)

        # Seleccionar canales a visualizar
        if channel_to_show is None or (isinstance(channel_to_show, str) and channel_to_show.lower() == 'all'):
            ch_indices = list(range(len(channel_names)))
        else:
            ch_indices = [int(channel_to_show)] if isinstance(channel_to_show, (str, int)) and channel_to_show in [str(i) for i in range(len(channel_names))] else []


        # Determinar sujeto o promedio
        matrix_source = "Promedio Global" if show_average and tensor.ndim == 4 else "Sujeto Individual"
        
        for ch_idx in ch_indices:
            ch_name = channel_names[ch_idx]
            
            matrix_to_plot = None
            title_subject_part = ""

            if tensor.ndim == 4: # Global
                if show_average:
                    matrix_to_plot = np.mean(tensor[:, ch_idx, :, :], axis=0)
                    title_subject_part = f"Promedio de {tensor.shape[0]} sujetos"
                else:
                    subj_idx = int(subject_id_or_index) if subject_id_or_index.isdigit() else subject_ids.index(subject_id_or_index)
                    matrix_to_plot = tensor[subj_idx, ch_idx, :, :]
                    title_subject_part = f"Sujeto: {subject_ids[subj_idx]}"
            else: # Individual
                matrix_to_plot = tensor[ch_idx, :, :]
                title_subject_part = f"Sujeto: {subject_ids[0]}"

            if matrix_to_plot is not None:
                matrix_processed = np.abs(matrix_to_plot) if symmetrize_mode == 'abs' else matrix_to_plot
                
                base_title = f"Canal: {ch_name}\n{title_subject_part}"
                
                save_path_matrix, save_path_hist = None, None
                if save_figure_prefix:
                    prefix_path = Path(save_figure_prefix)
                    clean_ch_name = "".join(c for c in ch_name if c.isalnum())
                    clean_subj_id = title_subject_part.replace(" ", "_").replace(":", "")
                    fig_stem = f"{prefix_path.name}_{clean_ch_name}_{clean_subj_id}"
                    save_path_matrix = prefix_path.with_name(f"{fig_stem}_matrix.png")
                    save_path_hist = prefix_path.with_name(f"{fig_stem}_hist.png")

                plot_matrix_with_network_labels(matrix_processed, base_title, network_labels, save_path_matrix)
                plot_hist(matrix_processed.flatten(), base_title, save_path_hist)

    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

# --- EJEMPLOS DE USO (AJUSTA LA RUTA AL TENSOR GLOBAL) ---

# ‼️ IMPORTANTE: Modifica esta línea con la ruta a tu nuevo archivo de tensor global ‼️
RUTA_TENSOR_GLOBAL = "/home/diego/Escritorio/AAL3_paper/AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced_v6.5.21_AAL3_131ROIs_OMST_GCE_Signed_Granger_lag1__ChNorm_ROIreorderedYeo17_ParallelTuned/GLOBAL_TENSOR_from_AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced_v6.5.21_AAL3_131ROIs_OMST_GCE_Signed_Granger_lag1__ChNorm_ROIreorderedYeo17_ParallelTuned.npz"

# Directorio para guardar las figuras generadas
SAVE_DIR = Path("./visualizaciones_tensor")

if Path(RUTA_TENSOR_GLOBAL).exists():
    # --- CASO 1: Visualizar el promedio de todos los sujetos para TODOS los canales ---
    print("\n\n--- CASO 1: Visualizando Promedio Global para TODOS los Canales ---")
    visualize_tensor_data(
        tensor_path_str=RUTA_TENSOR_GLOBAL,
        channel_to_show="all", 
        show_average=True,
        save_figure_prefix=str(SAVE_DIR / "promedio_global")
    )

    # --- CASO 2: Visualizar un sujeto específico (el primero, índice 0) para un canal específico (el de Granger, índice 4) ---
    print("\n\n--- CASO 2: Visualizando Sujeto en índice 0, Canal 'Granger' ---")
    visualize_tensor_data(
        tensor_path_str=RUTA_TENSOR_GLOBAL,
        channel_to_show=4, # Índice del canal de Granger
        subject_id_or_index="0", # Índice del primer sujeto
        show_average=False,
        save_figure_prefix=str(SAVE_DIR / "sujeto_0_canal_granger")
    )
else:
    print(f"ADVERTENCIA: El archivo de tensor no se encontró en la ruta especificada.")
    print(f"Por favor, actualiza la variable 'RUTA_TENSOR_GLOBAL' con la ruta correcta a tu archivo .npz")