"""
Componente para preprocesamiento de datos
Descarga datos, limpia y divide en train/test usando lógica de DelayModel
"""
from typing import NamedTuple
from kfp import dsl


@dsl.component(
    base_image="python:3.12",
    packages_to_install=[
        "pandas>=2.1.4",
        "scikit-learn>=1.3.2",
        "numpy>=1.26.0",
        "google-cloud-storage==2.14.0",
    ],
)
def data_preprocessing(
    input_data_path: str,
    train_split: float = 0.8,
    random_state: int = 42,
) -> NamedTuple("Outputs", [("train_data_path", str), ("test_data_path", str)]):
    """
    Preprocesa los datos usando la lógica de DelayModel: descarga, limpia y divide en train/test
    
    Args:
        input_data_path: Ruta GCS del dataset de entrada
        train_split: Proporción de datos para entrenamiento (0-1)
        random_state: Semilla para reproducibilidad
    
    Returns:
        Tupla con rutas de datos de train y test
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    from google.cloud import storage
    import os
    from collections import namedtuple
    
    # Construir rutas de salida basándose en input_data_path
    if input_data_path.endswith('.csv'):
        base_path = input_data_path[:-4]
    else:
        base_path = input_data_path
    
    output_train_path = f"{base_path}_train.csv"
    output_test_path = f"{base_path}_test.csv"
    
    # Función auxiliar get_min_diff (incluida en el componente)
    def get_min_diff(row):
        """Calcula la diferencia mínima entre fechas programadas y reales"""
        try:
            fecha_o = pd.to_datetime(row['Fecha-O'], errors='coerce')
            fecha_i = pd.to_datetime(row['Fecha-I'], errors='coerce')
            
            if pd.isna(fecha_o) or pd.isna(fecha_i):
                return 0
            
            # Calcular diferencia en minutos
            # (fecha_o - fecha_i) es un Timedelta, usar total_seconds() para obtener segundos y convertir a minutos
            timedelta = fecha_o - fecha_i
            min_diff = timedelta.total_seconds() / 60.0
            return float(min_diff)
        except (KeyError, ValueError, TypeError):
            return 0
    
    # Lista de top 10 features (ajustar según tu análisis)
    top_10_features = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]
    
    # Inicializar cliente de GCS
    storage_client = storage.Client()
    
    # Descargar datos desde GCS
    bucket_name = input_data_path.split("/")[2]
    blob_path = "/".join(input_data_path.split("/")[3:])
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    local_path = "/tmp/data.csv"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    
    # Cargar datos
    df = pd.read_csv(local_path)
    
    # Preprocesar usando lógica exacta de DelayModel.preprocess
    # Paso 1: Copiar datos (como en DelayModel.preprocess línea 36)
    data = df.copy()
    
    # Paso 2: Crear target delay si no existe (lógica de DelayModel.preprocess líneas 38-40)
    if 'delay' not in data.columns:
        data['min_diff'] = data.apply(get_min_diff, axis=1)
        data['delay'] = np.where(data['min_diff'] > 15, 1, 0)
    
    # Paso 3: Shuffle con las columnas específicas y random_state=111 
    # (exactamente como en DelayModel.preprocess línea 41)
    required_cols = ['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'delay']
    available_cols = [col for col in required_cols if col in data.columns]
    
    # Solo hacer shuffle si tenemos 'delay' (como en DelayModel cuando target_column no es None)
    if 'delay' in available_cols:
        # Shuffle exactamente como en DelayModel.preprocess línea 41
        data = shuffle(data[available_cols], random_state=111)
    
    # Paso 4: Dividir en train/test DESPUÉS del shuffle (pero ANTES del encoding categórico)
    # El encoding se hará en training/evaluation para mantener consistencia con DelayModel
    train_df, test_df = train_test_split(
        data, 
        test_size=1 - train_split, 
        random_state=random_state,
        stratify=data['delay'] if 'delay' in data.columns else None
    )
    
    # Guardar datasets procesados localmente
    # NOTA: No hacemos encoding aquí porque DelayModel.preprocess hace el encoding
    # justo después del shuffle, pero como dividimos train/test, el encoding debe hacerse
    # por separado en training y evaluation para asegurar consistencia de columnas
    train_local = "/tmp/train_data.csv"
    test_local = "/tmp/test_data.csv"
    train_df.to_csv(train_local, index=False)
    test_df.to_csv(test_local, index=False)
    
    # Subir a GCS
    train_blob = bucket.blob(output_train_path.split(f"gs://{bucket_name}/")[-1])
    test_blob = bucket.blob(output_test_path.split(f"gs://{bucket_name}/")[-1])
    
    train_blob.upload_from_filename(train_local)
    test_blob.upload_from_filename(test_local)
    
    # Retornar rutas
    Outputs = namedtuple("Outputs", ["train_data_path", "test_data_path"])
    return Outputs(
        train_data_path=output_train_path,
        test_data_path=output_test_path
    )

