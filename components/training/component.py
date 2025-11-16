"""
Componente para entrenar el modelo usando DelayModel con XGBoost
Entrena un modelo de ML y lo guarda en GCS
"""
from typing import NamedTuple
from kfp import dsl


@dsl.component(
    base_image="python:3.12",
    packages_to_install=[
        "pandas>=2.1.4",
        "scikit-learn>=1.3.2",
        "numpy>=1.26.0",
        "xgboost>=2.0.3",
        "google-cloud-storage==2.14.0",
    ],
)
def train_model(
    train_data_path: str,
    model_output_path: str,
    feature_columns_output_path: str,
    learning_rate: float = 0.01,
    random_state: int = 1,
) -> NamedTuple("Outputs", [("model_path", str), ("model_metrics", str), ("feature_columns_path", str)]):
    """
    Entrena un modelo usando la lógica de DelayModel con XGBoost
    
    Args:
        train_data_path: Ruta GCS del dataset de entrenamiento
        model_output_path: Ruta GCS donde guardar el modelo entrenado
        feature_columns_output_path: Ruta GCS donde guardar las columnas de features para consistencia
        learning_rate: Tasa de aprendizaje para XGBoost
        random_state: Semilla para reproducibilidad
    
    Returns:
        Tupla con ruta del modelo, métricas y columnas de features
    """
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import accuracy_score
    import joblib
    import json
    from google.cloud import storage
    import os
    from collections import namedtuple
    
    # Lista de top 10 features
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
    
    # Descargar datos de entrenamiento
    bucket_name = train_data_path.split("/")[2]
    blob_path = "/".join(train_data_path.split("/")[3:])
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    local_path = "/tmp/train_data.csv"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    
    # Cargar datos
    data = pd.read_csv(local_path)
    
    # Preprocesar usando lógica de DelayModel.preprocess
    # Encoding categórico
    opera_dummies = pd.get_dummies(data['OPERA'], prefix='OPERA', dummy_na=True)
    tipovuelo_dummies = pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO', dummy_na=True)
    mes_dummies = pd.get_dummies(data['MES'], prefix='MES')
    
    features = pd.concat([opera_dummies, tipovuelo_dummies, mes_dummies], axis=1)
    
    # Guardar columnas de features para consistencia en predicción
    feature_columns = features.columns.tolist()
    
    # Filtrar a top 10 features (si están disponibles)
    available_top_features = [f for f in top_10_features if f in features.columns]
    if available_top_features:
        features = features[available_top_features]
    else:
        # Si no están disponibles, usar las primeras 10 columnas
        features = features.iloc[:, :10]
    
    # Separar target
    target = data['delay']
    
    # Calcular scale_pos_weight para balancear clases
    n_y0 = len(target[target == 0])
    n_y1 = len(target[target == 1])
    scale = n_y0 / n_y1 if n_y1 > 0 else 1.0
    
    # Entrenar modelo XGBoost usando lógica de DelayModel.fit
    xgb_model = xgb.XGBClassifier(
        random_state=random_state,
        learning_rate=learning_rate,
        scale_pos_weight=scale
    )
    xgb_model.fit(features, target)
    
    # Calcular métricas básicas
    y_pred = xgb_model.predict(features)
    accuracy = accuracy_score(target, y_pred)
    
    metrics = {
        "accuracy": float(accuracy),
        "n_samples": len(features),
        "n_features": features.shape[1],
        "scale_pos_weight": float(scale),
        "n_y0": int(n_y0),
        "n_y1": int(n_y1)
    }
    
    # Guardar modelo localmente
    model_local = "/tmp/model.joblib"
    joblib.dump(xgb_model, model_local)
    
    # Guardar métricas
    metrics_local = "/tmp/metrics.json"
    with open(metrics_local, "w") as f:
        json.dump(metrics, f)
    
    # Guardar columnas de features para consistencia
    feature_columns_local = "/tmp/feature_columns.json"
    with open(feature_columns_local, "w") as f:
        json.dump({"feature_columns": list(features.columns), "all_columns": feature_columns}, f)
    
    # Subir modelo a GCS
    model_blob_path = model_output_path.split(f"gs://{bucket_name}/")[-1]
    model_blob = bucket.blob(model_blob_path)
    model_blob.upload_from_filename(model_local)
    
    # Subir métricas a GCS
    metrics_blob_path = model_output_path.replace(".joblib", "_metrics.json")
    metrics_blob_path = metrics_blob_path.split(f"gs://{bucket_name}/")[-1]
    metrics_blob = bucket.blob(metrics_blob_path)
    metrics_blob.upload_from_filename(metrics_local)
    
    # Subir columnas de features a GCS
    feature_columns_blob_path = feature_columns_output_path.split(f"gs://{bucket_name}/")[-1]
    feature_columns_blob = bucket.blob(feature_columns_blob_path)
    feature_columns_blob.upload_from_filename(feature_columns_local)
    
    metrics_path = model_output_path.replace(".joblib", "_metrics.json")
    
    Outputs = namedtuple("Outputs", ["model_path", "model_metrics", "feature_columns_path"])
    return Outputs(
        model_path=model_output_path,
        model_metrics=metrics_path,
        feature_columns_path=feature_columns_output_path
    )

