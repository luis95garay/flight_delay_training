"""
Componente para evaluar el modelo entrenado usando DelayModel
Evalúa el modelo con datos de test y genera métricas
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
def evaluate_model(
    model_path: str,
    feature_columns_path: str,
    test_data_path: str,
    evaluation_output_path: str,
) -> NamedTuple("Outputs", [("evaluation_metrics", str), ("accuracy", float)]):
    """
    Evalúa el modelo entrenado con datos de test usando lógica de DelayModel

    Args:
        model_path: Ruta GCS del modelo entrenado
        feature_columns_path: Ruta GCS del archivo con columnas de features guardadas
        test_data_path: Ruta GCS del dataset de test
        evaluation_output_path: Ruta GCS donde guardar métricas de evaluación

    Returns:
        Tupla con ruta de métricas y accuracy
    """
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import json
    from google.cloud import storage
    import os
    from collections import namedtuple

    # Lista local de features esperadas para asegurar consistencia con entrenamiento
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
        "OPERA_Copa Air",
    ]

    # Inicializar cliente de GCS
    storage_client = storage.Client()
    
    # Descargar modelo
    bucket_name = model_path.split("/")[2]
    bucket = storage_client.bucket(bucket_name)
    
    model_blob_path = "/".join(model_path.split("/")[3:])
    model_blob = bucket.blob(model_blob_path)
    model_local = "/tmp/model.joblib"
    os.makedirs(os.path.dirname(model_local), exist_ok=True)
    model_blob.download_to_filename(model_local)
    model = joblib.load(model_local)
    
    # Descargar columnas de features guardadas
    feature_columns_blob_path = "/".join(feature_columns_path.split("/")[3:])
    feature_columns_blob = bucket.blob(feature_columns_blob_path)
    feature_columns_local = "/tmp/feature_columns.json"
    feature_columns_blob.download_to_filename(feature_columns_local)
    
    with open(feature_columns_local, "r") as f:
        feature_info = json.load(f)
    expected_features = feature_info.get("feature_columns", top_10_features)
    
    # Descargar datos de test
    test_blob_path = "/".join(test_data_path.split("/")[3:])
    test_blob = bucket.blob(test_blob_path)
    test_local = "/tmp/test_data.csv"
    test_blob.download_to_filename(test_local)
    
    # Cargar datos de test
    data = pd.read_csv(test_local)
    
    # Preprocesar usando lógica de DelayModel.preprocess
    # Encoding categórico (mismo que en training)
    opera_dummies = pd.get_dummies(data['OPERA'], prefix='OPERA', dummy_na=True)
    tipovuelo_dummies = pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO', dummy_na=True)
    mes_dummies = pd.get_dummies(data['MES'], prefix='MES')
    
    features = pd.concat([opera_dummies, tipovuelo_dummies, mes_dummies], axis=1)
    
    # Asegurar consistencia de columnas con entrenamiento
    # Agregar columnas faltantes con 0
    for col in expected_features:
        if col not in features.columns:
            features[col] = 0
    
    # Seleccionar solo las columnas esperadas
    features = features[expected_features]
    
    # Separar target
    y_test = data['delay']
    
    # Predecir usando lógica de DelayModel.predict
    y_pred = model.predict(features)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    metrics = {
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": cm,
        "n_test_samples": len(features)
    }
    
    # Guardar métricas
    metrics_local = "/tmp/evaluation_metrics.json"
    with open(metrics_local, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Subir métricas a GCS
    eval_blob_path = "/".join(evaluation_output_path.split("/")[3:])
    eval_blob = bucket.blob(eval_blob_path)
    eval_blob.upload_from_filename(metrics_local)
    
    Outputs = namedtuple("Outputs", ["evaluation_metrics", "accuracy"])
    return Outputs(
        evaluation_metrics=evaluation_output_path,
        accuracy=float(accuracy)
    )

