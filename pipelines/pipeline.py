"""
Pipeline principal de Vertex AI para entrenamiento de modelo
Usa Kubeflow Pipelines SDK v2
"""
from kfp.dsl import pipeline

from components.data_preprocessing.component import data_preprocessing
from components.training.component import train_model
from components.evaluation.component import evaluate_model
from utils.config_loader import load_config


# Cargamos configuración para obtener valores por defecto y evitar hardcodear rutas
_config = load_config()
_DEFAULT_BUCKET_NAME = _config["gcp"]["bucket_name"]
_DEFAULT_MODEL_NAME = _config["model"]["name"]


@pipeline(
    name="flight-delay-training-pipeline",
    description="Pipeline completo para entrenar modelo de predicción de retrasos de vuelos usando DelayModel con XGBoost",
)
def flight_delay_training_pipeline(
    project_id: str,
    region: str,
    input_data_path: str,
    train_split: float = 0.8,
    random_state: int = 42,
    learning_rate: float = 0.01,
    bucket_name: str = _DEFAULT_BUCKET_NAME,
    model_name: str = _DEFAULT_MODEL_NAME,
):
    """
    Pipeline completo de ML para entrenamiento de modelo usando DelayModel con XGBoost

    Args:
        project_id: ID del proyecto de GCP
        region: Región de GCP (ej: us-central1)
        input_data_path: Ruta GCS del dataset de entrada
        train_split: Proporción de datos para entrenamiento
        random_state: Semilla para reproducibilidad en división train/test
        learning_rate: Tasa de aprendizaje para XGBoost
        bucket_name: Nombre del bucket de GCS donde se guardan los artefactos
        model_name: Nombre del modelo
    """

    # Rutas de salida basadas en configuración/argumentos
    model_output_path = f"gs://{bucket_name}/models/{model_name}.joblib"
    feature_columns_output_path = f"gs://{bucket_name}/models/{model_name}_feature_columns.json"
    evaluation_output_path = f"gs://{bucket_name}/evaluations/{model_name}_evaluation.json"

    # Paso 1: Preprocesamiento de datos
    # Las rutas de train/test se construyen automáticamente dentro del componente
    preprocess_op = data_preprocessing(
        input_data_path=input_data_path,
        train_split=train_split,
        random_state=random_state,
    )
    
    # Paso 2: Entrenamiento del modelo usando DelayModel con XGBoost
    train_op = train_model(
        train_data_path=preprocess_op.outputs["train_data_path"],
        model_output_path=model_output_path,
        feature_columns_output_path=feature_columns_output_path,
        learning_rate=learning_rate,
        random_state=1,  # Random state específico para XGBoost
    )
    
    # Paso 3: Evaluación del modelo
    evaluate_op = evaluate_model(
        model_path=train_op.outputs["model_path"],
        feature_columns_path=train_op.outputs["feature_columns_path"],
        test_data_path=preprocess_op.outputs["test_data_path"],
        evaluation_output_path=evaluation_output_path,
    )


if __name__ == "__main__":
    # Compilar el pipeline
    from kfp import compiler
    
    compiler.Compiler().compile(
        pipeline_func=flight_delay_training_pipeline,
        package_path="flight_delay_pipeline.json"
    )
    print("Pipeline compilado exitosamente en flight_delay_pipeline.json")

