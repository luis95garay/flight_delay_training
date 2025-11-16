"""
Script para enviar el pipeline a Vertex AI

Requiere que el paquete esté instalado en modo desarrollo: pip install -e .
"""
import sys
from google.cloud import aiplatform
from kfp import compiler
from pipelines.pipeline import flight_delay_training_pipeline
import yaml


def submit_pipeline(
    config_path: str = "configs/config.yaml",
    pipeline_file: str = "flight_delay_pipeline.json",
):
    """
    Compila y envía el pipeline a Vertex AI
    
    Args:
        config_path: Ruta al archivo de configuración
        pipeline_file: Nombre del archivo compilado del pipeline
    """
    # Cargar configuración
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    gcp_config = config.get("gcp", {})
    pipeline_config = config.get("pipeline", {})
    
    project_id = gcp_config.get("project_id")
    region = gcp_config.get("region", "us-central1")
    
    if not project_id:
        print("✗ Error: project_id no configurado en config.yaml")
        sys.exit(1)
    
    # Compilar pipeline
    print("Compilando pipeline...")
    compiler.Compiler().compile(
        pipeline_func=flight_delay_training_pipeline,
        package_path=pipeline_file
    )
    print(f"✓ Pipeline compilado en {pipeline_file}")
    
    # Inicializar Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    # Obtener parámetros del pipeline desde la configuración
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    
    # Preparar parámetros del pipeline
    parameter_values = {
        "project_id": project_id,
        "region": region,
        "input_data_path": data_config.get("source_path"),
        "train_split": data_config.get("train_split", 0.8),
        "random_state": data_config.get("random_state", 42),
        "learning_rate": model_config.get("parameters", {}).get("learning_rate", 0.01),
        "model_name": model_config.get("name", "flight_delay_model"),
    }
    
    # Validar parámetros requeridos
    if not parameter_values.get("input_data_path"):
        print("✗ Error: input_data_path (data.source_path) no configurado en config.yaml")
        sys.exit(1)
    
    # Crear job de pipeline
    job = aiplatform.PipelineJob(
        display_name=pipeline_config.get("display_name", "Flight Delay Training Pipeline"),
        template_path=pipeline_file,
        pipeline_root=f"gs://{gcp_config.get('bucket_name')}/pipeline_root",
        enable_caching=True,
        parameter_values=parameter_values,
    )
    
    # Ejecutar pipeline
    print("Enviando pipeline a Vertex AI...")
    job.run()
    print(f"✓ Pipeline enviado. Job ID: {job.resource_name}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    submit_pipeline(config)

