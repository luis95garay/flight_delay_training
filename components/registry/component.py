"""
Componente para registrar un modelo en Vertex AI Model Registry
Sube el artefacto a una carpeta limpia con model.joblib y realiza el registro.
"""
from kfp import dsl
from typing import NamedTuple


@dsl.component(
    base_image="python:3.12",
    packages_to_install=[
        "google-cloud-storage==2.14.0",
        "google-cloud-aiplatform>=1.45.0",
    ],
)
def register_model(
    model_path: str,
    project_id: str,
    location: str = "us-central1",
    model_display_name: str = "flight-delay-xgb",
) -> NamedTuple("Outputs", [("vertex_model_resource_name", str), ("artifact_uri", str)]):
    """
    Registra el modelo en Vertex AI Model Registry a partir de un artefacto en GCS.

    Args:
        model_path: Ruta GCS del archivo del modelo (p.ej., gs://bucket/models/name.joblib)
        project_id: ID de proyecto de GCP
        location: Región (p.ej., us-central1)
        model_display_name: Nombre visible del modelo en el registry
    """
    from collections import namedtuple
    from google.cloud import storage, aiplatform
    import os

    # Derivar bucket y rutas
    bucket_name = model_path.split("/")[2]
    relative_model_path = "/".join(model_path.split("/")[3:])  # p.ej. models/name.joblib
    base_without_ext, _ = os.path.splitext(relative_model_path)  # p.ej. models/name
    registry_dir_rel = f"{base_without_ext}/"  # p.ej. models/name/
    registry_model_rel = f"{registry_dir_rel}model.joblib"  # p.ej. models/name/model.joblib
    artifact_uri = f"gs://{bucket_name}/{registry_dir_rel}"

    # Subir copia del modelo a ruta limpia como model.joblib
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    src_blob = bucket.blob(relative_model_path)
    # Descargar temporal y volver a subir como model.joblib para asegurar nombre esperado
    local_tmp = "/tmp/model.joblib"
    os.makedirs("/tmp", exist_ok=True)
    src_blob.download_to_filename(local_tmp)
    dst_blob = bucket.blob(registry_model_rel)
    dst_blob.upload_from_filename(local_tmp)

    # Registrar en Vertex
    aiplatform.init(project=project_id, location=location, staging_bucket=f"gs://{bucket_name}")
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest",
        description="Registered via conditional step after evaluation",
        labels={"pipeline": "flight_delay", "source": "kfp"},
    )

    Outputs = namedtuple("Outputs", ["vertex_model_resource_name", "artifact_uri"])
    return Outputs(vertex_model_resource_name=model.resource_name, artifact_uri=artifact_uri)


