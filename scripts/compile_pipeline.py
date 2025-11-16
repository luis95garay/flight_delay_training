"""
Script para compilar el pipeline de Vertex AI

Requiere que el paquete esté instalado en modo desarrollo: pip install -e .
"""
import sys
from kfp import compiler
from pipelines.pipeline import flight_delay_training_pipeline


def compile_pipeline(output_path: str = "flight_delay_pipeline.json"):
    """
    Compila el pipeline de Kubeflow
    
    Args:
        output_path: Ruta donde guardar el pipeline compilado
    """
    try:
        compiler.Compiler().compile(
            pipeline_func=flight_delay_training_pipeline,
            package_path=output_path
        )
        print(f"✓ Pipeline compilado exitosamente en {output_path}")
    except Exception as e:
        print(f"✗ Error al compilar pipeline: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "flight_delay_pipeline.json"
    compile_pipeline(output)

