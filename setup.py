from setuptools import setup, find_packages

setup(
    name="flight_delay_training",
    version="0.1.0",
    description="Pipeline de entrenamiento de modelo ML en Vertex AI",
    packages=find_packages(),
    install_requires=[
        "kfp==2.14.6",
        "google-cloud-aiplatform>=1.100.0",
        "google-cloud-storage==2.14.0",
        "pandas>=2.1.4",
        "scikit-learn>=1.3.2",
        "numpy>=1.26.0",
        "xgboost>=2.0.3",
        "pyyaml>=6.0.1",
        "joblib>=1.3.2",
    ],
    python_requires=">=3.8",
    # Opcional: permite ejecutar scripts como comandos después de pip install -e .
    entry_points={
        "console_scripts": [
            "compile-pipeline=scripts.compile_pipeline:compile_pipeline",
            "submit-pipeline=scripts.submit_pipeline:submit_pipeline",
        ],
    },
)

