# Flight Delay Training Pipeline

Pipeline de Vertex AI para entrenamiento de modelo de ML para predicción de retrasos de vuelos.

## 📁 Estructura del Proyecto

```
flight_delay_training/
├── components/              # Componentes reutilizables del pipeline
│   ├── data_preprocessing/  # Componente de preprocesamiento de datos
│   ├── training/            # Componente de entrenamiento
│   └── evaluation/          # Componente de evaluación
├── pipelines/               # Definiciones de pipelines
│   └── pipeline.py         # Pipeline principal
├── scripts/                 # Scripts auxiliares
│   ├── compile_pipeline.py # Compilar pipeline
│   └── submit_pipeline.py  # Enviar pipeline a Vertex AI
├── utils/                   # Utilidades
│   └── config_loader.py    # Cargador de configuración
├── configs/                 # Configuraciones
│   └── config.yaml         # Configuración principal
├── requirements.txt         # Dependencias Python
├── setup.py                # Configuración del paquete
└── README.md               # Este archivo
```

## 🚀 Inicio Rápido

### Prerrequisitos

1. **Google Cloud Platform**
   - Proyecto de GCP configurado
   - Vertex AI API habilitada
   - Cloud Storage bucket creado
   - Autenticación configurada (usar `gcloud auth application-default login`)

2. **Python 3.8+**

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

### Configuración

1. Edita `configs/config.yaml` con tus valores:
   - `project_id`: Tu Project ID de GCP
   - `bucket_name`: Nombre de tu bucket de Cloud Storage
   - `region`: Región donde ejecutar el pipeline
   - Ajusta parámetros del modelo según necesites

2. Prepara tus datos:
   - Sube tu dataset CSV a Cloud Storage
   - Actualiza `input_data_path` en la configuración o al ejecutar el pipeline

### Uso

#### Compilar el pipeline:

```bash
python scripts/compile_pipeline.py
```

Esto generará `flight_delay_pipeline.json` que puedes usar para ejecutar el pipeline.

#### Ejecutar el pipeline:

**Opción 1: Usar el script de envío**
```bash
python scripts/submit_pipeline.py
```

**Opción 2: Usar Python directamente**
```python
from google.cloud import aiplatform
from pipelines.pipeline import flight_delay_training_pipeline
from kfp import compiler

# Compilar
compiler.Compiler().compile(
    pipeline_func=flight_delay_training_pipeline,
    package_path="flight_delay_pipeline.json"
)

# Inicializar Vertex AI
aiplatform.init(project="your-project-id", location="us-central1")

# Ejecutar
job = aiplatform.PipelineJob(
    display_name="Flight Delay Training",
    template_path="flight_delay_pipeline.json",
    pipeline_root="gs://your-bucket/pipeline_root"
)
job.run()
```

## 🔧 Componentes del Pipeline

### 1. Data Preprocessing (`components/data_preprocessing/`)
- Descarga datos desde Cloud Storage
- Limpia y transforma datos
- Divide en conjuntos de entrenamiento y prueba

### 2. Training (`components/training/`)
- Entrena modelo de ML (Random Forest por defecto)
- Guarda modelo en Cloud Storage
- Genera métricas de entrenamiento

### 3. Evaluation (`components/evaluation/`)
- Evalúa modelo con datos de test
- Genera métricas detalladas (accuracy, classification report, confusion matrix)
- Guarda resultados en Cloud Storage

## 📝 Personalización

### Modificar preprocesamiento de datos
Edita `components/data_preprocessing/component.py` para agregar tu lógica específica de limpieza y transformación.

### Cambiar algoritmo de ML
Edita `components/training/component.py` para cambiar el modelo (actualmente usa Random Forest).

### Agregar más componentes
Crea nuevos componentes siguiendo la estructura existente en `components/`.

## 🔍 Monitoreo

Una vez ejecutado, puedes monitorear el pipeline en:
- **Vertex AI Console**: https://console.cloud.google.com/vertex-ai/pipelines
- Revisa los logs de cada componente para debugging

## 📚 Recursos

- [Documentación de Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines)
- [Kubeflow Pipelines SDK](https://www.kubeflow.org/docs/components/pipelines/sdk/python/)
- [Vertex AI Python SDK](https://cloud.google.com/python/docs/reference/aiplatform/latest)

## 📄 Licencia

Este proyecto es un template base. Ajusta según tus necesidades.
