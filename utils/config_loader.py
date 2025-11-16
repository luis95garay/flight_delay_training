"""
Utilidad para cargar configuración desde archivo YAML
"""
import yaml
from typing import Dict, Any


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Carga configuración desde archivo YAML
    
    Args:
        config_path: Ruta al archivo de configuración
    
    Returns:
        Diccionario con la configuración
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

