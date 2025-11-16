"""
Funciones auxiliares para el procesamiento de datos
"""
import pandas as pd
from typing import List


def get_min_diff(row: pd.Series) -> float:
    """
    Calcula la diferencia mínima entre fechas/horas programadas y reales.
    
    Args:
        row: Fila del DataFrame con información de fechas/horas
        
    Returns:
        Diferencia mínima en minutos
    """
    try:
        # Ajustar según las columnas de tu dataset
        # Ejemplo común: diferencia entre fecha programada y fecha real
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
        # Si las columnas no existen o hay error, retornar 0
        return 0


# Lista de las top 10 features más importantes
# Ajustar según tu análisis de importancia de features
top_10_features: List[str] = [
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

