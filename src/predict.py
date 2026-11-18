"""Función autónoma de predicción de tamaño y similitud por razas (sin acceso a audios originales)."""

import json
import numpy as np

from config import PROTOTYPES_FILE
from features import extract_features_from_file


def softmax_percentages(similarities: np.ndarray, temperature: float = 0.25) -> np.ndarray:
    """
    Convierte un vector de similitudes en porcentajes normalizados cuya suma es 100 %.
    Se resta el máximo para estabilidad numérica antes de aplicar exp(s / T).
    """
    scaled = (similarities - np.max(similarities)) / temperature
    exps = np.exp(scaled)
    return 100.0 * exps / np.sum(exps)


def predecir_ladrido(
    ruta_audio: str,
    prototypes_path: str = PROTOTYPES_FILE,
    temperature: float = 0.25,
) -> dict:
    """
    Recibe la ruta de una grabación de ladrido (con cualquier frecuencia de muestreo)
    y retorna:
      - 'tamano': predicción ('Grande' o 'Pequeño')
      - 'similitud_tamano': similitud coseno con cada prototipo de tamaño
      - 'porcentajes_razas': diccionario {raza: porcentaje} cuya suma es 100 %
    """
    data = np.load(prototypes_path, allow_pickle=False)
    mu = data["mu"]
    sigma = data["sigma"]
    size_names = data["size_names"].tolist()
    size_vectors = data["size_vectors"]
    breed_names = data["breed_names"].tolist()
    breed_vectors = data["breed_vectors"]
    feature_params = json.loads(str(data["feature_params_json"]))

    # Extraer características del audio (remuestreando automáticamente si fs != target_sr)
    x_raw = extract_features_from_file(ruta_audio, **feature_params)
    x_std = (x_raw - mu) / sigma
    x_norm = np.linalg.norm(x_std) + 1e-12

    # Similitud coseno contra prototipos de tamaño
    size_sims = {
        name: float(np.dot(x_std, vec) / (x_norm * (np.linalg.norm(vec) + 1e-12)))
        for name, vec in zip(size_names, size_vectors)
    }
    predicted_size = max(size_sims, key=size_sims.get)

    # Similitud coseno contra prototipos de raza -> porcentajes normalizados (suma = 100 %)
    breed_sims = np.array([
        float(np.dot(x_std, vec) / (x_norm * (np.linalg.norm(vec) + 1e-12)))
        for vec in breed_vectors
    ])
    breed_pcts = softmax_percentages(breed_sims, temperature=temperature)

    return {
        "tamano": predicted_size,
        "similitud_tamano": size_sims,
        "porcentajes_razas": {
            name: round(float(pct), 2) for name, pct in zip(breed_names, breed_pcts)
        },
    }
