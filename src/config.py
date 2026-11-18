"""Configuración centralizada del pipeline de análisis y clasificación de ladridos."""

import os

# Rutas base (relativas a la ubicación de este archivo dentro de Proyecto_Final_TDS1/src)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)

METADATA_CSV = os.path.join(PROJECT_DIR, "metadataDogsSpeakTDS.csv")
DOGSPEAK_DIR = os.path.join(PROJECT_DIR, "DogSpeakTDS", "DogSpeak_Seleccion")
OWN_DOGS_DIR = os.path.join(PROJECT_DIR, "Ladridos y fotos perros")
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "resultados")
PROTOTYPES_FILE = os.path.join(ARTIFACTS_DIR, "prototipos_referencia.npz")

# Mapeo de razas de DogsSpeakTDS a tamaño
BREED_TO_SIZE = {
    "chihuahua": "Pequeño",
    "shiba inu": "Pequeño",
    "german shepherd": "Grande",
    "husky": "Grande",
    "pitbull": "Grande",
}

BREEDS = sorted(BREED_TO_SIZE.keys())
SIZES = ["Pequeño", "Grande"]

# Parámetros configurables para la extracción de características (MFCCs)
DEFAULT_FEATURE_PARAMS = {
    "target_sr": 16000,   # Frecuencia de muestreo común (Hz)
    "win_ms": 50.0,       # Duración de la ventana en ms (rango permitido: 20 - 250 ms)
    "overlap": 0.50,      # Solape entre ventanas (rango permitido: 0.10 - 0.50)
    "window_type": "hamming",
    "n_fft": 1024,        # Puntos de la FFT (>= muestras de ventana)
    "n_mels": 40,         # Número de filtros triangulares en escala Mel
    "n_mfcc": 13,         # Número de coeficientes cepstrales (DCT)
    "fmin": 50.0,         # Frecuencia mínima del banco de filtros (Hz)
    "fmax": 8000.0,       # Frecuencia máxima del banco de filtros (Hz)
    "include_c0": False,  # Excluir c0 (energía global) para mayor invariancia al volumen
    "vad_threshold": 0.05,  # Umbral relativo de energía por ventana para descartar silencios (0.0 desactiva)
}
