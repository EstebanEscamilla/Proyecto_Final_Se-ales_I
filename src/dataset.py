"""Carga de metadatos y extracción de características sobre DogsSpeakTDS."""

import os
import numpy as np
import pandas as pd

from config import BREED_TO_SIZE, DOGSPEAK_DIR, METADATA_CSV
from features import extract_features_from_file


def load_dogspeak_metadata(
    csv_path: str = METADATA_CSV,
    audio_root: str = DOGSPEAK_DIR,
) -> pd.DataFrame:
    """Lee metadataDogsSpeakTDS.csv, resuelve rutas de audio y asigna la etiqueta de tamaño."""
    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Indexar todos los .wav en DogSpeak_Seleccion por nombre de archivo
    wav_map = {}
    for dp, _, filenames in os.walk(audio_root):
        for fname in filenames:
            if fname.lower().endswith(".wav"):
                wav_map[fname] = os.path.join(dp, fname)

    df["filepath"] = df["filename"].map(wav_map)
    missing = df["filepath"].isna().sum()
    if missing > 0:
        raise FileNotFoundError(f"No se encontraron {missing} archivos de audio en {audio_root}")

    df["size"] = df["breed"].map(BREED_TO_SIZE)
    if df["size"].isna().any():
        unknown = df.loc[df["size"].isna(), "breed"].unique()
        raise ValueError(f"Razas sin mapeo de tamaño: {unknown}")

    return df


def extract_dataset_features(df: pd.DataFrame, **feature_kwargs) -> np.ndarray:
    """Extrae la matriz de características MFCC (N_audios x n_mfcc) para todo el DataFrame."""
    vectors = [
        extract_features_from_file(path, **feature_kwargs)
        for path in df["filepath"]
    ]
    return np.vstack(vectors)
