"""Cálculo de prototipos, análisis de concentración y guardado de parámetros de referencia."""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

from config import ARTIFACTS_DIR, BREEDS, DEFAULT_FEATURE_PARAMS, PROTOTYPES_FILE, SIZES
from dataset import extract_dataset_features, load_dogspeak_metadata
from eda_stats import compare_two_groups


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similitud coseno entre dos vectores."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distancia euclidiana L2 entre dos vectores."""
    return float(np.linalg.norm(a - b))


def build_prototypes_and_analyze(
    output_dir: str = ARTIFACTS_DIR,
    prototypes_path: str = PROTOTYPES_FILE,
    **feature_overrides,
) -> dict:
    """
    1. Extrae MFCCs de DogsSpeakTDS.
    2. Estandariza características (z-score) y calcula prototipos por tamaño y por raza.
    3. Analiza la concentración de cada categoría alrededor de su prototipo.
    4. Guarda prototipos y parámetros en un archivo .npz autónomo.
    """
    os.makedirs(output_dir, exist_ok=True)
    params = {**DEFAULT_FEATURE_PARAMS, **feature_overrides}

    df = load_dogspeak_metadata()
    X_raw = extract_dataset_features(df, **params)

    # Estandarización z-score global sobre DogsSpeakTDS
    mu = X_raw.mean(axis=0)
    sigma = X_raw.std(axis=0) + 1e-12
    X_std = (X_raw - mu) / sigma

    # Prototipos de tamaño (Pequeño vs Grande)
    size_prototypes = {
        sz: X_std[df["size"] == sz].mean(axis=0)
        for sz in SIZES
    }

    # Prototipos por raza (5 razas)
    breed_prototypes = {
        br: X_std[df["breed"] == br].mean(axis=0)
        for br in BREEDS
    }

    # Análisis de concentración alrededor de cada prototipo de tamaño
    cos_to_own = np.zeros(len(df))
    dist_to_own = np.zeros(len(df))
    for i, sz in enumerate(df["size"]):
        proto = size_prototypes[sz]
        cos_to_own[i] = cosine_similarity(X_std[i], proto)
        dist_to_own[i] = euclidean_distance(X_std[i], proto)

    df["cos_to_proto"] = cos_to_own
    df["dist_to_proto"] = dist_to_own

    # Pruebas estadísticas de concentración entre Pequeños y Grandes
    cos_peq = df.loc[df["size"] == "Pequeño", "cos_to_proto"].values
    cos_gra = df.loc[df["size"] == "Grande", "cos_to_proto"].values
    res_cos = compare_two_groups(cos_peq, cos_gra, "Pequeño", "Grande")

    dist_peq = df.loc[df["size"] == "Pequeño", "dist_to_proto"].values
    dist_gra = df.loc[df["size"] == "Grande", "dist_to_proto"].values
    res_dist = compare_two_groups(dist_peq, dist_gra, "Pequeño", "Grande")

    print("=== CONCENTRACIÓN ALREDEDOR DEL PROTOTIPO (SIMILITUD COSENO) ===")
    print(f"Pequeño -> media: {cos_peq.mean():.4f}, mediana: {np.median(cos_peq):.4f}, DE: {cos_peq.std():.4f}")
    print(f"Grande  -> media: {cos_gra.mean():.4f}, mediana: {np.median(cos_gra):.4f}, DE: {cos_gra.std():.4f}")
    print(res_cos)

    print("\n=== CONCENTRACIÓN ALREDEDOR DEL PROTOTIPO (DISTANCIA EUCLIDIANA) ===")
    print(f"Pequeño -> media: {dist_peq.mean():.4f}, mediana: {np.median(dist_peq):.4f}, DE: {dist_peq.std():.4f}")
    print(f"Grande  -> media: {dist_gra.mean():.4f}, mediana: {np.median(dist_gra):.4f}, DE: {dist_gra.std():.4f}")
    print(res_dist)

    # Gráficos comparativos de concentración
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    df.boxplot(column="cos_to_proto", by="size", ax=axes[0, 0])
    axes[0, 0].set_title(f"Similitud Coseno al Prototipo ({res_cos['prueba']} p={res_cos['p_valor']:.3f})")
    axes[0, 0].set_ylabel("Similitud coseno")

    axes[0, 1].hist(cos_peq, bins=18, alpha=0.6, label="Pequeño", color="#f4a261", edgecolor="black")
    axes[0, 1].hist(cos_gra, bins=18, alpha=0.6, label="Grande", color="#2a9d8f", edgecolor="black")
    axes[0, 1].set_title("Distribución de Similitud Coseno")
    axes[0, 1].set_xlabel("Similitud coseno")
    axes[0, 1].legend()

    df.boxplot(column="dist_to_proto", by="size", ax=axes[1, 0])
    axes[1, 0].set_title(f"Distancia Euclidiana al Prototipo ({res_dist['prueba']} p={res_dist['p_valor']:.3f})")
    axes[1, 0].set_ylabel("Distancia L2")

    axes[1, 1].hist(dist_peq, bins=18, alpha=0.6, label="Pequeño", color="#f4a261", edgecolor="black")
    axes[1, 1].hist(dist_gra, bins=18, alpha=0.6, label="Grande", color="#2a9d8f", edgecolor="black")
    axes[1, 1].set_title("Distribución de Distancia Euclidiana")
    axes[1, 1].set_xlabel("Distancia L2")
    axes[1, 1].legend()

    plt.suptitle("")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "concentracion_prototipos.png"), dpi=200)
    plt.close(fig)

    # Guardar prototipos y parámetros para inferencia autónoma sin acceder a los audios
    np.savez(
        prototypes_path,
        mu=mu,
        sigma=sigma,
        size_names=np.array(SIZES),
        size_vectors=np.vstack([size_prototypes[s] for s in SIZES]),
        breed_names=np.array(BREEDS),
        breed_vectors=np.vstack([breed_prototypes[b] for b in BREEDS]),
        feature_params_json=np.array(json.dumps(params)),
    )
    print(f"\nPrototipos y parámetros guardados en: {prototypes_path}")

    return {
        "df": df,
        "X_std": X_std,
        "size_prototypes": size_prototypes,
        "breed_prototypes": breed_prototypes,
        "res_cos": res_cos,
        "res_dist": res_dist,
    }


if __name__ == "__main__":
    build_prototypes_and_analyze()
