"""Análisis estadístico descriptivo e inferencial de la base de datos DogsSpeakTDS."""

import os
import wave
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import ARTIFACTS_DIR
from dataset import load_dogspeak_metadata


def get_audio_info(filepath: str) -> tuple[int, float]:
    """Retorna (frecuencia_muestreo, duracion_segundos) de un archivo WAV."""
    with wave.open(filepath, "rb") as w:
        sr = w.getframerate()
        dur = w.getnframes() / float(sr)
    return sr, dur


def compare_two_groups(group_a: np.ndarray, group_b: np.ndarray, name_a: str, name_b: str) -> dict:
    """
    Aplica el protocolo del Lab 3:
    Shapiro-Wilk (normalidad) -> Levene (homocedasticidad) -> t-test o Mann-Whitney U.
    """
    _, p_norm_a = stats.shapiro(group_a)
    _, p_norm_b = stats.shapiro(group_b)
    both_normal = (p_norm_a > 0.05) and (p_norm_b > 0.05)

    _, p_levene = stats.levene(group_a, group_b)
    equal_var = p_levene > 0.05

    if both_normal:
        stat_val, p_val = stats.ttest_ind(group_a, group_b, equal_var=equal_var)
        test_name = "t-test Student" if equal_var else "t-test Welch"
    else:
        stat_val, p_val = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
        test_name = "Mann-Whitney U"

    return {
        "grupo_a": name_a,
        "grupo_b": name_b,
        "shapiro_p_a": p_norm_a,
        "shapiro_p_b": p_norm_b,
        "levene_p": p_levene,
        "prueba": test_name,
        "estadistico": stat_val,
        "p_valor": p_val,
    }


def run_eda(output_dir: str = ARTIFACTS_DIR) -> pd.DataFrame:
    """Ejecuta el análisis descriptivo y las pruebas estadísticas sobre DogsSpeakTDS."""
    os.makedirs(output_dir, exist_ok=True)
    df = load_dogspeak_metadata()

    infos = [get_audio_info(p) for p in df["filepath"]]
    df["sr"] = [x[0] for x in infos]
    df["duration_s"] = [x[1] for x in infos]

    print("=== RESUMEN DESCRIPTIVO DOGSSPEAK-TDS ===")
    print(f"Total grabaciones: {len(df)} | Perros únicos: {df['dog_id'].nunique()}")
    print(f"Frecuencias de muestreo presentes: {df['sr'].unique().tolist()} Hz")
    print(f"Duración (s) -> media: {df['duration_s'].mean():.3f}, DE: {df['duration_s'].std():.3f}, "
          f"mediana: {df['duration_s'].median():.3f}, min: {df['duration_s'].min():.3f}, max: {df['duration_s'].max():.3f}")
    print("\nConteo por raza:\n", df["breed"].value_counts())
    print("\nConteo por tamaño:\n", df["size"].value_counts())
    print("\nConteo por sexo:\n", df["sex"].value_counts())

    # 1. Prueba Chi-cuadrado: dependencia entre raza y sexo
    ct_breed_sex = pd.crosstab(df["breed"], df["sex"])
    chi2, p_chi2, dof, _ = stats.chi2_contingency(ct_breed_sex)
    print("\n=== PRUEBA CHI-CUADRADO: RAZA vs SEXO ===")
    print(ct_breed_sex)
    print(f"chi2 = {chi2:.4f}, gl = {dof}, p-valor = {p_chi2:.4e}")

    # 2. Pruebas adicionales: duración del ladrido según tamaño y según sexo
    dur_peq = df.loc[df["size"] == "Pequeño", "duration_s"].values
    dur_gra = df.loc[df["size"] == "Grande", "duration_s"].values
    res_dur_size = compare_two_groups(dur_peq, dur_gra, "Pequeño", "Grande")
    print("\n=== DURACIÓN DEL LADRIDO: PEQUEÑO vs GRANDE ===")
    print(res_dur_size)

    dur_f = df.loc[df["sex"] == "female", "duration_s"].values
    dur_m = df.loc[df["sex"] == "male", "duration_s"].values
    res_dur_sex = compare_two_groups(dur_f, dur_m, "Hembra", "Macho")
    print("\n=== DURACIÓN DEL LADRIDO: HEMBRA vs MACHO ===")
    print(res_dur_sex)

    # 3. Gráficos descriptivos
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ct_breed_sex.plot(kind="bar", ax=axes[0], color=["#e07a5f", "#3d405b"])
    axes[0].set_title(f"Raza vs Sexo (Chi2 p={p_chi2:.2e})")
    axes[0].set_ylabel("Número de grabaciones")
    axes[0].tick_params(axis="x", rotation=30)

    df.boxplot(column="duration_s", by="size", ax=axes[1])
    axes[1].set_title(f"Duración por Tamaño ({res_dur_size['prueba']} p={res_dur_size['p_valor']:.3f})")
    axes[1].set_ylabel("Duración (s)")

    axes[2].hist(df["duration_s"], bins=20, color="#81b29a", edgecolor="black", alpha=0.85)
    axes[2].set_title("Distribución de duración de ladridos")
    axes[2].set_xlabel("Duración (s)")
    axes[2].set_ylabel("Frecuencia")

    plt.suptitle("")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "eda_dogspeak.png"), dpi=200)
    plt.close(fig)

    return df


if __name__ == "__main__":
    run_eda()
