"""Extracción de características acústicas basada en bancos de filtros Mel y MFCCs."""

from math import gcd
import numpy as np
from scipy.fft import dct
from scipy.io import wavfile
from scipy.signal import get_window, resample_poly

from config import DEFAULT_FEATURE_PARAMS


def hz_to_mel(freq_hz: np.ndarray | float) -> np.ndarray | float:
    """Convierte frecuencia en Hz a escala Mel (Stevens & Newmann, 1937)."""
    return 2595.0 * np.log10(1.0 + np.asarray(freq_hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    """Convierte escala Mel a frecuencia en Hz."""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def load_and_resample(audio_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Carga un archivo WAV, lo convierte a mono normalizado en [-1, 1] y remuestrea a target_sr."""
    sr, data = wavfile.read(audio_path)

    # Convertir a float64 en rango [-1, 1] según el tipo de dato original
    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        signal = data.astype(np.float64) / max_val
    else:
        signal = data.astype(np.float64)

    # Convertir estéreo/multicanal a mono promediando canales
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    # Eliminar componente DC
    signal = signal - np.mean(signal)

    # Remuestrear si la frecuencia de muestreo difiere de target_sr
    if sr != target_sr:
        divisor = gcd(int(sr), int(target_sr))
        up = int(target_sr) // divisor
        down = int(sr) // divisor
        signal = resample_poly(signal, up, down)

    # Normalizar amplitud pico para invariancia a ganancia de grabación
    peak = np.max(np.abs(signal))
    if peak > 1e-12:
        signal = signal / peak

    return signal, target_sr


def create_mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int = 40,
    fmin: float = 50.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Construye una matriz de banco de filtros triangulares espaciados en escala Mel."""
    if fmax is None:
        fmax = sr / 2.0
    fmax = min(fmax, sr / 2.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Bins de frecuencia correspondientes en la rFFT
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    n_freqs = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_freqs), dtype=np.float64)

    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center > left:
            for k in range(left, min(center, n_freqs)):
                filterbank[m - 1, k] = (k - left) / (center - left)
        if right > center:
            for k in range(max(center, 0), min(right, n_freqs)):
                filterbank[m - 1, k] = (right - k) / (right - center)

    # Normalización por área (Slaney-style) para mantener energía equilibrada por banda
    enorm = 2.0 / np.maximum(hz_points[2 : n_mels + 2] - hz_points[:n_mels], 1e-12)
    filterbank *= enorm[:, np.newaxis]

    return filterbank


def frame_signal(
    signal: np.ndarray,
    sr: int,
    win_ms: float = 50.0,
    overlap: float = 0.50,
    window_type: str = "hamming",
) -> np.ndarray:
    """Divide la señal en ventanas solapadas y aplica la función ventana."""
    if not (20.0 <= win_ms <= 250.0):
        raise ValueError(f"win_ms debe estar entre 20 y 250 ms (recibido: {win_ms})")
    if not (0.10 <= overlap <= 0.50):
        raise ValueError(f"overlap debe estar entre 0.10 y 0.50 (recibido: {overlap})")

    win_len = max(1, int(round((win_ms / 1000.0) * sr)))
    hop_len = max(1, int(round(win_len * (1.0 - overlap))))

    # Rellenar con ceros si el audio es más corto que una ventana
    if len(signal) < win_len:
        signal = np.pad(signal, (0, win_len - len(signal)), mode="constant")

    num_frames = 1 + (len(signal) - win_len) // hop_len
    indices = (
        np.arange(win_len)[np.newaxis, :]
        + np.arange(num_frames)[:, np.newaxis] * hop_len
    )
    frames = signal[indices]

    win = get_window(window_type, win_len, fftbins=True)
    return frames * win[np.newaxis, :]


def extract_mfcc_vector(
    signal: np.ndarray,
    sr: int,
    win_ms: float = DEFAULT_FEATURE_PARAMS["win_ms"],
    overlap: float = DEFAULT_FEATURE_PARAMS["overlap"],
    window_type: str = DEFAULT_FEATURE_PARAMS["window_type"],
    n_fft: int = DEFAULT_FEATURE_PARAMS["n_fft"],
    n_mels: int = DEFAULT_FEATURE_PARAMS["n_mels"],
    n_mfcc: int = DEFAULT_FEATURE_PARAMS["n_mfcc"],
    fmin: float = DEFAULT_FEATURE_PARAMS["fmin"],
    fmax: float = DEFAULT_FEATURE_PARAMS["fmax"],
    include_c0: bool = DEFAULT_FEATURE_PARAMS["include_c0"],
    vad_threshold: float = DEFAULT_FEATURE_PARAMS["vad_threshold"],
) -> np.ndarray:
    """
    Extrae un vector MFCC representativo del audio:
    Enventanado -> (VAD por energía) -> rFFT -> Potencia -> Banco Mel -> Log -> DCT -> Promedio temporal.
    """
    frames = frame_signal(signal, sr, win_ms=win_ms, overlap=overlap, window_type=window_type)

    # Descartar ventanas silenciosas usando energía relativa al pico (si vad_threshold > 0)
    if vad_threshold > 0.0:
        frame_energies = np.sum(frames ** 2, axis=1)
        max_energy = np.max(frame_energies)
        if max_energy > 1e-12:
            active_frames = frames[frame_energies >= vad_threshold * max_energy]
            if len(active_frames) > 0:
                frames = active_frames

    win_len = frames.shape[1]
    fft_size = max(n_fft, 1 << (win_len - 1).bit_length())

    # Espectro de potencia por ventana
    spectrum = np.fft.rfft(frames, n=fft_size, axis=1)
    power_spec = (np.abs(spectrum) ** 2) / fft_size

    # Aplicación del banco de filtros Mel
    mel_fb = create_mel_filterbank(sr=sr, n_fft=fft_size, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_energies = np.dot(power_spec, mel_fb.T)
    log_mel = np.log(np.maximum(mel_energies, 1e-12))

    # Transformada Discreta del Coseno (DCT tipo II) -> MFCCs por ventana
    mfcc_frames = dct(log_mel, type=2, axis=1, norm="ortho")

    start_idx = 0 if include_c0 else 1
    end_idx = start_idx + n_mfcc
    mfcc_selected = mfcc_frames[:, start_idx:end_idx]

    # Promedio temporal sobre las ventanas para obtener un único vector por grabación
    return np.mean(mfcc_selected, axis=0)


def extract_features_from_file(audio_path: str, **kwargs) -> np.ndarray:
    """Carga un archivo de audio (con cualquier fs) y retorna su vector MFCC."""
    params = {**DEFAULT_FEATURE_PARAMS, **kwargs}
    target_sr = params.pop("target_sr")
    signal, sr = load_and_resample(audio_path, target_sr=target_sr)
    return extract_mfcc_vector(signal, sr, **params)
