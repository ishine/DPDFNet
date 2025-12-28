import argparse
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import librosa
from tflite_runtime.interpreter import Interpreter
from tqdm import tqdm


TFLITE_DIR = Path('./model_zoo/tflite')

# ===== STFT / iSTFT params (as in the snippet) =====
WIN_LEN = 320      # 16 kHz: 320
HOP_SIZE = WIN_LEN // 2  # 50% hop


def vorbis_window(window_len: int) -> np.ndarray:
    window_size_h = window_len / 2
    indices = np.arange(window_len)
    sin = np.sin(0.5 * np.pi * (indices + 0.5) / window_size_h)
    window = np.sin(0.5 * np.pi * sin * sin)
    return window.astype(np.float32)


def get_wnorm(window_len: int, frame_size: int) -> float:
    # window_len - #samples of the window; frame_size - hop size
    return 1.0 / (window_len ** 2 / (2 * frame_size))


# ---------- Pre/Post processing ----------
_WIN = vorbis_window(WIN_LEN)
_WNORM = get_wnorm(WIN_LEN, HOP_SIZE)


def preprocessing(waveform_16k: np.ndarray) -> np.ndarray:
    """
    waveform_16k: 1D float32 numpy array at 16 kHz, mono, range ~[-1,1]
    Returns complex STFT as real/imag split: [B=1, T, F, 2] float32
    """
    # Librosa returns [F, T]; match original by using center=False here
    spec = librosa.stft(
        y=waveform_16k.astype(np.float32, copy=False),
        n_fft=WIN_LEN,
        hop_length=HOP_SIZE,
        win_length=WIN_LEN,
        window=_WIN,
        center=False,
        pad_mode="reflect"
    )  # [F, T] complex64
    spec = (spec.T * _WNORM).astype(np.complex64)  # [T, F]
    spec_ri = np.stack([spec.real, spec.imag], axis=-1).astype(np.float32)  # [T, F, 2]
    return spec_ri[None, ...]  # [1, T, F, 2]


def postprocessing(spec_e: np.ndarray) -> np.ndarray:
    """
    spec_e: [1, T, F, 2] float32
    Returns waveform (1D float32, 16 kHz)
    """
    # Recreate complex STFT with shape [F, T]
    spec_c = spec_e[0].astype(np.float32)  # [T, F, 2]
    spec = (spec_c[..., 0] + 1j * spec_c[..., 1]).T.astype(np.complex64)  # [F, T]

    waveform_e = librosa.istft(
        spec,
        hop_length=HOP_SIZE,
        win_length=WIN_LEN,
        window=_WIN,
        center=True,
        length=None,
    ).astype(np.float32)

    waveform_e = waveform_e / _WNORM
    waveform_e = np.concatenate([waveform_e[WIN_LEN * 2:], np.zeros(WIN_LEN * 2, dtype=np.float32)])
    return waveform_e.astype(np.float32)


# ---------- Audio utilities ----------
def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    # Average channels to mono
    return np.mean(audio, axis=1)


def ensure_16k(waveform: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    if sr == target_sr:
        return waveform.astype(np.float32, copy=False)
    return librosa.resample(waveform.astype(np.float32, copy=False), orig_sr=sr, target_sr=target_sr)


def resample_back(waveform_16k: np.ndarray, target_sr: int) -> np.ndarray:
    if target_sr == 16000:
        return waveform_16k
    return librosa.resample(waveform_16k.astype(np.float32, copy=False), orig_sr=16000, target_sr=target_sr)


def pcm16_safe(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


# ---------- Core processing ----------
def enhance_file(in_path: Path, out_path: Path, model_name: str) -> None:
    # Load audio
    audio, sr_in = sf.read(str(in_path), always_2d=False)
    audio = to_mono(audio)

    # Convert dtypes and resample to 16k for the model
    audio = audio.astype(np.float32, copy=False)
    audio_16k = ensure_16k(audio, sr_in, 16000)

    # Alignment compensation #1
    audio_16k_pad = np.pad(audio_16k, (0, WIN_LEN), mode='constant', constant_values=0)
    
    # STFT to frames (streaming)
    spec = preprocessing(audio_16k_pad)  # [1, T, F, 2]
    num_frames = spec.shape[1]

    # New interpreter per file ensures stateful models (RNN/LSTM) start clean
    interpreter = Interpreter(model_path=str(TFLITE_DIR / (model_name + '.tflite')))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Frame-by-frame inference
    outputs = []

    for t in tqdm(range(num_frames), desc=f"{in_path.name}", unit="frm", leave=False):
        frame = spec[:, t:t + 1]  # [1, 1, F, 2]
        # Some TFLite builds are picky about contiguity/dtype
        frame = np.ascontiguousarray(frame, dtype=np.float32)

        interpreter.set_tensor(input_details[0]["index"], frame)
        interpreter.invoke()
        y = interpreter.get_tensor(output_details[0]["index"])  # expected [1,1,F,2]
        outputs.append(np.ascontiguousarray(y, dtype=np.float32))

    # Concatenate along time dimension
    spec_e = np.concatenate(outputs, axis=1).astype(np.float32)  # [1, T, F, 2]

    # iSTFT to waveform (16 kHz), then back to original SR for saving
    enhanced_16k = postprocessing(spec_e)
    enhanced = resample_back(enhanced_16k, sr_in)

    # Alignment compensation #2
    enhanced = enhanced[:audio.size]

    # Save as 16-bit PCM WAV, mono, original sample rate
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), pcm16_safe(enhanced), sr_in, subtype="PCM_16")


def main():
    parser = argparse.ArgumentParser(description="Enhance WAV files with a DPDFNet TFLite model (streaming).")
    parser.add_argument("--noisy_dir", type=str, required=True, help="Folder with noisy *.wav files (non-recursive).")
    parser.add_argument("--enhanced_dir", type=str, required=True, help="Output folder for enhanced WAVs.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="dpdfnet8",
        choices=["baseline", "dpdfnet2", "dpdfnet4", "dpdfnet8"],
        help=(
            "Name of the model to use. Options: "
            "'baseline', 'dpdfnet2', 'dpdfnet4', 'dpdfnet8'. "
            "Default is 'dpdfnet8'."
        ),
    )
    args = parser.parse_args()

    noisy_dir = Path(args.noisy_dir)
    enhanced_dir = Path(args.enhanced_dir)
    model_name = args.model_name

    if not noisy_dir.is_dir():
        print(f"ERROR: --noisy_dir does not exist or is not a directory: {noisy_dir}", file=sys.stderr)
        sys.exit(1)

    wavs = sorted(p for p in noisy_dir.glob("*.wav") if p.is_file())
    if not wavs:
        print(f"No .wav files found in {noisy_dir} (non-recursive).")
        sys.exit(0)

    print(f"Model: {model_name}")
    print(f"Input : {noisy_dir}")
    print(f"Output: {enhanced_dir}")
    print(f"Found {len(wavs)} file(s). Enhancing...\n")

    for wav in wavs:
        out_path = enhanced_dir / (wav.stem + f'_{model_name}.wav')
        try:
            enhance_file(wav, out_path, model_name)
        except Exception as e:
            print(f"[SKIP] {wav.name} due to error: {e}", file=sys.stderr)

    print("\nProcessing complete. Outputs saved in:", enhanced_dir)


if __name__ == "__main__":
    main()
