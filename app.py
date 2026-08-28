#!/usr/bin/env python3
from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass, asdict
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from scipy.fft import dct
from scipy.signal import get_window
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

@dataclass
class ExtractionConfig:
    segment_tier: str = "Segment"
    vowel_tier: str = "Vowel"
    word_tier: str = "Word"
    stress_tier: str = "Stress"
    phonemic_nasality_tier: str = "PhonemicNasality"
    naf_training_tier: str = "NAFTraining"
    speaker_separator: str = "_"

    n_time_points: int = 31
    edge_exclusion_pct: float = 5.0
    window_ms: float = 30.0

    f0_floor: float = 60.0
    f0_ceiling: float = 500.0

    formant_ceiling: float = 5500.0
    max_number_of_formants: int = 5
    formant_window_length: float = 0.025
    formant_preemphasis_from: float = 50.0

    p0_min_hz: float = 200.0
    p0_max_hz: float = 500.0
    p1_min_hz: float = 850.0
    p1_max_hz: float = 1050.0

    spectral_max_hz: float = 5360.0
    nasal_low_max_hz: float = 320.0
    tilt_min_hz: float = 500.0
    tilt_max_hz: float = 5000.0

    n_mfcc: int = 13
    n_mel_filters: int = 26
    mel_min_hz: float = 50.0
    mel_max_hz: float = 8000.0


@dataclass
class Interval:
    start: float
    end: float
    label: str


# ============================================================
# TEXTGRID HELPERS
# ============================================================

def safe_float(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def get_intervals(tg, tier_name: str, required: bool = False) -> list[Interval]:
    try:
        # Praat exposes tier names and interval counts, but not a portable
        # TextGrid "Get tier number" command through every parselmouth build.
        # Resolve the requested tier explicitly instead.
        n_tiers = int(call(tg, "Get number of tiers"))
        idx = next(
            i for i in range(1, n_tiers + 1)
            if str(call(tg, "Get tier name", i)) == tier_name
        )
        n = int(call(tg, "Get number of intervals", idx))
    except Exception:
        if required:
            raise RuntimeError(f"Required TextGrid tier not found: {tier_name}")
        return []

    out = []
    for i in range(1, n + 1):
        label = str(call(tg, "Get label of interval", idx, i)).strip()
        if not label:
            continue
        start = float(call(tg, "Get start time of interval", idx, i))
        end = float(call(tg, "Get end time of interval", idx, i))
        out.append(Interval(start, end, label))
    return out


def containing(intervals: list[Interval], t: float):
    for x in intervals:
        if x.start <= t <= x.end:
            return x
    return None


def nearest_segment_index(segments: list[Interval], t: float):
    if not segments:
        return None

    for i, seg in enumerate(segments):
        if seg.start <= t <= seg.end:
            return i

    mids = np.array([(s.start + s.end) / 2 for s in segments])
    return int(np.argmin(np.abs(mids - t)))


def segment_context(segments: list[Interval], vowel: Interval) -> dict:
    mid = (vowel.start + vowel.end) / 2
    idx = nearest_segment_index(segments, mid)

    if idx is None:
        return {
            "prev2_segment": None,
            "prev_segment": None,
            "segment": None,
            "next_segment": None,
            "next2_segment": None,
        }

    def lab(j):
        return segments[j].label if 0 <= j < len(segments) else None

    return {
        "prev2_segment": lab(idx - 2),
        "prev_segment": lab(idx - 1),
        "segment": lab(idx),
        "next_segment": lab(idx + 1),
        "next2_segment": lab(idx + 2),
    }


# ============================================================
# SPECTRAL HELPERS
# ============================================================

def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def power_spectrum(samples: np.ndarray, sr: float):
    x = np.asarray(samples, dtype=float)
    if len(x) < 8:
        return np.array([]), np.array([])

    x = x - np.mean(x)
    x *= get_window("hann", len(x), fftbins=True)

    nfft = max(2048, int(2 ** np.ceil(np.log2(len(x)))))
    X = np.fft.rfft(x, n=nfft)
    power = np.abs(X) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    return freqs, power


def db(power):
    return 10.0 * np.log10(np.maximum(power, 1e-20))


def local_peak(freqs, power, lo, hi):
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return np.nan, np.nan

    f = freqs[mask]
    p = power[mask]
    j = int(np.argmax(p))
    return float(f[j]), float(db(p[j]))


def harmonic_amplitudes(freqs, power, f0, n=20):
    """Measure local spectral maxima anchored to integer F0 harmonics."""
    out = {}
    for k in range(1, n + 1):
        if not np.isfinite(f0) or f0 <= 0:
            out[f"H{k}_Hz"] = np.nan
            out[f"H{k}_dB"] = np.nan
            continue

        target = k * f0
        # Keep neighbouring harmonic searches separate. A narrow search also
        # prevents a strong formant peak from being recycled as many harmonics.
        halfwidth = max(1.5 * (freqs[1] - freqs[0]), 0.12 * f0)
        fr, amp = local_peak(
            freqs, power,
            target - halfwidth,
            target + halfwidth
        )
        out[f"H{k}_Hz"] = fr
        out[f"H{k}_dB"] = amp
    return out


def harmonic_near_formant(harmonics, formant_hz, bandwidth_hz):
    """Return the harmonic closest to a formant, i.e. A1/A2/A3 by definition."""
    if not np.isfinite(formant_hz):
        return np.nan, np.nan, np.nan

    candidates = []
    for k in range(1, 21):
        hz = harmonics.get(f"H{k}_Hz", np.nan)
        amp = harmonics.get(f"H{k}_dB", np.nan)
        if np.isfinite(hz) and np.isfinite(amp):
            candidates.append((abs(hz - formant_hz), k, hz, amp))

    if not candidates:
        return np.nan, np.nan, np.nan

    _, k, hz, amp = min(candidates)
    tolerance = max(100.0, safe_float(bandwidth_hz) / 2.0)
    if abs(hz - formant_hz) > tolerance:
        return np.nan, np.nan, np.nan
    return float(k), float(hz), float(amp)


def harmonic_in_range(harmonics, lo, hi, allowed_k=None):
    """Return the highest-amplitude F0-anchored harmonic in a target region."""
    ks = allowed_k if allowed_k is not None else range(1, 21)
    candidates = []
    for k in ks:
        hz = harmonics.get(f"H{k}_Hz", np.nan)
        amp = harmonics.get(f"H{k}_dB", np.nan)
        if lo <= hz <= hi and np.isfinite(amp):
            candidates.append((amp, k, hz))
    if not candidates:
        return np.nan, np.nan, np.nan
    amp, k, hz = max(candidates)
    return float(k), float(hz), float(amp)


def harmonic_prominence(harmonics, k):
    if not np.isfinite(k):
        return np.nan
    k = int(k)
    target = harmonics.get(f"H{k}_dB", np.nan)
    neighbours = [
        harmonics.get(f"H{j}_dB", np.nan)
        for j in (k - 1, k + 1)
        if 1 <= j <= 20
    ]
    neighbours = [x for x in neighbours if np.isfinite(x)]
    if not np.isfinite(target) or not neighbours:
        return np.nan
    return float(target - np.mean(neighbours))


def spectral_moments(freqs, power, max_hz):
    mask = (freqs > 0) & (freqs <= max_hz)
    f = freqs[mask]
    w = power[mask]

    if len(f) == 0 or np.sum(w) <= 0:
        return (np.nan,) * 4

    w = w / np.sum(w)
    cog = np.sum(w * f)
    var = np.sum(w * (f - cog) ** 2)
    sd = np.sqrt(max(var, 0))

    if sd <= 0:
        return float(cog), float(sd), np.nan, np.nan

    skew = np.sum(w * ((f - cog) / sd) ** 3)
    kurt = np.sum(w * ((f - cog) / sd) ** 4) - 3.0
    return float(cog), float(sd), float(skew), float(kurt)


def band_energy(freqs, power, lo, hi):
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return np.nan
    return float(np.sum(power[mask]))


def spectral_tilt(freqs, power, lo, hi):
    mask = (freqs >= lo) & (freqs <= hi) & (power > 0)
    if np.sum(mask) < 20:
        return np.nan

    x = freqs[mask] / 1000.0
    y = db(power[mask])

    keep = y > np.nanpercentile(y, 20)
    if np.sum(keep) < 10:
        return np.nan

    slope, _ = np.polyfit(x[keep], y[keep], 1)
    return float(slope)


def mfccs(freqs, power, sr, cfg: ExtractionConfig):
    max_hz = min(cfg.mel_max_hz, sr / 2.0)
    min_hz = min(cfg.mel_min_hz, max_hz - 1.0)

    mel_pts = np.linspace(
        hz_to_mel(min_hz),
        hz_to_mel(max_hz),
        cfg.n_mel_filters + 2
    )
    hz_pts = mel_to_hz(mel_pts)

    energies = []
    for i in range(1, len(hz_pts) - 1):
        left, center, right = hz_pts[i - 1:i + 2]
        tri = np.zeros_like(freqs)

        a = (freqs >= left) & (freqs <= center)
        b = (freqs > center) & (freqs <= right)

        if center > left:
            tri[a] = (freqs[a] - left) / (center - left)
        if right > center:
            tri[b] = (right - freqs[b]) / (right - center)

        energies.append(np.sum(power * tri))

    loge = np.log(np.maximum(np.asarray(energies), 1e-20))
    coeff = dct(loge, type=2, norm="ortho")[:cfg.n_mfcc]

    return {
        f"MFCC_{i:02d}": float(v)
        for i, v in enumerate(coeff)
    }


# ============================================================
# PRAAT MEASURES
# ============================================================

def get_formants(sound: parselmouth.Sound, cfg: ExtractionConfig):
    try:
        formant = sound.to_formant_burg(
            time_step=0.0,
            max_number_of_formants=cfg.max_number_of_formants,
            maximum_formant=cfg.formant_ceiling,
            window_length=cfg.formant_window_length,
            pre_emphasis_from=cfg.formant_preemphasis_from,
        )

        t = (sound.xmin + sound.xmax) / 2
        vals = {}

        for n in (1, 2, 3):
            vals[f"F{n}_Hz"] = safe_float(
                call(formant, "Get value at time", n, t, "Hertz", "Linear")
            )
            vals[f"B{n}_Hz"] = safe_float(
                call(formant, "Get bandwidth at time", n, t, "Hertz", "Linear")
            )

        return vals

    except Exception:
        return {
            f"{p}{n}_Hz": np.nan
            for n in (1, 2, 3)
            for p in ("F", "B")
        }


def get_f0(
    sound: parselmouth.Sound,
    cfg: ExtractionConfig,
    t: float | None = None,
    pitch=None,
):
    try:
        if pitch is None:
            pitch = sound.to_pitch_ac(
                time_step=0.0,
                pitch_floor=cfg.f0_floor,
                pitch_ceiling=cfg.f0_ceiling,
            )
        if t is None:
            t = (sound.xmin + sound.xmax) / 2
        return safe_float(
            call(pitch, "Get value at time", t, "Hertz", "Linear")
        )
    except Exception:
        return np.nan


def get_intensity(
    sound: parselmouth.Sound,
    cfg: ExtractionConfig,
    t: float | None = None,
    intensity=None,
):
    try:
        if intensity is None:
            intensity = sound.to_intensity(
                minimum_pitch=cfg.f0_floor,
                time_step=0.0,
                subtract_mean=True
            )
        if t is None:
            t = (sound.xmin + sound.xmax) / 2
        return safe_float(
            call(intensity, "Get value at time", t, "Cubic")
        )
    except Exception:
        return np.nan


# ============================================================
# ONE WINDOW
# ============================================================

def extract_window_features(
    sound: parselmouth.Sound,
    center_s: float,
    cfg: ExtractionConfig,
    pitch=None,
    intensity=None,
) -> dict:

    half = cfg.window_ms / 2000.0
    lo = max(sound.xmin, center_s - half)
    hi = min(sound.xmax, center_s + half)

    if hi - lo < 0.010:
        return {}

    win = sound.extract_part(
        from_time=lo,
        to_time=hi,
        window_shape=parselmouth.WindowShape.RECTANGULAR,
        preserve_times=False
    )

    # Pitch and intensity need more than a 15–30 ms spectral slice,
    # especially for low-F0 speakers. Measure them in the original signal.
    f0 = get_f0(sound, cfg, center_s, pitch=pitch)
    intensity = get_intensity(sound, cfg, center_s, intensity=intensity)
    form = get_formants(win, cfg)

    samples = win.values[0]
    sr = float(win.sampling_frequency)
    freqs, power = power_spectrum(samples, sr)

    if len(freqs) == 0:
        return {}

    harmonics = harmonic_amplitudes(freqs, power, f0, n=20)

    A = {}
    for n in (1, 2, 3):
        ak, af, ad = harmonic_near_formant(
            harmonics,
            form[f"F{n}_Hz"],
            form[f"B{n}_Hz"],
        )
        A[f"A{n}_harmonic"] = ak
        A[f"A{n}_Hz"] = af
        A[f"A{n}_dB"] = ad

    p0k, p0f, p0db = harmonic_in_range(
        harmonics,
        cfg.p0_min_hz,
        cfg.p0_max_hz,
        allowed_k=(1, 2),
    )

    p1k, p1f, p1db = harmonic_in_range(
        harmonics,
        cfg.p1_min_hz,
        cfg.p1_max_hz,
    )

    p0_prominence = harmonic_prominence(harmonics, p0k)
    p1_prominence = harmonic_prominence(harmonics, p1k)

    cog, ssd, skew, kurt = spectral_moments(
        freqs, power, cfg.spectral_max_hz
    )

    low_e = band_energy(
        freqs, power,
        0.0,
        cfg.nasal_low_max_hz
    )

    high_e = band_energy(
        freqs, power,
        cfg.nasal_low_max_hz,
        cfg.spectral_max_hz
    )

    ratio = (
        10.0 * np.log10(low_e / high_e)
        if np.isfinite(low_e)
        and np.isfinite(high_e)
        and low_e > 0
        and high_e > 0
        else np.nan
    )

    x = np.asarray(samples, dtype=float)
    rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else np.nan
    peak_abs = float(np.max(np.abs(x))) if len(x) else np.nan

    result = {
        "window_start_s": lo,
        "window_end_s": hi,

        "f0_Hz": f0,
        "intensity_dB": intensity,
        "digital_RMS": rms,
        "digital_peak_abs": peak_abs,

        **form,
        **harmonics,
        **A,

        "P0_Hz": p0f,
        "P0_dB": p0db,
        "P0_harmonic": p0k,
        "P0_prominence_dB": p0_prominence,
        "P1_Hz": p1f,
        "P1_dB": p1db,
        "P1_harmonic": p1k,
        "P1_prominence_dB": p1_prominence,

        "A1_P0_dB": (
            A["A1_dB"] - p0db
            if np.isfinite(A["A1_dB"]) and np.isfinite(p0db)
            else np.nan
        ),

        "A1_P1_dB": (
            A["A1_dB"] - p1db
            if np.isfinite(A["A1_dB"]) and np.isfinite(p1db)
            else np.nan
        ),

        "A3_P0_dB": (
            A["A3_dB"] - p0db
            if np.isfinite(A["A3_dB"]) and np.isfinite(p0db)
            else np.nan
        ),

        "H1_H2_dB": (
            harmonics["H1_dB"] - harmonics["H2_dB"]
            if np.isfinite(harmonics["H1_dB"])
            and np.isfinite(harmonics["H2_dB"])
            else np.nan
        ),

        "spectral_tilt_dB_per_kHz": spectral_tilt(
            freqs,
            power,
            cfg.tilt_min_hz,
            cfg.tilt_max_hz
        ),

        "spectral_CoG_Hz": cog,
        "spectral_SD_Hz": ssd,
        "spectral_skew": skew,
        "spectral_kurtosis": kurt,

        "energy_low_nasal": low_e,
        "energy_high": high_e,
        "nasal_murmur_ratio_dB": ratio,

        **mfccs(freqs, power, sr, cfg),
    }

    result["f0_valid"] = bool(np.isfinite(f0))
    result["formant_valid"] = bool(
        all(np.isfinite(form[f"F{i}_Hz"]) for i in (1, 2, 3))
    )
    result["P0_candidate_valid"] = bool(
        np.isfinite(p0db) and np.isfinite(p0_prominence)
    )
    result["P1_candidate_valid"] = bool(
        np.isfinite(p1db) and np.isfinite(p1_prominence)
    )
    result["harmonic_valid"] = bool(np.isfinite(A["A1_dB"]))

    return result


# ============================================================
# ONE FILE PAIR
# ============================================================

def extract_pair(
    wav_path: Path,
    textgrid_path: Path,
    cfg: ExtractionConfig,
    speaker: str | None = None,
):
    sound = parselmouth.Sound(str(wav_path))
    tg = parselmouth.Data.read(str(textgrid_path))
    try:
        pitch_track = sound.to_pitch_ac(
            time_step=0.0,
            pitch_floor=cfg.f0_floor,
            pitch_ceiling=cfg.f0_ceiling,
        )
        intensity_track = sound.to_intensity(
            minimum_pitch=cfg.f0_floor,
            time_step=0.0,
            subtract_mean=True,
        )
    except Exception:
        pitch_track = None
        intensity_track = None

    vowels = get_intervals(
        tg,
        cfg.vowel_tier,
        required=True
    )

    segments = get_intervals(
        tg,
        cfg.segment_tier,
        required=True
    )

    words = get_intervals(tg, cfg.word_tier)
    stresses = get_intervals(tg, cfg.stress_tier)
    phon_nas = get_intervals(
        tg,
        cfg.phonemic_nasality_tier
    )
    naf_training = get_intervals(tg, cfg.naf_training_tier)

    if speaker is None:
        speaker = (
            wav_path.stem.split(cfg.speaker_separator)[0]
            if cfg.speaker_separator
            else wav_path.stem
        )

    low = cfg.edge_exclusion_pct / 100.0
    high = 1.0 - low

    half_window_s = cfg.window_ms / 2000.0

    token_rows = []
    trajectory_rows = []

    for j, v in enumerate(vowels, start=1):
        duration = v.end - v.start

        if duration <= 0:
            continue

        mid = (v.start + v.end) / 2
        token_id = f"{wav_path.stem}__V{j:05d}"

        # A spectral window centred near a vowel edge can otherwise include
        # neighbouring consonantal material.  The user-selected edge exclusion
        # is therefore a *minimum*: widen it dynamically by half the analysis
        # window so every retained window lies fully inside the vowel interval.
        safe_low = max(low, half_window_s / duration)
        safe_high = min(high, 1.0 - half_window_s / duration)
        trajectory_extractable = safe_low < safe_high
        time_norm = (
            np.linspace(safe_low, safe_high, cfg.n_time_points)
            if trajectory_extractable
            else np.array([])
        )

        word = containing(words, mid)
        stress = containing(stresses, mid)
        pn = containing(phon_nas, mid)
        training = containing(naf_training, mid)

        phonemic_nasality_01 = np.nan
        if pn is not None:
            label = pn.label.strip()
            if label in {"0", "1"}:
                phonemic_nasality_01 = int(label)

        naf_training_01 = np.nan
        if training is not None and training.label.strip() in {"0", "1"}:
            naf_training_01 = int(training.label.strip())

        token_rows.append({
            "token_id": token_id,
            "speaker": speaker,
            "file": wav_path.name,
            "word": word.label if word else None,
            "vowel": v.label,
            "start_s": v.start,
            "end_s": v.end,
            "duration_ms": duration * 1000.0,
            "analysis_time_low_pct": safe_low * 100.0,
            "analysis_time_high_pct": safe_high * 100.0,
            "trajectory_extractable": trajectory_extractable,
            "trajectory_status": (
                "ok"
                if trajectory_extractable
                else "vowel_shorter_than_analysis_window"
            ),
            "stress": stress.label if stress else None,
            "phonemic_nasality_01": phonemic_nasality_01,
            "naf_training_01": naf_training_01,
            **segment_context(segments, v),
        })

        for tn in time_norm:
            center = v.start + tn * duration

            row = {
                "token_id": token_id,
                "speaker": speaker,
                "file": wav_path.name,
                "vowel": v.label,
                "time_norm": float(tn),
                "time_pct": float(tn * 100.0),
                "time_from_vowel_onset_ms": float(
                    tn * duration * 1000.0
                ),
                "absolute_time_s": float(center),
                "vowel_duration_ms": duration * 1000.0,
                "phonemic_nasality_01": phonemic_nasality_01,
                "naf_training_01": naf_training_01,
            }

            row.update(
                extract_window_features(
                    sound,
                    center,
                    cfg,
                    pitch=pitch_track,
                    intensity=intensity_track,
                )
            )

            trajectory_rows.append(row)

    return pd.DataFrame(token_rows), pd.DataFrame(trajectory_rows)


# ============================================================
# EXCEL
# ============================================================

def qc_summary(tokens_df, traj_df):
    rows = []

    rows.append({
        "metric": "n_tokens",
        "value": len(tokens_df)
    })

    rows.append({
        "metric": "n_trajectory_rows",
        "value": len(traj_df)
    })

    for col in [
        "f0_valid",
        "formant_valid",
        "P0_candidate_valid",
        "P1_candidate_valid",
        "harmonic_valid"
    ]:
        if col in traj_df.columns:
            rows.append({
                "metric": f"{col}_proportion",
                "value": pd.to_numeric(
                    traj_df[col],
                    errors="coerce"
                ).mean()
            })

    return pd.DataFrame(rows)


def naf_feature_columns(df: pd.DataFrame) -> list[str]:
    """Carignan-inspired multi-feature set available in this extractor."""
    fixed = [
        "F1_Hz", "F2_Hz", "F3_Hz", "B1_Hz", "B2_Hz", "B3_Hz",
        "A1_dB", "A2_dB", "A3_dB", "P0_dB", "P1_dB",
        "A1_P0_dB", "A1_P1_dB", "A3_P0_dB", "H1_H2_dB",
        "spectral_CoG_Hz", "nasal_murmur_ratio_dB",
    ]
    mfcc = sorted(c for c in df.columns if c.startswith("MFCC_"))
    return [c for c in fixed + mfcc if c in df.columns]


def add_naf_scores(trajectories: pd.DataFrame):
    """Fit a final speaker-wise PCA/linear NAF model from labelled fillers.

    `NAFTraining` must label balanced oral and nasal training vowels as 0/1.
    A token-stratified 75/25 holdout is reported as a diagnostic; the exported
    `naf_score` is then fit on all available labelled filler rows.
    """
    out = trajectories.copy()
    out["naf_score"] = np.nan
    out["naf_status"] = "no_training_label"
    summary = []
    features = naf_feature_columns(out)

    if not features or "naf_training_01" not in out:
        return out, pd.DataFrame(summary)

    rng = np.random.default_rng(20260828)
    for speaker, idx in out.groupby("speaker", dropna=False).groups.items():
        idx = np.asarray(list(idx))
        sub = out.loc[idx]
        labels = pd.to_numeric(sub["naf_training_01"], errors="coerce")
        labelled = labels.isin([0, 1])
        class_token_counts = {
            label: sub.loc[labels == label, "token_id"].nunique()
            for label in (0, 1)
        }
        if (
            labelled.sum() < 40
            or labels[labelled].nunique() != 2
            or min(class_token_counts.values()) < 4
        ):
            out.loc[idx, "naf_status"] = "need_more_0_and_1_training_rows"
            summary.append({
                "speaker": speaker,
                "status": "need_more_0_and_1_training_rows",
                "n_training_rows": int(labelled.sum()),
                "n_oral_training_tokens": class_token_counts[0],
                "n_nasal_training_tokens": class_token_counts[1],
            })
            continue

        X = sub[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        finite_share = np.isfinite(X).mean(axis=0)
        keep = finite_share >= 0.90
        X = X[:, keep]
        used = np.asarray(features)[keep]
        if X.shape[1] < 3:
            out.loc[idx, "naf_status"] = "too_many_missing_features"
            summary.append({
                "speaker": speaker,
                "status": "too_many_missing_features",
                "n_training_rows": int(labelled.sum()),
            })
            continue

        med = np.nanmedian(X, axis=0)
        X[~np.isfinite(X)] = np.take(med, np.where(~np.isfinite(X))[1])
        center = X.mean(axis=0)
        scale = X.std(axis=0, ddof=0)
        scale[scale == 0] = 1.0
        Xz = (X - center) / scale
        _, singular, vt = np.linalg.svd(Xz, full_matrices=False)
        rank = int(np.sum(singular > 1e-8))
        Z = Xz @ vt[:rank].T

        labelled_tokens = sub.loc[labelled, "token_id"].unique()
        train_tokens = set()
        for label in (0, 1):
            class_tokens = sub.loc[labels == label, "token_id"].unique()
            rng.shuffle(class_tokens)
            n_train_tokens = max(1, int(np.floor(0.75 * len(class_tokens))))
            train_tokens.update(class_tokens[:n_train_tokens])
        train = labelled.to_numpy() & sub["token_id"].isin(train_tokens).to_numpy()
        test = labelled.to_numpy() & ~sub["token_id"].isin(train_tokens).to_numpy()
        y = labels.to_numpy(float)

        beta_cv, *_ = np.linalg.lstsq(
            np.column_stack([np.ones(train.sum()), Z[train]]), y[train], rcond=None
        )
        test_pred = np.column_stack([np.ones(test.sum()), Z[test]]) @ beta_cv
        test_rmse = float(np.sqrt(np.mean((test_pred - y[test]) ** 2))) if test.any() else np.nan

        beta_final, *_ = np.linalg.lstsq(
            np.column_stack([np.ones(labelled.sum()), Z[labelled]]), y[labelled], rcond=None
        )
        out.loc[idx, "naf_score"] = np.column_stack([np.ones(len(idx)), Z]) @ beta_final
        out.loc[idx, "naf_status"] = "ok"
        summary.append({
            "speaker": speaker,
            "status": "ok",
            "n_training_rows": int(labelled.sum()),
                "n_training_tokens": int(len(labelled_tokens)),
                "n_oral_training_tokens": class_token_counts[0],
                "n_nasal_training_tokens": class_token_counts[1],
            "n_features": int(len(used)),
            "n_pcs": rank,
            "holdout_rmse_01": test_rmse,
        })

    return out, pd.DataFrame(summary)


def results_to_excel_bytes(tokens, trajectories, failures, cfg, naf_summary=None):
    output = BytesIO()

    config_df = pd.DataFrame(
        list(asdict(cfg).items()),
        columns=["parameter", "value"]
    )

    notes = pd.DataFrame({
        "Notes": [
            "PhonemicNasality 0/1 is stored as metadata only.",
            "It is not used in acoustic extraction.",
            "NAFTraining 0/1 labels optional oral/nasal filler tokens for speaker-wise NAF fitting.",
            "For category-blind fPCA, exclude phonemic_nasality_01 from the feature matrix.",
            "P0/P1 are automatic candidate detections and require pilot QC.",
            "Trajectory windows are constrained to remain fully inside each vowel.",
            "Tokens shorter than the analysis window have no trajectory rows; inspect trajectory_status in Tokens.",
            "No speaker normalization or z-scoring is performed here."
        ]
    })

    qc = qc_summary(tokens, trajectories)

    EXCEL_MAX = 1_048_000

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        tokens.to_excel(
            writer,
            sheet_name="Tokens",
            index=False
        )

        if len(trajectories) <= EXCEL_MAX:
            trajectories.to_excel(
                writer,
                sheet_name="Trajectories",
                index=False
            )
        else:
            n_chunks = math.ceil(
                len(trajectories) / EXCEL_MAX
            )
            for i in range(n_chunks):
                chunk = trajectories.iloc[
                    i * EXCEL_MAX:
                    (i + 1) * EXCEL_MAX
                ]
                chunk.to_excel(
                    writer,
                    sheet_name=f"Trajectories_{i+1}",
                    index=False
                )

        qc.to_excel(
            writer,
            sheet_name="QC_Summary",
            index=False
        )

        config_df.to_excel(
            writer,
            sheet_name="Config",
            index=False
        )

        failures.to_excel(
            writer,
            sheet_name="Failures",
            index=False
        )

        if naf_summary is not None and not naf_summary.empty:
            naf_summary.to_excel(
                writer,
                sheet_name="NAF_QC",
                index=False
            )

        notes.to_excel(
            writer,
            sheet_name="Notes",
            index=False
        )

    output.seek(0)
    return output.getvalue()


# ============================================================
# STREAMLIT HELPERS
# ============================================================

def save_uploaded_file(uploaded, directory: Path):
    path = directory / uploaded.name
    path.write_bytes(uploaded.getbuffer())
    return path


def pair_uploads(wav_uploads, tg_uploads):
    wav_map = {
        Path(x.name).stem.lower(): x
        for x in wav_uploads
    }

    tg_map = {
        Path(x.name).stem.lower(): x
        for x in tg_uploads
    }

    common = sorted(set(wav_map) & set(tg_map))
    missing_tg = sorted(set(wav_map) - set(tg_map))
    missing_wav = sorted(set(tg_map) - set(wav_map))

    pairs = [
        (wav_map[k], tg_map[k])
        for k in common
    ]

    return pairs, missing_tg, missing_wav


# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(
    page_title="Breton Nasality Acoustic Extractor",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Breton Nasality Acoustic Extractor")
st.caption(
    "WAV + TextGrid → time-varying vowel acoustics → Excel workbook"
)

with st.expander("TextGrid structure", expanded=False):
    st.markdown(
        """
**Required tiers**
- `Segment`
- `Vowel`

**Recommended / optional**
- `Word`
- `Stress`
- `PhonemicNasality`
- `NAFTraining` (optional; labels oral/nasal training fillers for NAF)

`Vowel` should contain vowel quality only, e.g. `a`, `e`, `ɛ`, `i`, `ɔ`, `u`.

`PhonemicNasality` may contain:
- `0` = phonemically oral
- `1` = phonemically nasal

The 0/1 category is stored only as metadata and is **not used for acoustic extraction**.
For speaker-wise NAF fitting, use `NAFTraining`: `0` for oral training fillers and `1` for nasal training fillers.
"""
    )

st.subheader("1. Upload files")

left, right = st.columns(2)

with left:
    wav_uploads = st.file_uploader(
        "WAV files",
        type=["wav"],
        accept_multiple_files=True
    )

with right:
    tg_uploads = st.file_uploader(
        "TextGrid files",
        type=["TextGrid", "textgrid"],
        accept_multiple_files=True
    )

st.subheader("2. Analysis settings")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Trajectory")
    n_time_points = st.slider(
        "Trajectory points",
        min_value=11,
        max_value=61,
        value=31,
        step=2
    )

    edge_exclusion_pct = st.slider(
        "Exclude vowel edges (%)",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=1.0
    )

    window_ms = st.slider(
        "Analysis window (ms)",
        min_value=15.0,
        max_value=60.0,
        value=30.0,
        step=1.0
    )

    st.markdown("#### F0")
    f0_range = st.slider(
        "F0 range (Hz)",
        min_value=40,
        max_value=700,
        value=(60, 500),
        step=5
    )

with col2:
    st.markdown("#### Formants")

    formant_ceiling = st.slider(
        "Formant ceiling (Hz)",
        min_value=3500,
        max_value=7500,
        value=5500,
        step=100
    )

    max_number_of_formants = st.slider(
        "Maximum number of formants",
        min_value=4,
        max_value=7,
        value=5,
        step=1
    )

    formant_window_ms = st.slider(
        "Formant window (ms)",
        min_value=15.0,
        max_value=40.0,
        value=25.0,
        step=1.0
    )

    formant_preemph = st.slider(
        "Pre-emphasis from (Hz)",
        min_value=20,
        max_value=200,
        value=50,
        step=5
    )

with col3:
    st.markdown("#### Nasal resonance search")

    p0_range = st.slider(
        "P0 candidate range (Hz)",
        min_value=50,
        max_value=1000,
        value=(200, 500),
        step=10
    )

    p1_range = st.slider(
        "P1 candidate range (Hz)",
        min_value=300,
        max_value=2000,
        value=(850, 1050),
        step=10
    )

    st.markdown("#### Spectrum")

    spectral_max_hz = st.slider(
        "Spectral maximum (Hz)",
        min_value=3000,
        max_value=10000,
        value=5360,
        step=100
    )

    nasal_low_max_hz = st.slider(
        "Low nasal-energy cutoff (Hz)",
        min_value=150,
        max_value=700,
        value=320,
        step=10
    )

st.markdown("#### Advanced spectral / MFCC settings")

a1, a2, a3 = st.columns(3)

with a1:
    tilt_range = st.slider(
        "Spectral tilt range (Hz)",
        min_value=100,
        max_value=8000,
        value=(500, 5000),
        step=100
    )

with a2:
    mel_range = st.slider(
        "MFCC frequency range (Hz)",
        min_value=20,
        max_value=10000,
        value=(50, 8000),
        step=10
    )

with a3:
    n_mfcc = st.slider(
        "Number of MFCCs",
        min_value=8,
        max_value=20,
        value=13,
        step=1
    )

    n_mel_filters = st.slider(
        "Mel filters",
        min_value=16,
        max_value=40,
        value=26,
        step=1
    )

st.markdown("#### TextGrid tier names")

t1, t2, t3, t4, t5, t6 = st.columns(6)

with t1:
    segment_tier = st.text_input(
        "Segment tier",
        value="Segment"
    )

with t2:
    vowel_tier = st.text_input(
        "Vowel tier",
        value="Vowel"
    )

with t3:
    word_tier = st.text_input(
        "Word tier",
        value="Word"
    )

with t4:
    stress_tier = st.text_input(
        "Stress tier",
        value="Stress"
    )

with t5:
    phonemic_nasality_tier = st.text_input(
        "Phonemic nasality tier",
        value="PhonemicNasality"
    )

with t6:
    naf_training_tier = st.text_input(
        "NAF training tier",
        value="NAFTraining"
    )

speaker_separator = st.text_input(
    "Speaker ID separator in WAV filename",
    value="_",
    help="The filename prefix before this separator is exported as speaker ID. Leave empty to use the whole filename."
)

run_naf = st.checkbox(
    "Fit speaker-wise NAF from NAFTraining 0/1 fillers",
    value=False,
    help="Fits PCA + numeric linear regression separately for each speaker, then exports naf_score and a token-held-out diagnostic."
)


cfg = ExtractionConfig(
    segment_tier=segment_tier,
    vowel_tier=vowel_tier,
    word_tier=word_tier,
    stress_tier=stress_tier,
    phonemic_nasality_tier=phonemic_nasality_tier,
    naf_training_tier=naf_training_tier,
    speaker_separator=speaker_separator,

    n_time_points=n_time_points,
    edge_exclusion_pct=edge_exclusion_pct,
    window_ms=window_ms,

    f0_floor=float(f0_range[0]),
    f0_ceiling=float(f0_range[1]),

    formant_ceiling=float(formant_ceiling),
    max_number_of_formants=max_number_of_formants,
    formant_window_length=float(formant_window_ms) / 1000.0,
    formant_preemphasis_from=float(formant_preemph),

    p0_min_hz=float(p0_range[0]),
    p0_max_hz=float(p0_range[1]),
    p1_min_hz=float(p1_range[0]),
    p1_max_hz=float(p1_range[1]),

    spectral_max_hz=float(spectral_max_hz),
    nasal_low_max_hz=float(nasal_low_max_hz),
    tilt_min_hz=float(tilt_range[0]),
    tilt_max_hz=float(tilt_range[1]),

    n_mfcc=n_mfcc,
    n_mel_filters=n_mel_filters,
    mel_min_hz=float(mel_range[0]),
    mel_max_hz=float(mel_range[1]),
)

st.subheader("3. Extract")

if wav_uploads and tg_uploads:
    pairs, missing_tg, missing_wav = pair_uploads(
        wav_uploads,
        tg_uploads
    )

    st.write(f"Matched pairs: **{len(pairs)}**")

    if missing_tg:
        st.warning(
            "WAV files without matching TextGrid: "
            + ", ".join(missing_tg)
        )

    if missing_wav:
        st.warning(
            "TextGrid files without matching WAV: "
            + ", ".join(missing_wav)
        )
else:
    pairs = []

if st.button(
    "🚀 Extract acoustic trajectories",
    type="primary",
    disabled=not bool(pairs)
):
    all_tokens = []
    all_traj = []
    failures = []

    progress = st.progress(0)
    status = st.empty()

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        for i, (wav_up, tg_up) in enumerate(pairs, start=1):
            status.write(
                f"Processing {wav_up.name} ({i}/{len(pairs)})"
            )

            wav_path = save_uploaded_file(
                wav_up,
                tmpdir
            )

            tg_path = save_uploaded_file(
                tg_up,
                tmpdir
            )

            try:
                tok, traj = extract_pair(
                    wav_path,
                    tg_path,
                    cfg=cfg
                )

                all_tokens.append(tok)
                all_traj.append(traj)

            except Exception as e:
                failures.append({
                    "wav": wav_up.name,
                    "textgrid": tg_up.name,
                    "error": repr(e)
                })

            progress.progress(
                i / len(pairs)
            )

    tokens_df = (
        pd.concat(
            all_tokens,
            ignore_index=True
        )
        if all_tokens
        else pd.DataFrame()
    )

    traj_df = (
        pd.concat(
            all_traj,
            ignore_index=True
        )
        if all_traj
        else pd.DataFrame()
    )

    failures_df = pd.DataFrame(failures)

    naf_summary_df = pd.DataFrame()
    if run_naf and not traj_df.empty:
        traj_df, naf_summary_df = add_naf_scores(traj_df)

    if tokens_df.empty:
        st.error("No vowel tokens were extracted. See the processing error below.")
    else:
        st.success(
            f"Done — {len(tokens_df):,} vowel tokens, "
            f"{len(traj_df):,} trajectory rows."
        )

    if not tokens_df.empty:
        st.markdown("#### Token preview")
        st.dataframe(
            tokens_df.head(30),
            use_container_width=True
        )

    if not traj_df.empty:
        st.markdown("#### Trajectory preview")
        preview_cols = [
            c for c in [
                "token_id",
                "vowel",
                "time_pct",
                "A1_P0_dB",
                "A1_P1_dB",
                "B1_Hz",
                "spectral_tilt_dB_per_kHz",
                "P0_candidate_valid",
                "P0_prominence_dB",
                "naf_score",
                "naf_status",
                "F1_Hz",
                "F2_Hz",
                "MFCC_00",
                "MFCC_01"
            ]
            if c in traj_df.columns
        ]

        st.dataframe(
            traj_df[preview_cols].head(50),
            use_container_width=True
        )

    xlsx = results_to_excel_bytes(
        tokens_df,
        traj_df,
        failures_df,
        cfg,
        naf_summary=naf_summary_df,
    )

    st.download_button(
        "⬇️ Download Excel workbook",
        data=xlsx,
        file_name="breton_nasality_acoustic_trajectories.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    if not failures_df.empty:
        st.error(
            f"{len(failures_df)} file pair(s) failed. "
            "The exact error is shown below and is also saved in the Failures sheet."
        )
        st.dataframe(failures_df, use_container_width=True)

st.divider()
st.caption(
    "Automatic P0/P1 values are candidate detections. "
    "Pilot QC is strongly recommended before inferential analysis."
)
