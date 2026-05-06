import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import CubicSpline, PchipInterpolator, CubicHermiteSpline
import json

DOMAIN_CONFIG = {
    "eye_tracking": {
        "unit": "ms",
        "requires_hz": True,
    },
    "medical": {
        "unit": "ms",
        "requires_hz": True,
    },
    "traffic": {
        "unit": "samples",
        "requires_hz": False,
    },
    "weather": {
        "unit": "samples",
        "requires_hz": False,
    },
    "water": {
        "unit": "samples",
        "requires_hz": False,
    },
    "generic": {
        "unit": "samples",
        "requires_hz": False,
    },
}


def _get_domain_unit(domain: str) -> str:
    if domain not in DOMAIN_CONFIG:
        raise ValueError(
            f"Unbekannte domain: '{domain}'. "
            f"Erlaubt sind: {list(DOMAIN_CONFIG.keys())}"
        )

    return DOMAIN_CONFIG[domain]["unit"]


def _validate_domain_hz(domain: str, hz: float | None) -> None:
    if domain not in DOMAIN_CONFIG:
        raise ValueError(
            f"Unbekannte domain: '{domain}'. "
            f"Erlaubt sind: {list(DOMAIN_CONFIG.keys())}"
        )

    if DOMAIN_CONFIG[domain]["requires_hz"]:
        if hz is None or hz <= 0:
            raise ValueError(f"Für domain='{domain}' muss hz > 0 angegeben werden.")


def _to_samples(value: float, unit: str, hz: float | None = None) -> int:
    if value < 0:
        raise ValueError("Zeit-/Lückenparameter müssen >= 0 sein.")

    if unit == "samples":
        return int(round(value))

    if unit == "ms":
        if hz is None or hz <= 0:
            raise ValueError("Für unit='ms' muss hz > 0 angegeben werden.")
        return int(round(value * hz / 1000.0))

    raise ValueError(f"Unbekannte Einheit: {unit}")


# =========================================================
# 1) NaN-Lücken finden
# =========================================================
def _find_nan_gaps(a: np.ndarray) -> list[tuple[int, int]]:
    isnan = np.isnan(a)
    gaps = []
    i = 0
    n = len(a)

    while i < n:
        if isnan[i]:
            start = i
            while i < n and isnan[i]:
                i += 1
            gaps.append((start, i - 1))
        else:
            i += 1

    return gaps


# =========================================================
# 2) Nahe Lücken optional zusammenführen
# =========================================================
def _merge_close_gaps(
    gaps: list[tuple[int, int]],
    min_valid_run_between: int = 4,
) -> list[tuple[int, int]]:
    if not gaps:
        return []

    merged = [gaps[0]]

    for g0, g1 in gaps[1:]:
        prev0, prev1 = merged[-1]
        valid_run = g0 - prev1 - 1

        if valid_run < min_valid_run_between:
            merged[-1] = (prev0, g1)
        else:
            merged.append((g0, g1))

    return merged


# =========================================================
# 3) Stützpunkte links/rechts sammeln
# =========================================================
def _support_indices(
    z: np.ndarray,
    left_idx: int,
    right_idx: int,
    window: int,
) -> list[int]:
    left = []
    i = left_idx
    while i >= 0 and len(left) < window:
        if np.isfinite(z[i]):
            left.append(i)
        i -= 1
    left.reverse()

    right = []
    i = right_idx
    while i < len(z) and len(right) < window:
        if np.isfinite(z[i]):
            right.append(i)
        i += 1

    return sorted(set(left + right))


# =========================================================
# 4) AIC / BIC
# =========================================================
def _score(rss: float, n: int, k: int, criterion: str) -> float:
    rss = max(float(rss), 1e-18)
    criterion = criterion.lower()

    if criterion == "aic":
        return n * np.log(rss / n) + 2 * k

    return n * np.log(rss / n) + np.log(n) * k


# =========================================================
# 5) Soft-Bound
# =========================================================
def _soft_bound(val: float, lower: float, upper: float) -> float:
    mid = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower)

    if half <= 0:
        return mid

    x = (val - mid) / half
    return mid + half * np.tanh(x)


# =========================================================
# 6) Global markante Extrema finden
# =========================================================
def find_markant_extrema(
    signal: np.ndarray,
    smooth_window_samples: int = 9,
    polyorder: int = 2,
    min_distance_samples: int = 6,
    prominence_factor: float = 0.6,
) -> dict:
    valid_mask = np.isfinite(signal)
    valid_idx = np.where(valid_mask)[0]
    x = signal[valid_mask]

    if len(x) < 10:
        return {
            "valid_indices": valid_idx,
            "smoothed_signal": x.copy(),
            "peaks_global": np.array([], dtype=int),
            "troughs_global": np.array([], dtype=int),
            "extrema_global": np.array([], dtype=int),
        }

    window_length = max(5, int(round(smooth_window_samples)))
    if window_length % 2 == 0:
        window_length += 1

    if window_length >= len(x):
        window_length = len(x) - 1
        if window_length % 2 == 0:
            window_length -= 1

    min_required = polyorder + 2
    if window_length < min_required:
        window_length = polyorder + 3
        if window_length % 2 == 0:
            window_length += 1
        if window_length >= len(x):
            window_length = len(x) - 1
            if window_length % 2 == 0:
                window_length -= 1

    if window_length < 3:
        return {
            "valid_indices": valid_idx,
            "smoothed_signal": x.copy(),
            "peaks_global": np.array([], dtype=int),
            "troughs_global": np.array([], dtype=int),
            "extrema_global": np.array([], dtype=int),
        }

    x_smooth = savgol_filter(x, window_length=window_length, polyorder=polyorder)

    sigma = float(np.std(x_smooth))
    prominence = max(1e-12, prominence_factor * sigma)
    min_distance_samples = max(1, int(round(min_distance_samples)))

    peaks, _ = find_peaks(
        x_smooth,
        prominence=prominence,
        distance=min_distance_samples,
    )

    troughs, _ = find_peaks(
        -x_smooth,
        prominence=prominence,
        distance=min_distance_samples,
    )

    peaks_global = valid_idx[peaks]
    troughs_global = valid_idx[troughs]
    extrema_local = np.sort(np.concatenate([peaks, troughs]))
    extrema_global = valid_idx[extrema_local]

    return {
        "valid_indices": valid_idx,
        "smoothed_signal": x_smooth,
        "peaks_global": peaks_global,
        "troughs_global": troughs_global,
        "extrema_global": extrema_global,
    }


# =========================================================
# 7) Erwartete Struktur in einer Lücke schätzen
# =========================================================
def _estimate_expected_extrema(
    signal: np.ndarray,
    extrema_global: np.ndarray,
    gap_start: int,
    gap_end: int,
    context_samples: int,
) -> dict:
    left_start = max(0, gap_start - context_samples)
    left_end = gap_start - 1

    right_start = gap_end + 1
    right_end = min(len(signal) - 1, gap_end + context_samples)

    left_len = max(0, left_end - left_start + 1)
    right_len = max(0, right_end - right_start + 1)
    context_len = left_len + right_len
    gap_len = gap_end - gap_start + 1

    context_mask = ((extrema_global >= left_start) & (extrema_global <= left_end)) | (
        (extrema_global >= right_start) & (extrema_global <= right_end)
    )
    context_extrema = np.sort(extrema_global[context_mask])

    if context_len <= 0 or len(context_extrema) == 0:
        return {
            "density": 0.0,
            "expected_extrema": 0.0,
            "avg_amplitude": 0.0,
        }

    density = len(context_extrema) / context_len
    expected_extrema = density * gap_len

    amplitudes = []
    for i in range(len(context_extrema) - 1):
        a = context_extrema[i]
        b = context_extrema[i + 1]
        if np.isfinite(signal[a]) and np.isfinite(signal[b]):
            amplitudes.append(abs(signal[b] - signal[a]))

    avg_amplitude = float(np.mean(amplitudes)) if amplitudes else 0.0

    return {
        "density": float(density),
        "expected_extrema": float(expected_extrema),
        "avg_amplitude": avg_amplitude,
    }


# =========================================================
# 8) Linear
# =========================================================
def _linear_fill_gap(
    z: np.ndarray,
    t: np.ndarray,
    g0: int,
    g1: int,
    left_idx: int,
    right_idx: int,
) -> None:
    z_left = z[left_idx]
    z_right = z[right_idx]
    t_left = t[left_idx]
    t_right = t[right_idx]

    if not (np.isfinite(z_left) and np.isfinite(z_right)):
        return

    if t_right == t_left:
        z[g0 : g1 + 1] = z_left
        return

    for i in range(g0, g1 + 1):
        alpha = (t[i] - t_left) / (t_right - t_left)
        z[i] = (1.0 - alpha) * z_left + alpha * z_right


# =========================================================
# 9) Polyfit
# =========================================================
def _polyfit_fill_gap(
    z: np.ndarray,
    t: np.ndarray,
    g0: int,
    g1: int,
    left_idx: int,
    right_idx: int,
    window: int,
    degrees: tuple[int, ...],
    criterion: str,
    boundary_margin: float = 0.15,
    use_soft_bound: bool = True,
) -> None:
    support = _support_indices(z, left_idx, right_idx, window=window)

    if len(support) < 2:
        fill_val = z[left_idx] if np.isfinite(z[left_idx]) else z[right_idx]
        z[g0 : g1 + 1] = fill_val
        return

    tt = t[support]
    zz = z[support]

    if len(zz) < 2:
        fill_val = z[left_idx] if np.isfinite(z[left_idx]) else z[right_idx]
        z[g0 : g1 + 1] = fill_val
        return

    z_min = float(np.min(zz))
    z_max = float(np.max(zz))
    span = z_max - z_min
    margin = boundary_margin * span if span > 0 else 1.0

    lower = z_min - margin
    upper = z_max + margin

    t_center = float(np.mean(tt))
    tt_local = tt - t_center

    best_score = np.inf
    best_coeff = None

    for deg in degrees:
        k = deg + 1
        if len(tt_local) < k:
            continue

        try:
            coeff = np.polyfit(tt_local, zz, deg)
            pred = np.polyval(coeff, tt_local)
            rss = np.sum((zz - pred) ** 2)
            current_score = _score(rss, n=len(tt_local), k=k, criterion=criterion)
        except np.linalg.LinAlgError:
            continue

        if current_score < best_score:
            best_score = current_score
            best_coeff = coeff

    if best_coeff is None:
        fill_val = z[left_idx] if np.isfinite(z[left_idx]) else z[right_idx]
        z[g0 : g1 + 1] = fill_val
        return

    for i in range(g0, g1 + 1):
        val = float(np.polyval(best_coeff, t[i] - t_center))

        if use_soft_bound:
            val = _soft_bound(val, lower, upper)
        else:
            val = float(np.clip(val, lower, upper))

        z[i] = val


# =========================================================
# 9b) Cubic Spline
# =========================================================
def _spline_fill_gap(
    z,
    t,
    g0,
    g1,
    left_idx,
    right_idx,
    window,
    boundary_margin=0.15,
    use_soft_bound=True,
):
    support = _support_indices(z, left_idx, right_idx, window=window)

    if len(support) < 4:
        _linear_fill_gap(z, t, g0, g1, left_idx, right_idx)
        return

    tt = t[support]
    zz = z[support]

    z_min = float(np.min(zz))
    z_max = float(np.max(zz))
    span = z_max - z_min
    margin = boundary_margin * span if span > 0 else 1.0

    lower = z_min - margin
    upper = z_max + margin

    try:
        spline = CubicSpline(tt, zz, bc_type="natural")
    except Exception:
        _linear_fill_gap(z, t, g0, g1, left_idx, right_idx)
        return

    for i in range(g0, g1 + 1):
        val = float(spline(t[i]))

        if use_soft_bound:
            val = _soft_bound(val, lower, upper)
        else:
            val = float(np.clip(val, lower, upper))

        z[i] = val


# =========================================================
# 9c) PCHIP
# =========================================================
def _pchip_fill_gap(
    z: np.ndarray,
    t: np.ndarray,
    g0: int,
    g1: int,
    left_idx: int,
    right_idx: int,
    window: int,
    boundary_margin: float = 0.15,
    use_soft_bound: bool = True,
) -> None:
    support = _support_indices(z, left_idx, right_idx, window=window)

    if len(support) < 2:
        fill_val = z[left_idx] if np.isfinite(z[left_idx]) else z[right_idx]
        z[g0 : g1 + 1] = fill_val
        return

    tt = t[support]
    zz = z[support]

    z_min = float(np.min(zz))
    z_max = float(np.max(zz))
    span = z_max - z_min
    margin = boundary_margin * span if span > 0 else 1.0

    lower = z_min - margin
    upper = z_max + margin

    try:
        pchip = PchipInterpolator(tt, zz)
    except Exception:
        _linear_fill_gap(z, t, g0, g1, left_idx, right_idx)
        return

    for i in range(g0, g1 + 1):
        val = float(pchip(t[i]))

        if use_soft_bound:
            val = _soft_bound(val, lower, upper)
        else:
            val = float(np.clip(val, lower, upper))

        z[i] = val


# =========================================================
# 10) Periodisch (Sinus)
# =========================================================
def _sinus_fill_gap(
    z: np.ndarray,
    t: np.ndarray,
    g0: int,
    g1: int,
    left_idx: int,
    right_idx: int,
    expected_extrema: float,
    avg_amplitude: float,
    sinus_amplitude_factor: float = 0.5,
    boundary_margin: float = 0.15,
) -> None:
    if left_idx < 0 or right_idx >= len(z):
        return

    if not (np.isfinite(z[left_idx]) and np.isfinite(z[right_idx])):
        return

    n = g1 - g0 + 1
    left_val = z[left_idx]
    right_val = z[right_idx]

    baseline = np.linspace(left_val, right_val, n)
    num_cycles = max(0.0, expected_extrema / 2.0)

    if num_cycles < 0.5 or avg_amplitude <= 0:
        z[g0 : g1 + 1] = baseline
        return

    phase = np.linspace(0, num_cycles * 2.0 * np.pi, n)
    oscillation = np.sin(phase)

    amplitude = sinus_amplitude_factor * avg_amplitude
    structured = baseline + amplitude * oscillation

    local_support = []
    for i in range(max(0, left_idx - 10), min(len(z), right_idx + 11)):
        if np.isfinite(z[i]):
            local_support.append(z[i])

    if len(local_support) > 0:
        z_min = float(np.min(local_support))
        z_max = float(np.max(local_support))
        span = z_max - z_min
        margin = boundary_margin * span if span > 0 else 1.0
        lower = z_min - margin
        upper = z_max + margin

        structured = np.array(
            [_soft_bound(val, lower, upper) for val in structured], dtype=float
        )

    z[g0 : g1 + 1] = structured


def _template_fill_gap(
    z: np.ndarray,
    t: np.ndarray,
    g0: int,
    g1: int,
    left_idx: int,
    right_idx: int,
    context_window: int = 300,
    boundary_margin: float = 0.15,
    use_soft_bound: bool = True,
) -> None:
    if left_idx < 0 or right_idx >= len(z):
        return

    if not (np.isfinite(z[left_idx]) and np.isfinite(z[right_idx])):
        return

    n = g1 - g0 + 1
    left_val = z[left_idx]
    right_val = z[right_idx]

    baseline = np.linspace(
        left_val, right_val, n
    )  # baseline = np.linspace(left_val, right_val, n + 2)[1:-1]

    left_start = max(0, left_idx - context_window + 1)
    left_segment = z[left_start : left_idx + 1]

    right_end = min(len(z), right_idx + context_window)
    right_segment = z[right_idx:right_end]

    candidates = []

    for segment in [left_segment, right_segment]:
        segment = segment[np.isfinite(segment)]

        if len(segment) >= n:
            candidates.append(segment[-n:])
        elif len(segment) >= 4:
            old_x = np.linspace(0.0, 1.0, len(segment))
            new_x = np.linspace(0.0, 1.0, n)
            candidates.append(np.interp(new_x, old_x, segment))

    if not candidates:
        z[g0 : g1 + 1] = baseline
        return

    template = np.mean(candidates, axis=0)

    template_trend = np.linspace(template[0], template[-1], n)
    movement = template - template_trend

    u = np.linspace(0.0, 1.0, n)
    envelope = 3 * u**2 - 2 * u**3

    structured = baseline + envelope * movement

    local_support = []
    for i in range(max(0, left_idx - 10), min(len(z), right_idx + 11)):
        if np.isfinite(z[i]):
            local_support.append(z[i])

    if len(local_support) > 0:
        z_min = float(np.min(local_support))
        z_max = float(np.max(local_support))
        span = z_max - z_min
        margin = boundary_margin * span if span > 0 else 1.0

        lower = z_min - margin
        upper = z_max + margin

        structured = np.array(
            [
                (
                    _soft_bound(val, lower, upper)
                    if use_soft_bound
                    else np.clip(val, lower, upper)
                )
                for val in structured
            ],
            dtype=float,
        )

    z[g0 : g1 + 1] = structured


# =========================================================
# 11) Periodisch (STL) - Platzhalter
# =========================================================
def _stl_fill_gap_placeholder(
    z: np.ndarray,
    t: np.ndarray,
    g0: int,
    g1: int,
    left_idx: int,
    right_idx: int,
) -> None:
    _linear_fill_gap(
        z=z,
        t=t,
        g0=g0,
        g1=g1,
        left_idx=left_idx,
        right_idx=right_idx,
    )


# =========================================================
# 11b) Saisonale Rekonstruktion
# =========================================================
def _seasonal_fill_gap(
    z: np.ndarray,
    g0: int,
    g1: int,
    period_samples: int,
    n_periods: int = 3,
) -> bool:
    gap_len = g1 - g0 + 1
    candidates = []

    for k in range(1, n_periods + 5):
        start = g0 - k * period_samples
        end = start + gap_len

        if start < 0:
            break

        segment = z[start:end]

        if len(segment) == gap_len and np.all(np.isfinite(segment)):
            candidates.append(segment.copy())

        if len(candidates) >= n_periods:
            break

    if not candidates:
        return False

    z[g0 : g1 + 1] = np.mean(candidates, axis=0)
    return True


# =========================================================
# 12) Methode wählen
# =========================================================
def _choose_method(
    gap_len_samples: int,
    expected_extrema: float,
    objective: str,
    use_expected_extrema: bool,
    small_gap_samples: int,
    medium_gap_samples: int,
    large_gap_visual_method: str = "sinus",
) -> str:
    if objective not in {"visual_reconstruction", "accurate_imputation"}:
        raise ValueError(
            "objective muss 'visual_reconstruction' oder " "'accurate_imputation' sein."
        )

    if large_gap_visual_method not in {"sinus", "template"}:
        raise ValueError("large_gap_visual_method muss 'sinus' oder 'template' sein.")

    if objective == "visual_reconstruction":
        if gap_len_samples <= small_gap_samples:
            return "linear"

        if gap_len_samples <= medium_gap_samples:
            return "spline"

        return large_gap_visual_method

    if gap_len_samples <= small_gap_samples:
        return "linear"

    if gap_len_samples <= medium_gap_samples:
        return "polyfit"

    return "pchip"


# =========================================================
# 13) Hauptfunktion
# =========================================================
def impute_missing_domain_aware(
    df: pd.DataFrame,
    x_col: str,
    hz: float | None = None,
    domain: str = "eye_tracking",
    objective: str = "visual_reconstruction",
    smooth_window: float = 35.0,
    smooth_polyorder: int = 2,
    min_distance: float = 25.0,
    prominence_factor: float = 0.6,
    context: float = 300.0,
    window: int = 25,
    degrees: tuple[int, ...] = (1, 2, 3),
    criterion: str = "bic",
    boundary_margin: float = 0.15,
    use_soft_bound: bool = True,
    merge_close_gaps: bool = False,
    min_valid_run_between: int = 4,
    sinus_amplitude_factor: float = 0.5,
    use_expected_extrema: bool = False,
    small_gap: float = 50.0,
    medium_gap: float = 150.0,
    period_samples: int | None = None,
    n_periods: int = 3,
    return_log: bool = False,
    log_path: str | None = None,
    large_gap_visual_method: str = "sinus",
) -> pd.DataFrame | tuple[pd.DataFrame, list[dict]]:
    """
    Imputation von NaN-Lücken.

    Die Einheit der Parameter small_gap, medium_gap, context,
    smooth_window und min_distance hängt von der domain ab.

    Domains:
    - eye_tracking -> ms, benötigt hz
    - medical      -> ms, benötigt hz
    - traffic      -> samples
    - weather      -> samples
    - water        -> samples
    - generic      -> samples
    """
    _validate_domain_hz(domain, hz)
    unit = _get_domain_unit(domain)

    if x_col not in df.columns:
        raise ValueError(f"Spalte '{x_col}' nicht im DataFrame vorhanden.")
    if window < 1:
        raise ValueError("window muss >= 1 sein.")
    if not degrees:
        raise ValueError("degrees darf nicht leer sein.")

    if period_samples is not None and period_samples < 1:
        raise ValueError("period_samples muss >= 1 sein.")
    if n_periods < 1:
        raise ValueError("n_periods muss >= 1 sein.")

    small_gap_samples = _to_samples(small_gap, unit, hz)
    medium_gap_samples = _to_samples(medium_gap, unit, hz)
    context_samples = max(1, _to_samples(context, unit, hz))
    smooth_window_samples = max(3, _to_samples(smooth_window, unit, hz))
    min_distance_samples = max(1, _to_samples(min_distance, unit, hz))

    if small_gap_samples > medium_gap_samples:
        raise ValueError("small_gap darf nicht größer als medium_gap sein.")

    out = df.copy()
    z = out[x_col].to_numpy(dtype=float)
    t = np.arange(len(z), dtype=float)

    gaps = _find_nan_gaps(z)

    logs = []

    if merge_close_gaps:
        gaps = _merge_close_gaps(
            gaps,
            min_valid_run_between=min_valid_run_between,
        )

    extrema_info = find_markant_extrema(
        signal=z,
        smooth_window_samples=smooth_window_samples,
        polyorder=smooth_polyorder,
        min_distance_samples=min_distance_samples,
        prominence_factor=prominence_factor,
    )

    extrema_global = extrema_info["extrema_global"]

    for g0, g1 in gaps:
        if not np.any(np.isnan(z[g0 : g1 + 1])):
            continue

        gap_len_samples = g1 - g0 + 1

        left_idx = g0 - 1
        right_idx = g1 + 1

        if left_idx < 0 or right_idx >= len(z):
            if left_idx < 0 and right_idx < len(z) and np.isfinite(z[right_idx]):
                z[g0 : g1 + 1] = z[right_idx]
            elif right_idx >= len(z) and left_idx >= 0 and np.isfinite(z[left_idx]):
                z[g0 : g1 + 1] = z[left_idx]
            continue

        if not (np.isfinite(z[left_idx]) and np.isfinite(z[right_idx])):
            continue

        est = _estimate_expected_extrema(
            signal=z,
            extrema_global=extrema_global,
            gap_start=g0,
            gap_end=g1,
            context_samples=context_samples,
        )

        expected_extrema = est["expected_extrema"]
        avg_amplitude = est["avg_amplitude"]

        if (
            domain in {"traffic", "weather"}
            and objective == "visual_reconstruction"
            and period_samples is not None
        ):
            if gap_len_samples <= small_gap_samples:
                method = "linear"
            elif gap_len_samples <= medium_gap_samples:
                method = "spline"
            else:
                method = "seasonal"
        else:
            method = _choose_method(
                gap_len_samples=gap_len_samples,
                expected_extrema=expected_extrema,
                objective=objective,
                use_expected_extrema=use_expected_extrema,
                small_gap_samples=small_gap_samples,
                medium_gap_samples=medium_gap_samples,
                large_gap_visual_method=large_gap_visual_method,
            )

        logs.append(
            {
                "gap_start": int(g0),
                "gap_end": int(g1),
                "gap_len_samples": int(gap_len_samples),
                "left_idx": int(left_idx),
                "right_idx": int(right_idx),
                "expected_extrema": float(expected_extrema),
                "avg_amplitude": float(avg_amplitude),
                "method": method,
            }
        )

        if method == "linear":
            _linear_fill_gap(
                z=z,
                t=t,
                g0=g0,
                g1=g1,
                left_idx=left_idx,
                right_idx=right_idx,
            )
        elif method == "polyfit":
            _polyfit_fill_gap(
                z=z,
                t=t,
                g0=g0,
                g1=g1,
                left_idx=left_idx,
                right_idx=right_idx,
                window=window,
                degrees=degrees,
                criterion=criterion,
                boundary_margin=boundary_margin,
                use_soft_bound=use_soft_bound,
            )

        elif method == "spline":
            _spline_fill_gap(
                z=z,
                t=t,
                g0=g0,
                g1=g1,
                left_idx=left_idx,
                right_idx=right_idx,
                window=window,
                boundary_margin=boundary_margin,
                use_soft_bound=use_soft_bound,
            )

        elif method == "pchip":
            _pchip_fill_gap(
                z=z,
                t=t,
                g0=g0,
                g1=g1,
                left_idx=left_idx,
                right_idx=right_idx,
                window=window,
                boundary_margin=boundary_margin,
                use_soft_bound=use_soft_bound,
            )

        elif method == "sinus":
            _sinus_fill_gap(
                z=z,
                t=t,
                g0=g0,
                g1=g1,
                left_idx=left_idx,
                right_idx=right_idx,
                expected_extrema=expected_extrema,
                avg_amplitude=avg_amplitude,
                sinus_amplitude_factor=sinus_amplitude_factor,
                boundary_margin=boundary_margin,
            )

        elif method == "template":
            _template_fill_gap(
                z=z,
                t=t,
                g0=g0,
                g1=g1,
                left_idx=left_idx,
                right_idx=right_idx,
                context_window=context_samples,
                boundary_margin=boundary_margin,
                use_soft_bound=use_soft_bound,
            )

        elif method == "seasonal":
            filled = _seasonal_fill_gap(
                z=z,
                g0=g0,
                g1=g1,
                period_samples=period_samples,
                n_periods=n_periods,
            )

            if not filled:
                _spline_fill_gap(
                    z=z,
                    t=t,
                    g0=g0,
                    g1=g1,
                    left_idx=left_idx,
                    right_idx=right_idx,
                    window=window,
                    boundary_margin=boundary_margin,
                    use_soft_bound=use_soft_bound,
                )

        elif method == "stl":
            _stl_fill_gap_placeholder(
                z=z,
                t=t,
                g0=g0,
                g1=g1,
                left_idx=left_idx,
                right_idx=right_idx,
            )
        else:
            raise RuntimeError(f"Unbekannte Methode: {method}")

    out[x_col] = z

    if log_path is not None:
        with open(log_path, "w") as f:
            json.dump(logs, f, indent=2)
    return out
