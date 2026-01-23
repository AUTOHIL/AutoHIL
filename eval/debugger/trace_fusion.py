#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, math, os
from typing import List, Optional, Tuple, Dict, Set
import numpy as np

# Non-interactive backend for saving images
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================== Default configuration (built-in parameters) ========================

DEFAULT_CONFIG = {
    # --- Methodology ---
    "mode": "value_dtw",      # ddtw | value_dtw | combo
    "alpha": 0.3,             # weight when mode="combo"
    "time_base": "relative",  # absolute | relative | progress

    # --- Resampling & smoothing ---
    "n_points": 1200,         # number of resampling points
    "trend_smooth": 51,       # trend smoothing window (1 = more spikes. 201 = more overall trend)

    # --- DTW params ---
    "band_frac": 0.0,         # DTW fractional band width
    "tau_allow_ms": 0.5,      # DTW absolute time band width (ms)

    # --- Terminal value / steady-state detection (Modified) ---
    # <<< Key modification: now only use this window size to take the mean >>>
    "steady_win": 180,        # window length (points) used at the end to compute the mean
    
    # --- (Code 1) Rigid-score params ---
    "rigid_gamma": 0.5,       # Gamma for rigid RMSE
    
    # --- (Fusion) FUSED_SCORE weights ---
    "w_shape_dtw": 0.3,       # weight: elastic shape (DTW/DDTW)
    "w_rigid_rmse": 0.4,      # weight: rigid difference (RMSE+Corr)
    "w_steady": 0.3,          # weight: terminal value difference
    
    # --- Visualization ---
    "dpi": 180,
    "img_prefix": "test",     # image filename prefix
}
# =================================================================


# ======================== IO: Read CSV and extract signal ========================
def read_trace_csv(
    path: str,
    symbol: Optional[str] = None,
    can_id: Optional[str] = None,
    type_filter: str = "TRACE",
) -> Tuple[np.ndarray, np.ndarray]:
    """Read CSV - logic unchanged"""
    if (symbol is None) and (can_id is None):
        raise ValueError("Please provide either `symbol` or `can_id`.")

    times: List[float] = []
    vals: List[float] = []

    with open(path, "r", newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        if header is None:
            raise ValueError(f"Empty CSV: {path}")

        cols = [h.strip() for h in header]
        def idx(name: str) -> int:
            try:
                return cols.index(name)
            except ValueError:
                for i, c in enumerate(cols):
                    if c.replace(" ", "").lower() == name.replace(" ", "").lower():
                        return i
                raise

        i_time = idx("Time(s)")
        i_type = idx("Type")
        i_sym  = idx("Symbol/CAN_ID")
        i_val  = idx("Value/CAN_Data")

        for row in rdr:
            if not row or len(row) <= max(i_time, i_type, i_sym, i_val):
                continue
            t = row[i_time].strip()
            typ = row[i_type].strip()
            sym = row[i_sym].strip()
            val = row[i_val].strip()

            if type_filter and typ != type_filter:
                continue
            if symbol is not None and sym != symbol:
                continue
            if can_id is not None and sym != can_id:
                continue

            try:
                tf = float(t)
                vf = float(val)
            except Exception:
                continue

            times.append(tf)
            vals.append(vf)

    if not times:
        raise ValueError(f"No data found for {symbol or can_id}")

    T = np.asarray(times, dtype=float)
    X = np.asarray(vals, dtype=float)
    o = np.argsort(T)
    return T[o], X[o]

def list_symbols(path: str, type_filter: str = "TRACE") -> Set[str]:
    syms = set()
    with open(path, "r", newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        if header is None:
            return syms
        cols = [h.strip() for h in header]
        try:
            i_type = cols.index("Type")
            i_sym  = cols.index("Symbol/CAN_ID")
        except ValueError:
            return syms
        for row in rdr:
            if not row or len(row) <= max(i_type, i_sym):
                continue
            if type_filter and row[i_type].strip() != type_filter:
                continue
            syms.add(row[i_sym].strip())
    return syms

# ======================== Preprocessing & features ========================
def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k is None or k <= 1:
        return x
    w = np.ones(int(k)) / int(k)
    return np.convolve(x, w, mode="same")

def z_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < eps:
        return np.zeros_like(x)
    return (x - mu) / (sd + eps)

def common_time_resample(
    t1: Optional[np.ndarray], x1: np.ndarray,
    t2: Optional[np.ndarray], x2: np.ndarray,
    n_points: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if t1 is None:
        t1 = np.linspace(0.0, 1.0, len(x1))
    if t2 is None:
        t2 = np.linspace(0.0, 1.0, len(x2))

    t_min = max(np.min(t1), np.min(t2))
    t_max = min(np.max(t1), np.max(t2))
    if t_max <= t_min:
        M = min(len(x1), len(x2), n_points)
        T = np.linspace(0.0, 1.0, M)
        X1r = np.interp(T, np.linspace(0, 1, len(x1)), x1)
        X2r = np.interp(T, np.linspace(0, 1, len(x2)), x2)
        return T, X1r, X2r

    T = np.linspace(t_min, t_max, n_points)
    X1r = np.interp(T, t1, x1)
    X2r = np.interp(T, t2, x2)
    return T, X1r, X2r

def derivative_centered(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    d = np.empty_like(x)
    if len(x) >= 2:
        d[1:-1] = (x[2:] - x[:-2]) * 0.5
        d[0] = x[1] - x[0]
        d[-1] = x[-1] - x[-2]
    else:
        d[:] = 0.0
    return d

# ======================== DTW ========================
def dtw_distance_features(
    A: np.ndarray, B: np.ndarray,
    band_frac: Optional[float] = 0.1,
    w_samples: Optional[int] = None
) -> float:
    n, k = A.shape
    m, kb = B.shape
    assert k == kb, "Feature dims mismatch."

    if w_samples is None:
        w = int(np.floor(float(band_frac) * max(n, m)))
    else:
        w = int(w_samples)

    w = max(w, abs(n - m)) 

    INF = 1e18
    D = np.full((n + 1, m + 1), INF, dtype=float)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        j0 = max(1, i - w)
        j1 = min(m, i + w)
        ai = A[i - 1]
        for j in range(j0, j1 + 1):
            diff = ai - B[j - 1]
            cost = float(np.dot(diff, diff))
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    return float(math.sqrt(D[n, m]))

# ======================== Terminal value / end-state extraction (REPLACED) ========================

def compute_terminal_state(
    t: np.ndarray, x: np.ndarray,
    win_size: int = 50
) -> Dict:
    """
    Force-extract the state of the last win_size points as the "steady/end value".
    No longer performs slope or fluctuation detection, ensuring it always returns the mean of the last segment.
    """
    n = len(x)
    # Ensure the window does not go out of bounds
    win_size = max(1, min(win_size, n))
    
    # Lock the last segment interval
    i1 = n
    i0 = n - win_size
    
    # Compute the mean of the interval
    tail_segment = x[i0:i1]
    val = float(np.mean(tail_segment))
    
    # Return a structure compatible with the previous steady_state, so plotting functions keep working
    return dict(
        i0=i0,
        i1=i1,
        value=val,
        t_start=float(t[i0])
    )

# ======================== Main pipeline ========================

def compare_from_csv(
    csv1: str, csv2: str,
    symbol: Optional[str] = None,
    can_id: Optional[str] = None
) -> Dict:
    
    config = DEFAULT_CONFIG
    mode = config["mode"]
    alpha = config["alpha"]
    n_points = config["n_points"]
    band_frac = config["band_frac"]
    tau_allow_ms = config["tau_allow_ms"]
    time_base = config["time_base"]
    trend_smooth = config["trend_smooth"]
    
    # <<< Key modification: get terminal window size >>>
    steady_win = config["steady_win"]
    
    rigid_gamma = config["rigid_gamma"]
    w_shape_dtw = config["w_shape_dtw"]
    w_rigid_rmse = config["w_rigid_rmse"]
    w_steady = config["w_steady"]

    # 1. Read
    t1_raw, x1_raw = read_trace_csv(csv1, symbol=symbol, can_id=can_id)
    t2_raw, x2_raw = read_trace_csv(csv2, symbol=symbol, can_id=can_id)

    if time_base == "relative":
        t1_use = t1_raw - t1_raw[0]
        t2_use = t2_raw - t2_raw[0]
    elif time_base == "progress":
        t1_use = np.linspace(0.0, 1.0, len(x1_raw))
        t2_use = np.linspace(0.0, 1.0, len(x2_raw))
    else:  
        t1_use, t2_use = t1_raw, t2_raw

    # 2. Resample (raw data)
    T, X1r, X2r = common_time_resample(t1_use, x1_raw, t2_use, x2_raw, n_points=n_points)
    L = len(T)

    # 3. Rigid Score - based on raw data
    X1r_norm = z_norm(X1r) 
    X2r_norm = z_norm(X2r)
    
    if np.std(X1r_norm) < 1e-8 or np.std(X2r_norm) < 1e-8:
        rigid_corr = 0.0
    else:
        rigid_corr = np.corrcoef(X1r_norm, X2r_norm)[0, 1]
        if np.isnan(rigid_corr):
            rigid_corr = 0.0

    rigid_score_corr = (rigid_corr + 1.0) / 2.0
    rigid_rmse = np.sqrt(np.mean((X1r_norm - X2r_norm) ** 2))
    rigid_score_rmse = np.exp(-rigid_rmse / rigid_gamma)
    rigid_trend_score = 0.5 * rigid_score_corr + 0.5 * rigid_score_rmse

    # 4. Shape Score - based on smoothed data
    k_trend = int(trend_smooth)
    X1_trend = moving_average(X1r, k_trend)
    X2_trend = moving_average(X2r, k_trend)

    X1z = z_norm(X1_trend)
    X2z = z_norm(X2_trend)
    
    if mode == "value_dtw":
        A = X1z[:, None]; B = X2z[:, None]
    elif mode == "ddtw":
        d1 = derivative_centered(X1z)
        d2 = derivative_centered(X2z)
        A = d1[:, None]; B = d2[:, None]
    elif mode == "combo":
        d1 = derivative_centered(X1z)
        d2 = derivative_centered(X2z)
        a = float(np.clip(alpha, 0.0, 1.0))
        A = np.stack([math.sqrt(a) * X1z, math.sqrt(1 - a) * d1], axis=1)
        B = np.stack([math.sqrt(a) * X2z, math.sqrt(1 - a) * d2], axis=1)
    else:
        raise ValueError("mode error")

    w_samples = None
    if tau_allow_ms is not None and len(T) >= 2:
        dt_eff_ms = (T[1] - T[0]) * 1000.0
        if dt_eff_ms > 0:
            w_samples = int(math.ceil(float(tau_allow_ms) / dt_eff_ms))
            w_samples = max(w_samples, abs(len(A) - len(B)))
    else:
        dt_eff_ms = (T[1] - T[0]) * 1000.0 if len(T) >= 2 else float("nan")

    D = dtw_distance_features(A, B, band_frac=band_frac, w_samples=w_samples)

    if L > 0 and np.isfinite(D):
        dtw_rms = D / math.sqrt(L)
        shape_trend_score = float(1.0 / (1.0 + dtw_rms))
    else:
        dtw_rms = float("nan")
        shape_trend_score = 0.0

    # 5. Steady Score (changed to: End State Score)
    # <<< Key modification: call compute_terminal_state instead of detect_steady_state >>>
    st1 = compute_terminal_state(T, X1_trend, win_size=steady_win)
    st2 = compute_terminal_state(T, X2_trend, win_size=steady_win)

    v1, v2 = st1["value"], st2["value"]
    xmin = float(min(X1_trend.min(), X2_trend.min()))
    xmax = float(max(X1_trend.max(), X2_trend.max()))
    dyn = max(xmax - xmin, 1e-9)
    
    # Compute terminal value difference score
    steady_score = float(max(0.0, 1.0 - abs(v1 - v2) / dyn))

    # 6. Fused Score
    w_sum_fused = max(w_shape_dtw + w_rigid_rmse + w_steady, 1e-9)
    w_s_dtw_n = w_shape_dtw / w_sum_fused
    w_r_rmse_n = w_rigid_rmse / w_sum_fused
    w_s_n = w_steady / w_sum_fused

    fused_score = float(
        w_s_dtw_n * shape_trend_score + 
        w_r_rmse_n * rigid_trend_score + 
        w_s_n * steady_score
    )

    # Auxiliary data
    end1 = float(np.mean(X1_trend[-steady_win:]))
    end2 = float(np.mean(X2_trend[-steady_win:]))
    d1_vis = derivative_centered(X1z)
    d2_vis = derivative_centered(X2z)

    return dict(
        mode=mode, alpha=alpha,
        shape_trend_score=shape_trend_score,
        dtw_distance=D,
        dtw_rms=dtw_rms,
        steady_score=steady_score,
        
        rigid_corr = rigid_corr,
        rigid_rmse = rigid_rmse,
        rigid_trend_score = rigid_trend_score,
        fused_score = fused_score,
        fused_weights = dict(shape=w_s_dtw_n, rigid=w_r_rmse_n, steady=w_s_n),
        
        resampled_time=T,
        x1_resampled=X1r,
        x2_resampled=X2r,
        x1_trend=X1_trend,
        x2_trend=X2_trend,
        trend_val_1=X1z,
        trend_val_2=X2z,
        trend_deriv_1=d1_vis,
        trend_deriv_2=d2_vis,
        trend_smooth_win=k_trend,
        steady_1=st1, steady_2=st2,
        end_mean_1=end1, end_mean_2=end2,
        symbol=symbol, can_id=can_id,
        rel_time_1=(t1_raw - t1_raw[0]),
        rel_time_2=(t2_raw - t2_raw[0]),
        raw_value_1=x1_raw, raw_value_2=x2_raw,
        time_base=time_base,
        dt_eff_ms=dt_eff_ms,
        w_samples=w_samples,
        band_frac=band_frac
    )

# ======================== Visualization ========================
def _xlabel_for_timebase(time_base: str) -> str:
    if time_base == "progress":
        return "Progress (0..1)"
    elif time_base == "relative":
        return "Relative Time (s)"
    else:
        return "Time (s)"

def plot_resampled_overlay(res: Dict, out_path: str, dpi: int = 180,
                           title: str = "ECU Trace (Resampled Overlay)"):
    T = res["resampled_time"]
    x1 = res["x1_resampled"]
    x2 = res["x2_resampled"]
    st1 = res["steady_1"]; st2 = res["steady_2"]

    plt.figure(figsize=(11,5.6))
    plt.plot(T, x1, label="Run#1 (resampled)")
    plt.plot(T, x2, label="Run#2 (resampled)", alpha=0.9)

    for st in (st1, st2):
        i0, i1 = st["i0"], st["i1"]
        if 0 <= i0 < i1 <= len(T):
            plt.axvspan(T[i0], T[i1-1], alpha=0.12)
    plt.axhline(res["end_mean_1"], ls="--", alpha=0.4)
    plt.axhline(res["end_mean_2"], ls="--", alpha=0.4)

    meta = f"mode={res['mode']}"
    if res["mode"] == "combo":
        meta += f", alpha={res['alpha']:.2f}"
    if res.get("w_samples") is not None:
        meta += f", w={res['w_samples']} (~{res['dt_eff_ms']:.3f}ms/pt)"
    else:
        meta += f", band={res['band_frac']:.3f}"

    plt.title(f"{title}")
    
    plt.xlabel(_xlabel_for_timebase(res.get("time_base","absolute")))
    plt.ylabel("Value (resampled)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()

def plot_trend_and_slope(res: Dict, out_path: str, dpi: int = 180,
                         title: str = "Trend (z-norm) and Local Slope"):
    T = res["resampled_time"]
    v1 = res["trend_val_1"]
    v2 = res["trend_val_2"]
    d1 = res["trend_deriv_1"]
    d2 = res["trend_deriv_2"]
    k_trend = res.get("trend_smooth_win", None)

    fig, axes = plt.subplots(2, 1, figsize=(11,6.0), sharex=True)
    ax0, ax1 = axes

    ax0.plot(T, v1, label="Run#1 trend (z-norm)")
    ax0.plot(T, v2, label="Run#2 trend (z-norm)", alpha=0.9)
    ax0.set_ylabel("Trend (z-norm)")
    
    if k_trend is not None and k_trend > 1:
        ax0.set_title(f"Smoothed Trend (window={k_trend} pts) - Used by ShapeScore")
    else:
        ax0.set_title(f"Original Trend (No Smoothing) - Used by ShapeScore")
        
    ax0.grid(True); ax0.legend()

    ax1.plot(T, d1, label="Run#1 d/dt(trend)", lw=1.0)
    ax1.plot(T, d2, label="Run#2 d/dt(trend)", lw=1.0, alpha=0.9)
    ax1.set_xlabel(_xlabel_for_timebase(res.get("time_base","absolute")))
    ax1.set_ylabel("Local slope")
    ax1.grid(True); ax1.legend()

    fig.suptitle(f"{title} (mode={res['mode']})", fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_side_by_side_relative(res: Dict, out_path: str, dpi: int = 180,
                               title: str = "ECU Trace (Relative Time, Side-by-Side)"):
    t1, x1 = res["rel_time_1"], res["raw_value_1"]
    t2, x2 = res["rel_time_2"], res["raw_value_2"]

    fig, axes = plt.subplots(1, 2, figsize=(12,5.0), sharey=False)
    ax1, ax2 = axes

    ax1.plot(t1, x1, label="Run#1", lw=1.2)
    ax1.set_title("Run#1 (relative time)")
    ax1.set_xlabel("Relative Time (s)")
    ax1.set_ylabel("Value")
    ax1.grid(True); ax1.legend()

    ax2.plot(t2, x2, label="Run#2", lw=1.2)
    ax2.set_title("Run#2 (relative time)")
    ax2.set_xlabel("Relative Time (s)")
    ax2.grid(True); ax2.legend()

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ======================== CLI ========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv1", required=True)
    ap.add_argument("--csv2", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--symbol")
    g.add_argument("--can_id")
    ap.add_argument("--outdir", default=".", help="Image output directory")
    ap.add_argument("--plot", action="store_true", help="Save three plots")
    
    args = ap.parse_args()

    try:
        res = compare_from_csv(
            args.csv1, args.csv2,
            symbol=args.symbol, can_id=args.can_id
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        try:
            s1 = list_symbols(args.csv1)
            s2 = list_symbols(args.csv2)
            inter = sorted(s1 & s2)
            if inter:
                print("[Hint] Common symbols in both CSVs:")
                for s in inter[:50]:
                    print("  -", s)
            else:
                print("[Hint] No common symbols detected; sample from csv1:")
                for s in sorted(s1)[:50]:
                    print("  -", s)
        except Exception:
            pass
        return

    print("=== Consistency Report ===")
    print(f"Signal       : {res['symbol'] or res['can_id']}")
    print("-" * 30)
    print(f"  Rigid Score : {res['rigid_trend_score']*100:.2f}")
    print(f"    - rigid_corr       : {res['rigid_corr']:.4f}")
    print(f"    - rigid_rmse       : {res['rigid_rmse']:.4f}")
    print(f"  Shape Score : {res['shape_trend_score']*100:.2f}")
    print(f"    - dtw_rms          : {res['dtw_rms']:.4f}")
    print(f"  EndState Score: {res['steady_score']*100:.2f}")
    print("-" * 30)
    
    fw = res['fused_weights']
    print(f"FUSED_SCORE    : {res['fused_score']*100:.2f}  "
          f"(W_shape={fw['shape']:.2f}, W_rigid={fw['rigid']:.2f}, W_endstate={fw['steady']:.2f})")

    if args.plot:
        os.makedirs(args.outdir, exist_ok=True)
        sig = (res['symbol'] or res['can_id'] or 'signal').replace('/', '_')
        extra = f"{res['mode']}"
        if res.get('w_samples') is not None:
            extra += f"_w{res['w_samples']}"
        else:
            extra += f"_band{res['band_frac']:.3f}"
        extra += f"_{res['time_base']}"
        extra += f"_smooth{res['trend_smooth_win']}"
        
        prefix = DEFAULT_CONFIG["img_prefix"] or f"{sig}_{extra}"
        dpi = DEFAULT_CONFIG["dpi"]

        res_path = os.path.join(args.outdir, f"{prefix}_resampled.png")
        plot_resampled_overlay(res, res_path, dpi=dpi, title="ECU Trace (Resampled Overlay)")
        print(f"[Saved] {res_path}")

        side_path = os.path.join(args.outdir, f"{prefix}_relative_side_by_side.png")
        plot_side_by_side_relative(res, side_path, dpi=dpi, title="ECU Trace (Relative Time, Side-by-Side)")
        print(f"[Saved] {side_path}")

        slope_path = os.path.join(args.outdir, f"{prefix}_slope.png")
        plot_trend_and_slope(res, slope_path, dpi=dpi, title=f"ShapeScore Internals (mode={res['mode']})")
        print(f"[Saved] {slope_path}")

if __name__ == "__main__":
    main()