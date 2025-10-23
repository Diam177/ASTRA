
# -*- coding: utf-8 -*-
"""
heatmap.py
One-file module for level-strength scoring and heatmap plotting.

API:
- compute_scores(level_prices, factors, *, spot, flip_side=None, weights=None,
                 type_map=None, norm="p90", lambda_types=0.2, tau=0.015,
                 side_gain=0.15, cap_percentile=95) -> pandas.DataFrame
- build_heatmap(levels_df, price_series=None, *, price_col="price",
                score_col="score", label_col=None, zmin=0, zmax=400,
                title=None, overlay_mode="path") -> plotly.graph_objects.Figure

Notes:
- No history used. All normalizations are within the current snapshot/session.
- factors: dict[str, pandas.Series] aligned to level_prices (same length/index).
- type_map: dict[str, str] mapping factor -> type group
  (e.g., {"AG":"pin","PZ":"pin","OI":"flow","VOL":"flow","GFLIP":"mode"}).
- flip_side: Optional[str] in {"P","N"}; controls side-dependent multipliers.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ---------------------------- Normalization helpers ----------------------------

def _norm_p90(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Robust scale to [0,1] by 90th percentile within current snapshot."""
    x = np.asarray(x, dtype=float)
    p90 = np.nanpercentile(x[np.isfinite(x)], 90) if np.isfinite(x).any() else 0.0
    denom = max(p90, eps)
    y = np.clip(x / denom, 0.0, 1.0)
    y[~np.isfinite(y)] = 0.0
    return y


def _norm_share(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Share of total, then re-scale to [0,1] by max share."""
    x = np.asarray(x, dtype=float)
    total = np.nansum(x)
    if not np.isfinite(total) or total <= eps:
        return np.zeros_like(x, dtype=float)
    share = x / total
    m = np.nanmax(share) if np.isfinite(share).any() else 0.0
    denom = max(m, eps)
    y = np.clip(share / denom, 0.0, 1.0)
    y[~np.isfinite(y)] = 0.0
    return y


def _norm_rank(x: np.ndarray) -> np.ndarray:
    """Rank-based [0,1] scaling, ties get average rank. All-zeros -> zeros."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return x
    order = np.argsort(x, kind="mergesort")  # stable
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    # handle ties: average ranks per unique value
    uniq, inv = np.unique(x, return_inverse=True)
    avg = np.bincount(inv, ranks) / np.bincount(inv)
    y = avg[inv] / n
    y[~np.isfinite(y)] = 0.0
    return y


def _normalize(x: np.ndarray, method: str) -> np.ndarray:
    method = method.lower()
    if method == "p90":
        return _norm_p90(x)
    if method == "share":
        return _norm_share(x)
    if method == "rank":
        return _norm_rank(x)
    raise ValueError(f"Unknown norm method: {method}")


# ------------------------------- Scoring config --------------------------------

@dataclass
class ScoreConfig:
    # Static priors (weights). Adjust to your preference.
    weights: Dict[str, float] = field(default_factory=lambda: {
        # examples: tune to your project
        "AG": 1.0,        # absolute gamma intensity
        "OI": 0.6,        # total open interest
        "VOL": 0.7,       # session option volume
        "PZ": 0.8,        # power zone or similar pin metric
        "GFLIP": 0.9,     # presence of gamma flip at/near level
    })
    # Grouping for multi-type bonus and side multipliers
    type_map: Dict[str, str] = field(default_factory=lambda: {
        "AG": "pin", "PZ": "pin", "OI": "flow", "VOL": "flow", "GFLIP": "mode"
    })
    # Normalization method for factors within the current snapshot
    norm: str = "p90"  # one of {"p90","share","rank"}

    # Multipliers
    lambda_types: float = 0.2    # strength of multi-type bonus
    tau: float = 0.015           # spot proximity decay ~1.5% of spot by default
    side_gain: float = 0.15      # side regime tilt strength per group

    # Percentile for session-scale cap to 400
    cap_percentile: float = 95.0


# ------------------------------- Core scoring ----------------------------------

def compute_scores(
    level_prices: pd.Series,
    factors: Dict[str, pd.Series],
    *,
    spot: float,
    flip_side: Optional[str] = None,   # "P" or "N" or None
    weights: Optional[Dict[str, float]] = None,
    type_map: Optional[Dict[str, str]] = None,
    norm: Optional[str] = None,
    lambda_types: Optional[float] = None,
    tau: Optional[float] = None,
    side_gain: Optional[float] = None,
    cap_percentile: Optional[float] = None,
) -> pd.DataFrame:
    """
    Vectorized scoring for all levels within the current snapshot.

    Parameters
    ----------
    level_prices : Series[float]
        Vector of level/strike prices (one per level/cluster).
    factors : dict[str, Series[float]]
        Keys are factor names. Values must align with level_prices index.
        Example factors: {"AG": |AG|, "OI": OI_call+OI_put, "VOL": Vol_call+Vol_put,
                          "PZ": pin_strength (0..1 or value), "GFLIP": {0,1}}.
        Missing factors are allowed; absent values are treated as 0.
    spot : float
        Current underlying price.
    flip_side : {"P","N"} or None
        Market regime side relative to gamma flip. If None, side multipliers are skipped.
    weights : Optional[dict[str,float]]
        Override of static priors. Falls back to ScoreConfig().weights.
    type_map : Optional[dict[str,str]]
        Override factor->type group. Falls back to ScoreConfig().type_map.
    norm : {"p90","share","rank"} or None
        Normalization method. Default "p90".
    lambda_types : float or None
        Strength of multi-type bonus. Default 0.2. Effective range ~[0.15,0.25].
    tau : float or None
        Decay scale of spot proximity in relative price units. Default 0.015.
    side_gain : float or None
        Tilt size for side-dependent weighting of groups. Default 0.15.
    cap_percentile : float or None
        Percentile for session-dependent scaling to 400. Default 95.

    Returns
    -------
    DataFrame with columns:
        price, base, m_types, m_spot, m_side, raw_score, score, and q_<factor> for each factor.
    """
    cfg = ScoreConfig()
    W = weights or cfg.weights
    TMAP = type_map or cfg.type_map
    norm_method = (norm or cfg.norm).lower()
    lambda_types = cfg.lambda_types if lambda_types is None else float(lambda_types)
    tau = cfg.tau if tau is None else float(tau)
    side_gain = cfg.side_gain if side_gain is None else float(side_gain)
    capp = cfg.cap_percentile if cap_percentile is None else float(cap_percentile)

    # Align factors to the same index
    idx = level_prices.index
    prices = level_prices.astype(float).to_numpy()
    S = float(spot)

    # Normalize each factor to [0,1] within snapshot
    q: Dict[str, np.ndarray] = {}
    for name, series in factors.items():
        x = pd.Series(series).reindex(idx).astype(float).to_numpy()
        if norm_method == "p90":
            q[name] = _norm_p90(x)
        elif norm_method == "share":
            q[name] = _norm_share(x)
        elif norm_method == "rank":
            q[name] = _norm_rank(x)
        else:
            raise ValueError(f"Unknown norm method: {norm_method}")

    # Base score (static priors). Missing factors default to 0.
    base = np.zeros(len(idx), dtype=float)
    for name, w in W.items():
        base += float(w) * q.get(name, np.zeros_like(base))

    # Multi-type smooth bonus
    # Count distinct groups with material presence
    present_groups = []
    for name, v in q.items():
        if np.nanmax(v) > 0:  # factor exists
            present_groups.append(TMAP.get(name, name))
    # per level: count of groups with q_level>0
    group_names = list(set(TMAP.get(n, n) for n in q.keys()))
    T = np.zeros_like(base)
    for g in group_names:
        # presence if any factor from group has q>0 at the level
        present = np.zeros_like(base)
        for n, v in q.items():
            if TMAP.get(n, n) == g:
                present = np.maximum(present, (v > 0).astype(float))
        T += present
    m_types = 1.0 + lambda_types * (np.power(np.clip(T, 1.0, None), 0.6) - 1.0)
    m_types = np.clip(m_types, 1.0, 1.35)

    # Proximity to spot
    d = np.abs(S - prices) / max(S, 1e-12)
    m_spot = np.exp(-d / max(tau, 1e-8))

    # Side-dependent tilt by group
    if flip_side in ("P", "N"):
        # groups tagged as "flow" get boosted on the "volatile" side, "pin" gets reduced; symmetric otherwise.
        # Heuristic: if flip_side == "N", market is in negative-gamma regime.
        if flip_side == "N":
            gain_flow, gain_pin = +side_gain, -side_gain
        else:  # "P"
            gain_flow, gain_pin = -side_gain, +side_gain

        # Compute per-level multiplier from group composition
        # Start at 1.0, then tilt proportional to group presence intensity
        m_side = np.ones_like(base)
        # aggregate intensities per group as max q among its factors at each level
        q_group: Dict[str, np.ndarray] = {}
        for g in group_names:
            acc = np.zeros_like(base)
            for n, v in q.items():
                if TMAP.get(n, n) == g:
                    acc = np.maximum(acc, v)
            q_group[g] = acc

        # apply tilts
        pin_intensity = q_group.get("pin", np.zeros_like(base))
        flow_intensity = q_group.get("flow", np.zeros_like(base))
        m_side *= (1.0 + gain_flow * flow_intensity)
        m_side *= (1.0 + gain_pin * pin_intensity)
        # other groups unchanged
    else:
        m_side = np.ones_like(base)

    # Raw score before percentile cap
    raw = base * m_types * m_spot * m_side

    # Session percentile cap to 400
    valid = raw[np.isfinite(raw)]
    pctl = np.percentile(valid, capp) if valid.size else 0.0
    if not np.isfinite(pctl) or pctl <= 1e-12:
        denom = max(valid.max() if valid.size else 1.0, 1.0)
    else:
        denom = pctl
    scaled = 400.0 * np.clip(raw / denom, 0.0, 1.0)

    # Build DataFrame
    out = pd.DataFrame(index=idx, data={
        "price": prices,
        "base": base,
        "m_types": m_types,
        "m_spot": m_spot,
        "m_side": m_side,
        "raw_score": raw,
        "score": scaled,
    })
    # attach factor intensities
    for name, v in q.items():
        out[f"q_{name}"] = v
    return out


# --------------------------------- Plotting ------------------------------------

def build_heatmap(
    levels_df: pd.DataFrame,
    price_series: Optional[pd.DataFrame] = None,
    *,
    price_col: str = "price",
    score_col: str = "score",
    label_col: Optional[str] = None,
    zmin: float = 0.0,
    zmax: float = 400.0,
    title: Optional[str] = None,
    overlay_mode: str = "path",   # {"path","line"}
    right_labels: Optional[Dict[float, str]] = None,
) -> go.Figure:
    """
    Build a Viridis heatmap of level strength with optional price overlay.

    levels_df must contain columns: price_col, score_col; optionally label_col.
    price_series, if provided, must contain ["timestamp","price"]. When given and
    overlay_mode="path", the heatmap is tiled across time to align a price path.
    """
    if price_col not in levels_df or score_col not in levels_df:
        raise ValueError(f"levels_df must have '{price_col}' and '{score_col}' columns")

    # Prepare axes
    lv = levels_df[[price_col, score_col]].dropna().copy()
    lv = lv.sort_values(price_col)
    y_prices = lv[price_col].to_numpy(dtype=float)
    scores = lv[score_col].to_numpy(dtype=float)
    scores = np.clip(scores, zmin, zmax)

    if price_series is not None and overlay_mode == "path":
        ps = price_series.copy()
        # Normalize column names to lowercase
        ps.columns = [str(c).lower() for c in ps.columns]
        # Normalize time axis
        if "time" in ps.columns:
            x = pd.to_datetime(ps["time"], errors="coerce")
        elif "timestamp" in ps.columns:
            x = pd.to_datetime(ps["timestamp"], unit="ms", errors="coerce")
        elif "t" in ps.columns:
            x = pd.to_datetime(ps["t"], unit="ms", errors="coerce")
        else:
            raise ValueError("price_series must contain a time column: 'time' or 'timestamp' or 't'")
        ps = ps.assign(__time=x).dropna(subset=["__time"]).sort_values("__time")
        x = ps["__time"]
        # Tile heatmap across time for alignment
        Z = np.tile(scores.reshape(-1, 1), (1, len(x)))
        fig = go.Figure(data=[
            go.Heatmap(
                z=Z,
                x=x,
                y=y_prices,
                colorscale="Viridis",
                zmin=zmin, zmax=zmax,
                colorbar=dict(title="Level strength", ticksuffix="")
            ),
        ])
        fig.update_layout(
            title=title or "Level Strength Heatmap",
            xaxis_title="Time",
            yaxis_title="Price",
            template="plotly_white",
            height=800,
        )
    else:
        # Single-column heatmap with optional horizontal price line
        Z = scores.reshape(-1, 1)
        fig = go.Figure(data=[
            go.Heatmap(
                z=Z,
                x=["strength"],
                y=y_prices,
                colorscale="Viridis",
                zmin=zmin, zmax=zmax,
                colorbar=dict(title="Level strength", ticksuffix="")
            )
        ])
        fig.update_layout(
            title=title or "Level Strength Heatmap",
            xaxis_title="",
            yaxis_title="Price",
            template="plotly_white",
            height=800,
            xaxis=dict(showticklabels=False)
        )
        if price_series is not None and "price" in price_series:
            last_price = float(price_series["price"].iloc[-1])
            fig.add_hline(y=last_price, line=dict(width=2), annotation_text="Spot")

    # --- Overlay: minute Price candles and VWAP like in key_levels ---
    try:
        if price_series is not None and len(fig.data) <= 1:
            pdf = price_series.copy()
            # Normalize column names to lowercase for flexible matching
            pdf.columns = [str(c).lower() for c in pdf.columns]
            # Normalize time column
            if "time" not in pdf.columns:
                if "timestamp" in pdf.columns:
                    pdf["time"] = pd.to_datetime(pdf["timestamp"], unit="ms", errors="coerce")
                elif "t" in pdf.columns:
                    pdf["time"] = pd.to_datetime(pdf["t"], unit="ms", errors="coerce")
            # If still no time column but index looks like datetime, use it
            if "time" not in pdf.columns and isinstance(pdf.index, pd.DatetimeIndex):
                pdf = pdf.reset_index().rename(columns={"index":"time"})
            if "time" not in pdf.columns:
                raise ValueError("price_series must contain ['timestamp','price'] or ['time','price']")
            pdf = pdf.dropna(subset=["time"]).sort_values("time")

            # Choose price columns
            col_price = None
            if "price" in pdf.columns:
                col_price = "price"
            elif "c" in pdf.columns:
                col_price = "c"

            # Candlestick if OHLC available
            has_ohlc = {"open","high","low","close"}.issubset(set(pdf.columns)) or {"o","h","l","c"}.issubset(set(pdf.columns))
            if has_ohlc:
                # Map polygon aliases
                o = pdf["open"] if "open" in pdf.columns else pd.to_numeric(pdf["o"], errors="coerce")
                h = pdf["high"] if "high" in pdf.columns else pd.to_numeric(pdf["h"], errors="coerce")
                l = pdf["low"]  if "low"  in pdf.columns else pd.to_numeric(pdf["l"], errors="coerce")
                c = pdf["close"] if "close" in pdf.columns else pd.to_numeric(pdf["c"], errors="coerce")
                fig.add_trace(go.Candlestick(
                    x=pdf["time"], open=pd.to_numeric(o, errors="coerce"),
                    high=pd.to_numeric(h, errors="coerce"),
                    low=pd.to_numeric(l, errors="coerce"),
                    close=pd.to_numeric(c, errors="coerce"),
                    name="Price",
                    showlegend=True,
                ))
            elif col_price is not None:
                fig.add_trace(go.Scatter(
                    x=pdf["time"], y=pd.to_numeric(pdf[col_price], errors="coerce"),
                    mode="lines",
                    line=dict(width=1.2),
                    name="Price",
                    hovertemplate="Time: %{x|%H:%M}<br>Price: %{y:.2f}<extra></extra>",
                    showlegend=True,
                ))

            # VWAP
            vwap_series = None
            if "vwap" in pdf.columns:
                vwap_series = pd.to_numeric(pdf["vwap"], errors="coerce")
            elif "vw" in pdf.columns:
                try:
                    vwap_series = pd.to_numeric(pdf["vw"], errors="coerce").expanding().mean()
                except Exception:
                    vwap_series = pd.to_numeric(pdf["vw"], errors="coerce")
            elif col_price is not None and "volume" in pdf.columns:
                vol = pd.to_numeric(pdf["volume"], errors="coerce").fillna(0.0)
                pr  = pd.to_numeric(pdf[col_price], errors="coerce").fillna(pd.NA)
                cum_vol = vol.cumsum()
                vwap_series = (pr.mul(vol)).cumsum() / cum_vol.replace(0, np.nan)
            if vwap_series is not None:
                fig.add_trace(go.Scatter(
                    x=pdf["time"], y=vwap_series,
                    mode="lines",
                    line=dict(width=1.0),
                    name="VWAP",
                    hovertemplate="Time: %{x|%H:%M}<br>VWAP: %{y:.2f}<extra></extra>",
                    showlegend=True,
                ))

        # --- Left y-axis ticks at all levels that have parameters ---
        q_cols = [c for c in levels_df.columns if c.startswith("q_")]
        if len(q_cols) > 0:
            lv_vals = (
                levels_df.loc[(levels_df[q_cols] > 0).any(axis=1), price_col]
                .astype(float).dropna().unique().tolist()
            )
        else:
            lv_vals = levels_df[price_col].astype(float).dropna().unique().tolist()
        if len(lv_vals) > 0:
            lv_vals = sorted(lv_vals)
            fig.update_yaxes(tickmode="array", tickvals=lv_vals,
                             ticktext=[("{:g}".format(v)) for v in lv_vals])
    except Exception as _e:
        # fail-safe: do not break chart rendering
        pass

    # Labels as hover text
    if label_col and label_col in levels_df.columns:
        # Build hovertext aligned with y_prices order
        lv2 = levels_df[[price_col, label_col]].dropna().copy().sort_values(price_col)
        # Expand to Z shape
        lab = lv2[label_col].astype(str).to_list()
        hover_y = [f"{p:.2f} | {t}" for p, t in zip(y_prices, lab)]
        if price_series is not None and overlay_mode == "path":
            hover = np.tile(np.array(hover_y, dtype=object).reshape(-1, 1), (1, Z.shape[1]))
        else:
            hover = np.array(hover_y, dtype=object).reshape(-1, 1)
        fig.data[0].update(hoverinfo="text", text=hover)

    
    # --- Optional: right-side strike labels identical to Key Levels ---
    try:
        if right_labels:
            # Determine the rightmost x value inside plot
            x_vals = None
            try:
                x_vals = list(fig.data[0]['x'])
            except Exception:
                x_vals = None
            x_right_val = x_vals[-1] if isinstance(x_vals, (list, tuple)) and len(x_vals) > 0 else None
            # Current y scale values for snapping
            y_axis_vals = None
            try:
                y_axis_vals = [float(v) for v in fig.data[0]['y']]
            except Exception:
                y_axis_vals = None
            eps = 0.05
            for k_raw, txt in sorted(right_labels.items(), key=lambda kv: kv[0]):
                try:
                    y0 = float(k_raw)
                except Exception:
                    continue
                y_use = y0
                if isinstance(y_axis_vals, list) and y_axis_vals:
                    nearest = min(y_axis_vals, key=lambda v: abs(v - y0))
                    if abs(nearest - y0) <= eps:
                        y_use = nearest
                if not isinstance(txt, str) or not txt.strip():
                    continue
                if x_right_val is not None:
                    fig.add_annotation(
                        x=x_right_val, xref="x",
                        y=float(y_use), yref="y",
                        text=txt.strip(),
                        showarrow=False,
                        xanchor="right", yanchor="bottom",
                        xshift=-2, yshift=6,
                        align="right",
                        font=dict(size=10, color="#FFFFFF"),
                        bgcolor="rgba(0,0,0,0.35)",
                        borderwidth=0.5,
                    )
                else:
                    fig.add_annotation(
                        x=1.0, xref="paper",
                        y=float(y_use), yref="y",
                        text=txt.strip(),
                        showarrow=False,
                        xanchor="right", yanchor="bottom",
                        yshift=6,
                        align="right",
                        font=dict(size=10, color="#FFFFFF"),
                        bgcolor="rgba(0,0,0,0.35)",
                        borderwidth=0.5,
                    )
    except Exception:
        # do not fail rendering because of labels
        pass

    return fig

# ===== astra: heatmap methodology v1 =====
import numpy as _np
import pandas as _pd
import plotly.graph_objects as go
import streamlit as st
import math as _math

def _np_clip01(x):
    return _np.minimum(1.0, _np.maximum(0.0, x))

def _median_diff_unique(K):
    try:
        v = _np.array(sorted(set(_pd.to_numeric(K, errors="coerce").dropna().astype(float).tolist())), dtype=float)
        if v.size < 2:
            return float("nan")
        d = _np.diff(v)
        return float(_np.median(d)) if d.size else float("nan")
    except Exception:
        return float("nan")

def _gaussian_smooth_by_K(df: _pd.DataFrame, cols, h):
    if not _np.isfinite(h) or h <= 0:
        return df.copy()
    out = df.copy()
    K = _pd.to_numeric(df.get("K"), errors="coerce").astype(float)
    mask = K.notna()
    Kv = K[mask].to_numpy()
    if Kv.size == 0:
        return out
    for c in cols:
        if c not in df.columns:
            continue
        x = _pd.to_numeric(df[c], errors="coerce")
        xv = x[mask].to_numpy(dtype=float)
        if xv.size == 0:
            out[c] = 0.0
            continue
        out_vals = _np.empty_like(xv)
        for i, k0 in enumerate(Kv):
            d = (Kv - k0) / h
            w = _np.exp(-0.5 * d * d)
            s = _np.sum(w * xv)
            W = _np.sum(w)
            out_vals[i] = s / W if W > 0 else 0.0
        y = _pd.Series(index=K.index, dtype=float)
        y.loc[mask] = out_vals
        y.loc[~mask] = _np.nan
        out[c] = y
    return out

def _ridge_proj_y_on_X(y: _pd.Series, X: _pd.DataFrame, lam=1e-6):
    try:
        Y = _pd.to_numeric(y, errors="coerce").astype(float).fillna(0.0).to_numpy()[:, None]
        Xn = _pd.DataFrame({c: _pd.to_numeric(X[c], errors="coerce").astype(float).fillna(0.0) for c in X.columns})
        Xv = Xn.to_numpy(float)
        XtX = Xv.T @ Xv
        XtY = Xv.T @ Y
        XtX_reg = XtX + lam * _np.eye(XtX.shape[0])
        coef = _np.linalg.solve(XtX_reg, XtY)
        y_hat = (Xv @ coef).ravel()
        resid = (Y.ravel() - y_hat)
        return resid
    except Exception:
        return _pd.to_numeric(y, errors="coerce").astype(float).fillna(0.0).to_numpy()

def _rank_top_k(vals: _pd.Series, side="pos", k=3):
    v = _pd.to_numeric(vals, errors="coerce").astype(float)
    if side == "pos":
        idx = v[v > 0].nlargest(k).index.tolist()
    else:
        idx = v[v < 0].nsmallest(k).index.tolist()
    return idx

def _component_scores(df: _pd.DataFrame, gflip=None, ticker_hint=None, rv_z=0.0):
    NG = _pd.to_numeric(df.get("NetGEX_1pct_M", df.get("NetGEX_1pct")/1e6), errors="coerce").astype(float).fillna(0.0)
    AG = _pd.to_numeric(df.get("AG_1pct_M", df.get("AG_1pct")), errors="coerce").astype(float).fillna(0.0)
    PZ = _pd.to_numeric(df.get("PZ"), errors="coerce").astype(float).fillna(0.0)
    COI = _pd.to_numeric(df.get("call_oi"), errors="coerce").astype(float).fillna(0.0)
    POI = _pd.to_numeric(df.get("put_oi"), errors="coerce").astype(float).fillna(0.0)
    CV = _pd.to_numeric(df.get("call_vol"), errors="coerce").astype(float).fillna(0.0)
    PV = _pd.to_numeric(df.get("put_vol"), errors="coerce").astype(float).fillna(0.0)

    K = _pd.to_numeric(df.get("K"), errors="coerce").astype(float)
    dK = _median_diff_unique(K)
    sm = _gaussian_smooth_by_K(_pd.DataFrame(dict(K=K, NG=NG, AG=AG, PZ=PZ, COI=COI, POI=POI, CV=CV, PV=PV)), 
                               cols=["NG","AG","PZ","COI","POI","CV","PV"], h=dK if _np.isfinite(dK) else 0.0)
    NG, AG, PZ, COI, POI, CV, PV = (sm[c] for c in ["NG","AG","PZ","COI","POI","CV","PV"])

    AG_resid = _ridge_proj_y_on_X(AG, _pd.DataFrame({"absNG": _np.abs(NG), "PZ": PZ}))
    PZ_resid = _ridge_proj_y_on_X(PZ, _pd.DataFrame({"absNG": _np.abs(NG), "AG": AG}))
    AG_resid = _pd.Series(AG_resid, index=df.index)
    PZ_resid = _pd.Series(PZ_resid, index=df.index)

    Ppos = _rank_top_k(NG, "pos", 3)
    Pneg = _rank_top_k(NG, "neg", 3)

    Pmax = float(_np.nanmax(_np.where(NG.values>0, NG.values, _np.nan))) if _np.any(NG.values>0) else 0.0
    Nmax = float(_np.nanmax(_np.where(NG.values<0, -NG.values, _np.nan))) if _np.any(NG.values<0) else 0.0
    s_pos = _np.where(NG.values>0, NG.values/(Pmax+1e-12), 0.0)
    s_neg = _np.where(NG.values<0, -NG.values/(Nmax+1e-12), 0.0)
    p = 0.8
    r = _np.ones_like(s_pos)
    r1, r2, r3 = 0.50, 0.25, 0.15
    for j, idxs in enumerate([Ppos[:1], Ppos[1:2], Ppos[2:3]]):
        for ix in idxs:
            r[ix] += [r1, r2, r3][j]
    for j, idxs in enumerate([Pneg[:1], Pneg[1:2], Pneg[2:3]]):
        for ix in idxs:
            r[ix] += [r1, r2, r3][j]
    z_minus = 1.0 + max(rv_z, 0.0)*0.2
    z = _np.where(NG.values<0, z_minus, 1.0)
    NG_score = z * _np.power(s_pos + s_neg, p) * r

    def pos_norm(x):
        xv = _np.maximum(_pd.to_numeric(x, errors="coerce").astype(float).fillna(0.0).values, 0.0)
        m = float(xv.max()) if xv.size else 0.0
        return (xv/(m+1e-12))**0.8 if m>0 else xv*0.0
    AG_s = pos_norm(AG_resid)
    PZ_s = pos_norm(PZ_resid)

    def apply_boost(base, series):
        s = _pd.to_numeric(series, errors="coerce").astype(float)
        idx = s.nlargest(3).index.tolist()
        out = base.copy()
        if len(idx)>0: out[idx[0]] *= (1+0.30)
        if len(idx)>1: out[idx[1]] *= (1+0.15)
        if len(idx)>2: out[idx[2]] *= (1+0.15)
        return out
    AG_score = apply_boost(AG_s, _pd.Series(AG_resid, index=df.index))
    PZ_score = apply_boost(PZ_s, _pd.Series(PZ_resid, index=df.index))

    def log_norm(x):
        x = _pd.to_numeric(x, errors="coerce").astype(float).fillna(0.0)
        y = _np.log1p(_np.maximum(x.values, 0.0))
        m = float(y.max()) if y.size else 0.0
        return y/(m+1e-12) if m>0 else y*0.0
    COI_n, POI_n = log_norm(COI), log_norm(POI)
    CV_n, PV_n = log_norm(CV), log_norm(PV)

    if gflip is not None and _np.isfinite(gflip):
        zone = _np.sign(K.values - float(gflip))
    else:
        zone = _np.zeros_like(K.values)
    w_call_pos, w_put_pos = 1.0, 0.6
    w_call_neg, w_put_neg = 0.6, 1.0
    w_call = _np.where(zone>=0, w_call_pos, w_call_neg)
    w_put = _np.where(zone>=0, w_put_pos, w_put_neg)
    OI_score  = w_call*COI_n + w_put*POI_n
    VOL_score = w_call*CV_n  + w_put*PV_n

    if gflip is not None and _np.isfinite(gflip) and _np.isfinite(dK) and dK>0:
        dist = (K.values - float(gflip))/(2.0*dK)
        G0 = _np.exp(-0.5*dist*dist)
        try:
            order = _np.argsort(_np.abs(K.values - float(gflip)))[:5]
            grad = _np.gradient(NG.values[order], K.values[order])
            grad_amp = float(_np.nanmax(_np.abs(grad)))
        except Exception:
            grad_amp = 0.0
        try:
            all_grad = _np.gradient(NG.values, K.values)
            q90 = float(_np.nanpercentile(_np.abs(all_grad), 90.0))
        except Exception:
            q90 = 0.0
        mult = 1.0 + 0.5*max(grad_amp/(q90+1e-12)-1.0, 0.0) if q90>0 else 1.0
        try:
            S = float(_pd.to_numeric(df.get("S"), errors="coerce").dropna().median())
            spot_mult = _math.exp(-abs(S - float(gflip)) / (3.0*(dK if dK>0 else 1.0)))
        except Exception:
            spot_mult = 1.0
        Gflip_score = G0 * mult * spot_mult
    else:
        Gflip_score = _np.zeros_like(K.values, dtype=float)

    return dict(NG_score=NG_score, AG_score=AG_score, PZ_score=PZ_score,
                OI_score=OI_score, VOL_score=VOL_score, Gflip_score=Gflip_score,
                dK=dK)

def _regime_weights(ticker:str, share_0dte:float, rv_z:float):
    if isinstance(ticker, str) and ticker.upper().endswith("SPY"):
        w = dict(NG=0.42, AG=0.18, PZ=0.18, OI=0.10, VOL=0.07, GFLIP=0.05)
        if (share_0dte is not None and share_0dte>=0.5) or (rv_z is not None and rv_z>=1.0):
            w["NG"]+=0.04; w["VOL"]+=0.02; w["GFLIP"]+=0.02; w["OI"]-=0.02; w["AG"]-=0.02
    else:
        w = dict(NG=0.50, AG=0.14, PZ=0.14, OI=0.06, VOL=0.08, GFLIP=0.08)
        if (share_0dte is not None and share_0dte>=0.5) or (rv_z is not None and rv_z>=1.0):
            w["NG"]+=0.05; w["VOL"]+=0.03; w["GFLIP"]+=0.02; w["OI"]-=0.03; w["AG"]-=0.03
    s = sum(w.values())
    for k in w: w[k] /= s if s>0 else 1.0
    return w

def _concave_aggregate(scores:dict, weights:dict):
    NG_s = _np_clip01(scores["NG_score"])
    AG_s = _np_clip01(scores["AG_score"])
    PZ_s = _np_clip01(scores["PZ_score"])
    OI_s = _np_clip01(scores["OI_score"])
    VOL_s= _np_clip01(scores["VOL_score"])
    GF_s = _np_clip01(scores["Gflip_score"])
    w = weights
    one_minus = (1 - w["NG"]*NG_s) * (1 - w["AG"]*AG_s) * (1 - w["PZ"]*PZ_s) * (1 - w["GFLIP"]*GF_s) * (1 - (w["OI"]+w["VOL"])*_np_clip01(0.5*OI_s + 0.5*VOL_s))
    H_raw = 1 - one_minus
    return H_raw, dict(NG_s=NG_s, AG_s=AG_s, PZ_s=PZ_s, OI_s=OI_s, VOL_s=VOL_s, Gflip_s=GF_s)

def _robust_norm(H_raw):
    if H_raw.size == 0:
        return H_raw
    q05 = float(_np.nanpercentile(H_raw, 5.0))
    q95 = float(_np.nanpercentile(H_raw, 95.0))
    H = (H_raw - q05) / ( (q95 - q05) + 1e-12 )
    return _np_clip01(H)

def compute_heat_scores(df_final:_pd.DataFrame, gflip=None, ticker_hint=None, rv_z:float=0.0, share_0dte:float=None) -> _pd.DataFrame:
    if df_final is None or len(df_final)==0:
        return _pd.DataFrame(columns=["K","H","H_raw","NG_s","AG_s","PZ_s","OI_s","VOL_s","Gflip_s"])
    if gflip is None:
        try:
            gflip = (getattr(df_final, "attrs", {}).get("gflip", {}) or {}).get("cross", None)
            if gflip is None and "G_FLIP" in df_final.columns:
                gflip = float(_pd.to_numeric(df_final["G_FLIP"], errors="coerce").dropna().iloc[0])
        except Exception:
            gflip = None
    if ticker_hint is None:
        try:
            ticker_hint = str(st.session_state.get("ticker","")).upper()
        except Exception:
            ticker_hint = ""
    scores = _component_scores(df_final, gflip=gflip, ticker_hint=ticker_hint, rv_z=rv_z)
    if share_0dte is None:
        try:
            share_0dte = float(getattr(df_final, "attrs", {}).get("mode", {}).get("share_0DTE", 0.0))
        except Exception:
            share_0dte = 0.0
    w = _regime_weights(ticker_hint or "", share_0dte=share_0dte, rv_z=rv_z)
    H_raw, parts = _concave_aggregate(scores, w)
    H = _robust_norm(H_raw.copy())
    out = _pd.DataFrame({
        "K": _pd.to_numeric(df_final.get("K"), errors="coerce").astype(float),
        "H": H,
        "H_raw": H_raw,
        "NG_s": parts["NG_s"],
        "AG_s": parts["AG_s"],
        "PZ_s": parts["PZ_s"],
        "OI_s": parts["OI_s"],
        "VOL_s": parts["VOL_s"],
        "Gflip_s": parts["Gflip_s"],
    })
    return out.sort_values("K").reset_index(drop=True)

def _render_level_strength_heatmap(df_final:_pd.DataFrame, price_df=None, gflip=None, spot=None):
    """
    Smooth heatmap along strikes with blurred transitions between levels.
    Only this function is modified. Other code untouched.
    """
    try:
        rv_z = 0.0  # placeholder
        scores = compute_heat_scores(df_final, gflip=gflip, rv_z=rv_z)
        if scores.empty:
            st.info("Нет данных для heatmap.")
            return

        # Base vectors
        K = _pd.to_numeric(scores["K"], errors="coerce").astype(float).to_numpy()
        H = _pd.to_numeric(scores["H"], errors="coerce").astype(float).to_numpy()

        # Sort by strike
        order = np.argsort(K)
        K = K[order]; H = H[order]

        # Upsample + gaussian blur along K to mimic soft boundaries
        def _smooth_along_strike(K, H, up=8, sigma=0.6):
            if len(K) < 2:
                return K, H
            K_fine = np.linspace(K.min(), K.max(), (len(K)-1)*up + 1)
            H_lin = np.interp(K_fine, K, H)

            # Gaussian kernel in index domain
            half = int(max(1, round(3*sigma*up)))
            idx = np.arange(-half, half+1)
            ker = np.exp(-0.5*(idx/(sigma*up))**2)
            ker = ker / ker.sum()
            H_blur = np.convolve(H_lin, ker, mode="same")
            return K_fine, H_blur

        K_fine, H_fine = _smooth_along_strike(K, H, up=12, sigma=0.6)

        # Time axis
        x_vals = None; price_vals = None
        if price_df is not None and len(price_df) > 0:
            pdf = price_df.copy()
            pdf.columns = [str(c).lower() for c in pdf.columns]
            if "time" in pdf.columns:
                x_vals = _pd.to_datetime(pdf["time"], errors="coerce")
            elif "timestamp" in pdf.columns:
                x_vals = _pd.to_datetime(pdf["timestamp"], unit="ms", errors="coerce")
            elif "t" in pdf.columns:
                x_vals = _pd.to_datetime(pdf["t"], unit="ms", errors="coerce")
            if "price" in pdf.columns:
                price_vals = _pd.to_numeric(pdf["price"], errors="coerce").astype(float)
            elif "close" in pdf.columns:
                price_vals = _pd.to_numeric(pdf["close"], errors="coerce").astype(float)
        if x_vals is None:
            x_vals = np.arange(128)

        Z = np.tile(H_fine.reshape(-1,1), (1, len(x_vals)))

        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            x=x_vals, y=K_fine, z=Z,
            zmin=0, zmax=1,
            colorscale="Viridis",
            zsmooth="best",
            colorbar=dict(title="Level strength")
        ))

        if price_vals is not None:
            fig.add_trace(go.Scatter(x=x_vals, y=price_vals, mode="lines", name="Price", line=dict(width=1)))

        fig.update_layout(title="Level Strength Heatmap", height=800, template="plotly_white")
        fig.update_xaxes(title="Time")
        fig.update_yaxes(title="Price")

        if gflip is None:
            try:
                gflip = (getattr(df_final, "attrs", {}).get("gflip", {}) or {}).get("cross", None)
            except Exception:
                gflip = None
        if gflip is not None:
            fig.add_hline(y=float(gflip), line=dict(color="#E4A339", width=1, dash="dot"))

        st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False, "scrollZoom": False})
    except Exception as e:
        import traceback
        st.error(f"Heatmap exception: {type(e).__name__}: {e}")
        st.code(traceback.format_exc())
    try:
        rv_z = 0.0  # TODO: populate if доступно
        scores = compute_heat_scores(df_final, gflip=gflip, rv_z=rv_z)
        if scores.empty:
            st.info("Нет данных для heatmap.")
            return
        K = scores["K"].to_numpy()
        H = scores["H"].to_numpy()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=H, y=K, orientation="h",
            marker=dict(color=H, colorscale="Turbo"),
            showlegend=False, name="Heat"
        ))
        fig.update_yaxes(title="Strike K")
        fig.update_xaxes(title="Heat", range=[0,1])
        if gflip is None:
            try:
                gflip = (getattr(df_final, "attrs", {}).get("gflip", {}) or {}).get("cross", None)
            except Exception:
                gflip = None
        if gflip is not None:
            fig.add_hline(y=float(gflip), line=dict(color="#E4A339", width=1, dash="dot"))
        st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False, "scrollZoom": False})
    except Exception as e:
        import traceback
        st.error(f"Heatmap exception: {type(e).__name__}: {e}")
        st.code(traceback.format_exc())
