
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
        if not {"timestamp", "price"}.issubset(ps.columns):
            raise ValueError("price_series must contain ['timestamp','price'] or ['time','price']")
        ps = ps.sort_values("timestamp")
        x = pd.to_datetime(ps["timestamp"])
        # Tile heatmap across time for alignment
        Z = np.tile(scores.reshape(-1, 1).astype(float), (1, len(x)))
        fig = go.Figure(data=[
            go.Heatmap(zsmooth="best", 
                z=Z,
                x=x,
                y=y_prices,
                colorscale="Viridis",
                zmin=zmin, zmax=zmax,
                colorbar=dict(title="Level strength", ticksuffix="")
            ),
        ])
        fig.update_layout(height=850, 
            title=title or "Level Strength Heatmap",
            xaxis_title="Time",
            yaxis_title="Price",
            template="plotly_white",
        )
    else:
        # Single-column heatmap with optional horizontal price line
        Z = scores.reshape(-1, 1)
        fig = go.Figure(data=[
            go.Heatmap(zsmooth="best", 
                z=Z,
                x=["strength"],
                y=y_prices,
                colorscale="Viridis",
                zmin=zmin, zmax=zmax,
                colorbar=dict(title="Level strength", ticksuffix="")
            )
        ])
        fig.update_layout(height=850, 
            title=title or "Level Strength Heatmap",
            xaxis_title="",
            yaxis_title="Price",
            template="plotly_white",
            xaxis=dict(showticklabels=False)
        )
# --- Overlay: minute candles + VWAP like in Key Levels ---
        try:
            if price_series is not None:
                pdf = price_series.copy()
                # flexible column names
                pdf.columns = [str(c).lower() for c in pdf.columns]
                # time normalization
                if "time" not in pdf.columns:
                    if "timestamp" in pdf.columns:
                        pdf["time"] = pd.to_datetime(pdf["timestamp"], unit="ms", errors="coerce")
                    elif "t" in pdf.columns:
                        pdf["time"] = pd.to_datetime(pdf["t"], unit="ms", errors="coerce")
                if "time" in pdf.columns:
                    pdf = pdf.dropna(subset=["time"]).sort_values("time")
                    # Candlesticks if OHLC available
                    has_ohlc = {"open","high","low","close"}.issubset(set(pdf.columns)) or {"o","h","l","c"}.issubset(set(pdf.columns))
                    if has_ohlc:
                        o = pdf["open"] if "open" in pdf.columns else pdf["o"]
                        h = pdf["high"] if "high" in pdf.columns else pdf["h"]
                        l = pdf["low"]  if "low"  in pdf.columns else pdf["l"]
                        c = pdf["close"] if "close" in pdf.columns else pdf["c"]
                        fig.add_trace(go.Candlestick(
                            x=pdf["time"],
                            open=pd.to_numeric(o, errors="coerce"),
                            high=pd.to_numeric(h, errors="coerce"),
                            low=pd.to_numeric(l, errors="coerce"),
                            close=pd.to_numeric(c, errors="coerce"),
                            name="Price",
                            showlegend=True,
                        ))
                    # VWAP
                    vwap_series = None
                    if "vwap" in pdf.columns:
                        vwap_series = pd.to_numeric(pdf["vwap"], errors="coerce")
                    elif "vw" in pdf.columns:
                        # Polygon per-bar VWAP proxy
                        vwap_series = pd.to_numeric(pdf["vw"], errors="coerce").expanding().mean()
                    elif {"price","volume"}.issubset(set(pdf.columns)):
                        vol = pd.to_numeric(pdf["volume"], errors="coerce").fillna(0.0)
                        pr  = pd.to_numeric(pdf["price"], errors="coerce").fillna(pd.NA)
                        cum_vol = vol.cumsum().replace(0, pd.NA)
                        vwap_series = (pr.mul(vol)).cumsum() / cum_vol
                    if vwap_series is not None:
                        fig.add_trace(go.Scatter(
                            x=pdf["time"], y=vwap_series,
                            mode="lines",
                            line=dict(width=1.0, color="#E4A339"),
                            name="VWAP",
                            showlegend=True,
                            hovertemplate="Time: %{x|%H:%M}<br>VWAP: %{y:.2f}<extra></extra>",
                        ))
        except Exception:
            # fail-safe overlay
            pass

        if price_series is not None and "price" in price_series:
            last_price = float(price_series["price"].iloc[-1])
            fig.add_hline(y=last_price, line=dict(width=2), annotation_text="Spot")

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

    
    # --- Optional overlay of key-level lines from final table ---
    if overlay_key_levels and overlay_source_df is not None:
        try:
            import numpy as _np
            import pandas as _pd
            _df = overlay_source_df
            cols = set(_df.columns)
            def _agg_pick(df, col, fun="max", n=1, sign=None):
                if col not in cols:
                    return []
                g = (
                    df[["K", col]].copy()
                    .assign(K=lambda x: _pd.to_numeric(x["K"], errors="coerce"),
                            V=lambda x: _pd.to_numeric(x[col], errors="coerce"))
                    .dropna()
                    .groupby("K", as_index=False)["V"].sum()
                    .sort_values("V", ascending=(fun=="min"))
                )
                if sign == "neg":
                    g = g[g["V"] < 0]
                elif sign == "pos":
                    g = g[g["V"] > 0]
                if g.empty:
                    return []
                if n == 1:
                    return [float(g.iloc[0]["K"])]
                return [float(v) for v in g.head(n)["K"].tolist()]

            ng_col = "NetGEX_1pct_M" if "NetGEX_1pct_M" in cols else ("NetGEX_1pct" if "NetGEX_1pct" in cols else None)
            ag_col = "AG_1pct_M" if "AG_1pct_M" in cols else ("AG_1pct" if "AG_1pct" in cols else None)
            overlay_levels = []

            if ng_col:
                overlay_levels += _agg_pick(_df, ng_col, fun="min", n=3, sign="neg")  # N1..N3
                overlay_levels += _agg_pick(_df, ng_col, fun="max", n=3, sign="pos")  # P1..P3
            overlay_levels += _agg_pick(_df, "put_oi", fun="max", n=1)
            overlay_levels += _agg_pick(_df, "call_oi", fun="max", n=1)
            overlay_levels += _agg_pick(_df, "put_vol", fun="max", n=1) if "put_vol" in cols else []
            overlay_levels += _agg_pick(_df, "call_vol", fun="max", n=1) if "call_vol" in cols else []
            if ag_col:
                overlay_levels += _agg_pick(_df, ag_col, fun="max", n=1)
            overlay_levels += _agg_pick(_df, "PZ", fun="max", n=1) if "PZ" in cols else []

            if ng_col:
                g = (
                    _df[["K", ng_col]].copy()
                    .assign(K=lambda x: _pd.to_numeric(x["K"], errors="coerce"),
                            V=lambda x: _pd.to_numeric(x[ng_col], errors="coerce"))
                    .dropna()
                    .groupby("K", as_index=False)["V"].sum()
                    .sort_values("K")
                )
                if not g.empty and g["V"].min() < 0 and g["V"].max() > 0:
                    Ks = g["K"].to_numpy(dtype=float)
                    Vs = g["V"].to_numpy(dtype=float)
                    idx = _np.where(_np.sign(Vs[:-1]) * _np.sign(Vs[1:]) < 0)[0]
                    if idx.size:
                        overlay_levels.append(float(Ks[idx[0]]))

            ys = sorted(set([float(y) for y in overlay_levels if _np.isfinite(y)]))
            for y in ys:
                fig.add_hline(y=y, line=dict(color="white", width=1, dash="dot"), layer="above", opacity=0.9)
        except Exception:
            pass
return fig
