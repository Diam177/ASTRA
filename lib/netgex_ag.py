
"""
netgex_ag.py — расчёт Net GEX и AG из "исправленных" данных (df_corr), опциональная агрегация.
Единицы: $ на 1% движения (и млн $ при scale=1e6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np
import pandas as pd

@dataclass
class NetGEXAGConfig:
    scale: float = 1e6       # млн $ на 1% движения
    aggregate: str = "none"  # 'none' | 'sum' | 'sum_union'

def _ensure_required_columns(df: pd.DataFrame) -> None:
    req = {"exp","K","side","oi","gamma_corr","S","mult"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"df_corr is missing columns: {sorted(miss)}")

def _window_strikes_for_exp(df_corr: pd.DataFrame, exp: str, windows: Optional[Dict[str, np.ndarray]]):
    Ks = np.array(sorted(df_corr.loc[df_corr["exp"]==exp, "K"].unique()), dtype=float)
    if windows is None: return Ks
    idx = windows.get(exp)
    if idx is None or len(idx)==0: return Ks
    idx = np.array(idx, dtype=int)
    idx = idx[(idx>=0)&(idx<len(Ks))]
    if len(idx)==0: return Ks
    return Ks[idx]

def compute_netgex_ag_per_expiry(df_corr: pd.DataFrame, exp: str,
                                 windows: Optional[Dict[str, np.ndarray]]=None,
                                 cfg: NetGEXAGConfig=NetGEXAGConfig()) -> pd.DataFrame:
    _ensure_required_columns(df_corr)
    g = df_corr.loc[df_corr["exp"]==exp].copy()
    if g.empty:
        return pd.DataFrame(columns=["exp","K","S","call_oi","put_oi","dg1pct_call","dg1pct_put","AG_1pct","NetGEX_1pct","AG_1pct_M","NetGEX_1pct_M"])

    Ks_keep = set(_window_strikes_for_exp(df_corr, exp, windows))

    dg_row = np.where(
        np.isfinite(g["gamma_corr"]) & (g["gamma_corr"]>0) &
        np.isfinite(g["S"]) & np.isfinite(g["mult"]) &
        np.isfinite(g["oi"]) & (g["oi"]>0),
        g["gamma_corr"].to_numpy() * (g["S"].to_numpy()**2) * 0.01 * g["mult"].to_numpy() * g["oi"].to_numpy(),
        0.0
    )
    g = g.assign(dg1pct_row=dg_row)
    agg = g.groupby(["K","side"], as_index=False).agg(oi_side=("oi","sum"), dg1pct_side=("dg1pct_row","sum"))
    pivot = agg.pivot_table(index="K", columns="side", values=["oi_side","dg1pct_side"], aggfunc="sum").fillna(0.0)
    pivot.columns = [f"{v}_{s.lower()}" for v,s in pivot.columns]
    pivot = pivot.reset_index()

    S_exp = float(np.nanmedian(g["S"].values)); pivot["S"]=S_exp
    pivot = pivot.rename(columns={
        "oi_side_c":"call_oi","oi_side_p":"put_oi",
        "dg1pct_side_c":"dg1pct_call","dg1pct_side_p":"dg1pct_put",
    })

    if Ks_keep: pivot = pivot[pivot["K"].isin(Ks_keep)]

    pivot["AG_1pct"]     = pivot["dg1pct_call"] + pivot["dg1pct_put"]
    pivot["NetGEX_1pct"] = pivot["dg1pct_call"] - pivot["dg1pct_put"]

    if cfg.scale and cfg.scale>0:
        pivot["AG_1pct_M"]     = pivot["AG_1pct"]/cfg.scale
        pivot["NetGEX_1pct_M"] = pivot["NetGEX_1pct"]/cfg.scale

    pivot.insert(0,"exp",exp)
    cols=["exp","K","S"]  + ["call_oi","put_oi","dg1pct_call","dg1pct_put","AG_1pct","NetGEX_1pct"]
    if "AG_1pct_M" in pivot.columns: cols += ["AG_1pct_M","NetGEX_1pct_M"]
    pivot = pivot[cols].sort_values("K").reset_index(drop=True)
    return pivot

def compute_netgex_ag(df_corr: pd.DataFrame, windows: Optional[Dict[str, np.ndarray]]=None,
                      cfg: NetGEXAGConfig=NetGEXAGConfig()):
    _ensure_required_columns(df_corr)
    results = {}
    for exp in sorted(df_corr["exp"].dropna().unique()):
        results[exp] = compute_netgex_ag_per_expiry(df_corr, exp, windows=windows, cfg=cfg)
    if cfg.aggregate=="none":
        return results

    frames = []
    for exp, df in results.items():
        keep = ["K","S","AG_1pct","NetGEX_1pct"]
        df2 = df[keep].copy(); df2["exp"]=exp
        frames.append(df2)

    if not frames:
        return pd.DataFrame(columns=["K","S","AG_1pct","NetGEX_1pct","AG_1pct_M","NetGEX_1pct_M"])

    if cfg.aggregate=="sum":
        common_K = set(frames[0]["K"].unique())
        for df in frames[1:]: common_K &= set(df["K"].unique())
        common_K = sorted(common_K)
        if not common_K:
            return pd.DataFrame(columns=["K","S","AG_1pct","NetGEX_1pct","AG_1pct_M","NetGEX_1pct_M"])
        base = pd.DataFrame({"K": common_K})
        base["S"] = float(np.nanmedian(pd.concat(frames)["S"].values))
        base["AG_1pct"]=0.0; base["NetGEX_1pct"]=0.0
        for df in frames:
            m = df[df["K"].isin(common_K)].groupby("K")[["AG_1pct","NetGEX_1pct"]].sum()
            base = base.merge(m, left_on="K", right_index=True, how="left", suffixes=("","_add")).fillna(0.0)
            base["AG_1pct"] += base.pop("AG_1pct_add")
            base["NetGEX_1pct"] += base.pop("NetGEX_1pct_add")
        if cfg.scale and cfg.scale>0:
            base["AG_1pct_M"] = base["AG_1pct"]/cfg.scale
            base["NetGEX_1pct_M"] = base["NetGEX_1pct"]/cfg.scale
        cols=["K","S"]  + ["AG_1pct","NetGEX_1pct"]
        if "AG_1pct_M" in base.columns: cols+=["AG_1pct_M","NetGEX_1pct_M"]
        return base[cols].sort_values("K").reset_index(drop=True)

    if cfg.aggregate=="sum_union":
        all_K = sorted(set().union(*[set(df["K"].unique()) for df in frames]))
        base = pd.DataFrame({"K": all_K})
        base["S"] = float(np.nanmedian(pd.concat(frames)["S"].values))
        base["AG_1pct"]=0.0; base["NetGEX_1pct"]=0.0
        for df in frames:
            m = df.groupby("K")[["AG_1pct","NetGEX_1pct"]].sum()
            base = base.merge(m, left_on="K", right_index=True, how="left", suffixes=("","_add")).fillna(0.0)
            base["AG_1pct"] += base.pop("AG_1pct_add")
            base["NetGEX_1pct"] += base.pop("NetGEX_1pct_add")
        if cfg.scale and cfg.scale>0:
            base["AG_1pct_M"] = base["AG_1pct"]/cfg.scale
            base["NetGEX_1pct_M"] = base["NetGEX_1pct"]/cfg.scale
        cols=["K","S"]  + ["AG_1pct","NetGEX_1pct"]
        if "AG_1pct_M" in base.columns: cols+=["AG_1pct_M","NetGEX_1pct_M"]
        return base[cols].sort_values("K").reset_index(drop=True)

    raise ValueError(f"Unknown aggregate mode: {cfg.aggregate!r}")
