
# -*- coding: utf-8 -*-
from __future__ import annotations

# --- Heatmap debug stub (avoids NameError at import-time) ---
if "_render_level_strength_heatmap" not in globals():
    import streamlit as st, traceback
    def _render_level_strength_heatmap(*args, **kwargs):
        try:
            st.warning("Heatmap debug: helper not yet defined; using stub")
            st.write({"args_types":[type(a).__name__ for a in args],
                      "kwargs_keys": list(kwargs.keys())})
            st.code("".join(traceback.format_stack(limit=6)))
        except Exception:
            pass




# --- Level Strength Heatmap helper (override stub) ---
def _render_level_strength_heatmap(df_final, price_df, gflip, spot_value):
    import pandas as pd, streamlit as st
    try:
        from lib.heatmap import compute_scores, build_heatmap
    except Exception as e:
        st.error(f"Heatmap import error: {e}")
        return
    try:
        if df_final is None or getattr(df_final, "empty", True):
            st.info("Heatmap: df_final is empty")
            return
        if "K" not in df_final.columns:
            st.info("Heatmap: column 'K' not found")
            return
        factors = {}
        if "AG_1pct_M" in df_final.columns:
            factors["AG"] = pd.to_numeric(df_final["AG_1pct_M"], errors="coerce").abs()
        elif "AG_1pct" in df_final.columns:
            factors["AG"] = pd.to_numeric(df_final["AG_1pct"], errors="coerce").abs()
        if {"call_oi","put_oi"}.issubset(df_final.columns):
            factors["OI"] = pd.to_numeric(df_final["call_oi"], errors="coerce").clip(lower=0) +                             pd.to_numeric(df_final["put_oi"], errors="coerce").clip(lower=0)
        if {"call_vol","put_vol"}.issubset(df_final.columns):
            factors["VOL"] = pd.to_numeric(df_final["call_vol"], errors="coerce").clip(lower=0) +                              pd.to_numeric(df_final["put_vol"], errors="coerce").clip(lower=0)
        if "PZ" in df_final.columns:
            factors["PZ"] = pd.to_numeric(df_final["PZ"], errors="coerce").clip(lower=0)
        if gflip is not None and "K" in df_final.columns:
            try:
                k_series = pd.to_numeric(df_final["K"], errors="coerce")
                k_near = float(k_series.iloc[(k_series - float(gflip)).abs().idxmin()])
                g = pd.Series(0.0, index=df_final.index)
                g.loc[k_series == k_near] = 1.0
                factors["GFLIP"] = g
            except Exception:
                pass
        S_val = None
        try:
            S_val = float(spot_value) if spot_value is not None else None
        except Exception:
            S_val = None
        if S_val is None and price_df_used is not None:
            try:
                S_val = float(pd.to_numeric(price_df_used["price"], errors="coerce").dropna().iloc[-1])
            except Exception:
                S_val = None
        if S_val is None or not factors:
            st.info("Heatmap: no spot or no factors")
            return
        flip_side = None
        for cand in ("NetGEX_1pct_M", "NetGEX_1pct"):
            if cand in df_final.columns:
                try:
                    flip_side = "N" if pd.to_numeric(df_final[cand], errors="coerce").mean() < 0 else "P"
                except Exception:
                    flip_side = None
                break
        # --- normalize price_df to required schema ['timestamp','price'] ---
        ps_norm = None
        if price_df is not None and getattr(price_df, "empty", False) is False:
            ps = price_df.copy()
            # map lowercase for detection
            lmap = {c.lower(): c for c in ps.columns}
            # choose time column
            if "timestamp" in lmap:
                ts = pd.to_datetime(ps[lmap["timestamp"]])
            elif "time" in lmap:
                ts = pd.to_datetime(ps[lmap["time"]])
            elif "datetime" in lmap:
                ts = pd.to_datetime(ps[lmap["datetime"]])
            elif "t" in lmap:
                ts = pd.to_datetime(pd.to_numeric(ps[lmap["t"]], errors="coerce"), unit="ms", utc=True)
            else:
                ts = None
            # choose price column
            price_candidates = ["price", "close", "c", "last", "p", "vw"]
            pr = None
            for cand in price_candidates:
                if cand in lmap:
                    pr = pd.to_numeric(ps[lmap[cand]], errors="coerce")
                    break
            if ts is not None and pr is not None:
                ps_norm = pd.DataFrame({"timestamp": ts, "price": pr}).dropna()
        # fallback: if normalization failed, skip overlay to avoid exceptions
        if ps_norm is None or ps_norm.empty:
            price_df_used = None
        else:
            price_df_used = ps_norm

        levels_df = compute_scores(df_final["K"], factors, spot=S_val, flip_side=flip_side, norm="p90")
        levels_df["label"] = None
        st.markdown("### Level Strength Heatmap")
try:
    fig_hm = build_heatmap(
        levels_df.rename(columns={"price":"price","score":"score"}),
        price_series=price_df_used,
        title=None,
        overlay_source_df=df_final,
    )
except TypeError as _sig_e:
    if "overlay_source_df" in str(_sig_e):
        fig_hm = build_heatmap(
            levels_df.rename(columns={"price":"price","score":"score"}),
            price_series=price_df_used,
            title=None,
        )
    else:
        raise
st.plotly_chart(fig_hm, use_container_width=True)

    except Exception as _hm_e:
        st.error(f"Heatmap exception: {_hm_e.__class__.__name__}: {_hm_e}")

import os
import json
import requests
from typing import Any, Dict, List, Tuple

import streamlit as st



# Advanced metrics block
try:
    from advanced_analysis_block import render_advanced_analysis_block
except Exception:
    render_advanced_analysis_block = None
# --- Intraday price loader for Key Levels (Polygon v2 aggs 1-minute) ---
def _load_session_price_df_for_key_levels(ticker: str, session_date_str: str, api_key: str, timeout: int = 30):
    import pandas as pd
    import pytz
    t_raw = (ticker or '').strip()
    t = t_raw.upper()
    if t in {'SPX','NDX','VIX','RUT','DJX'} and not t_raw.startswith('I:'):
        t = f'I:{t}'
    if not t or not session_date_str:
        return None
    base = "https://api.polygon.io"
    url = f"{base}/v2/aggs/ticker/{t}/range/1/minute/{session_date_str}/{session_date_str}?adjusted=true&sort=asc&limit=50000"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        js = r.json()
    except Exception:
        return None
    res = js.get("results") or []
    if not res:
        return None
    tz = pytz.timezone("America/New_York")
    # Build dataframe
    times   = [pd.to_datetime(x.get("t"), unit="ms", utc=True).tz_convert(tz) for x in res]
    price   = [x.get("c") for x in res]
    volume  = [x.get("v") for x in res]
    vwap_api= [x.get("vw") for x in res]
    opens   = [x.get("o") for x in res]
    highs   = [x.get("h") for x in res]
    lows    = [x.get("l") for x in res]
    closes  = [x.get("c") for x in res]
    df = pd.DataFrame({
        "time":   times,
        "price":  pd.to_numeric(price,  errors="coerce"),
        "volume": pd.to_numeric(volume, errors="coerce"),
    })
    # Add OHLC columns (used by candlesticks)
    try:
        df["open"]  = pd.to_numeric(opens,  errors="coerce")
        df["high"]  = pd.to_numeric(highs,  errors="coerce")
        df["low"]   = pd.to_numeric(lows,   errors="coerce")
        df["close"] = pd.to_numeric(closes, errors="coerce")
    except Exception:
        pass
    
    # --- VWAP logic with SPX proxy ---
    # expose per-bar 'vw' for indices
    try:
        df["vw"] = pd.to_numeric(vwap_api, errors="coerce")
    except Exception:
        pass

    total_vol = float(df["volume"].fillna(0.0).sum())
    if "vw" in df.columns and df["vw"].notna().any() and total_vol == 0.0:
        # index case: use Polygon per-bar 'vw' directly
        df["vwap"] = df["vw"]
    elif any(v is not None for v in vwap_api) and total_vol > 0.0:
        vw = pd.Series(pd.to_numeric(vwap_api, errors="coerce"), index=df.index)
        vol = df["volume"].fillna(0.0)
        cum_vol = vol.cumsum().replace(0, pd.NA)
        df["vwap"] = (vw * vol).cumsum() / cum_vol
    else:
        df["vwap"] = pd.NA

    # SPY proxy only for SPX if vwap still NaN
    t_up = (ticker or "").upper()
    if t_up in {"SPX","^SPX","I:SPX"} and (df["vwap"].isna().all() or not df["vwap"].notna().any()):
        try:
            base = "https://api.polygon.io"
            url_spy = (f"{base}/v2/aggs/ticker/SPY/range/1/minute/"
                       f"{session_date_str}/{session_date_str}"
                       f"?adjusted=true&sort=asc&limit=50000&apiKey={api_key}")
            r2 = requests.get(url_spy, timeout=timeout)
            r2.raise_for_status()
            js2 = r2.json() or {}
            res2 = js2.get("results") or []
            if res2:
                import pandas as pd, numpy as np, pytz
                tz = pytz.timezone("America/New_York")
                spy_time = pd.to_datetime([x.get("t") for x in res2], unit="ms", utc=True).tz_convert(tz)
                spy_df = pd.DataFrame({
                    "time": spy_time,
                    "spy_close": pd.Series(pd.to_numeric([x.get("c") for x in res2], errors="coerce")),
                    "spy_vol":   pd.Series(pd.to_numeric([x.get("v") for x in res2], errors="coerce")).fillna(0.0),
                    "spy_vw":    pd.Series(pd.to_numeric([x.get("vw") for x in res2], errors="coerce")),
                }).sort_values("time")
                # RTH filter
                spy_df = spy_df.set_index("time").between_time("09:30", "16:00").reset_index()
                px = spy_df["spy_vw"].where(spy_df["spy_vw"].notna(), spy_df["spy_close"])
                cumv = spy_df["spy_vol"].cumsum().replace(0, pd.NA)
                spy_df["spy_vwap"] = (px.mul(spy_df["spy_vol"])).cumsum() / cumv
                spy_df["minute"] = spy_df["time"].dt.floor("min")

                # SPX minutes in ET
                df_et = df.copy()
                try:
                    df_et["time"] = df_et["time"].dt.tz_convert(tz)
                except Exception:
                    df_et["time"] = pd.to_datetime(df_et["time"], utc=True).dt.tz_convert(tz)
                df_et["minute"] = df_et["time"].dt.floor("min")

                merged = pd.merge(df_et[["minute","price"]].rename(columns={"price":"spx_price"}),
                                  spy_df[["minute","spy_close"]], on="minute", how="inner").sort_values("minute")
                if not merged.empty and merged["spy_close"].iloc[0] and merged["spx_price"].iloc[0]:
                    alpha0 = float(merged["spx_price"].iloc[0]) / float(merged["spy_close"].iloc[0])
                else:
                    alpha0 = 10.0

                ratio_map = (merged.set_index("minute")["spx_price"] / merged.set_index("minute")["spy_close"]).dropna()
                minutes = spy_df["minute"].drop_duplicates().sort_values().reset_index(drop=True)
                alphas = []
                last_alpha = alpha0
                session_start = minutes.iloc[0] if len(minutes) else None
                for m in minutes:
                    w_start = m - pd.Timedelta(minutes=15)
                    r_win = ratio_map[(ratio_map.index > w_start) & (ratio_map.index <= m)]
                    if len(r_win) >= 5:
                        q1, q99 = r_win.quantile(0.01), r_win.quantile(0.99)
                        raw_alpha = r_win.clip(q1, q99).median()
                    else:
                        raw_alpha = last_alpha
                    if session_start is not None and (m - session_start) <= pd.Timedelta(minutes=30):
                        low = 0.98 * alpha0; high = 1.02 * alpha0
                        raw_alpha = min(max(raw_alpha, low), high)
                    delta = raw_alpha - last_alpha
                    max_step = 0.005 * last_alpha
                    if abs(delta) > abs(max_step):
                        raw_alpha = last_alpha + (max_step if delta > 0 else -max_step)
                    alphas.append(raw_alpha)
                    last_alpha = raw_alpha
                alpha_series = pd.Series(alphas, index=minutes, name="alpha")

                alpha_aligned = pd.merge(df_et[["minute"]], alpha_series, left_on="minute", right_index=True, how="left")["alpha"].fillna(method="ffill")
                spy_vwap_aligned = pd.merge(df_et[["minute"]], spy_df[["minute","spy_vwap"]], on="minute", how="left")["spy_vwap"].fillna(method="ffill")
                proxy_vwap = alpha_aligned.values * spy_vwap_aligned.values

                df["vwap"] = df["vwap"].where(df["vwap"].notna(), proxy_vwap)
        except Exception as e:
            try:
                import streamlit as st
                code = getattr(getattr(e, "response", None), "status_code", None)
                st.warning(f"SPY proxy VWAP недоступен ({code}) {type(e).__name__}: {e}. Использую TWAP.", icon="⚠️")
            except Exception:
                pass

    # Final fallback: TWAP
    if df["vwap"].isna().all():
        pr = df["price"].fillna(method="ffill")
        df["vwap"] = pr.expanding().mean()
    return df

# --- Helpers to hide tables from main page ---
def _st_hide_df(*args, **kwargs):
    # no-op: we suppress table rendering on main page per requirements
    return None
def _st_hide_subheader(*args, **kwargs):
    # no-op: suppress section headers for tables
    return None
from lib.netgex_chart import render_netgex_bars, _compute_gamma_flip_from_table
from lib.key_levels import render_key_levels

# Project imports
from lib.sanitize_window import sanitize_and_window_pipeline
from lib.tiker_data import (
    list_future_expirations,
    download_snapshot_json,
    get_spot_snapshot,
    PolygonError,
)

st.set_page_config(page_title="GammaStrat — df_raw", layout="wide")

# --- Helpers -----------------------------------------------------------------
def _coerce_results(data: Any) -> List[Dict]:
    """
    Приводит разные варианты JSON к list[dict] записей опционов.
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("results", "options", "data"):
            v = data.get(key)
            if isinstance(v, list):
                return v
    return []


def _infer_spot_from_snapshot(raw: List[Dict]) -> float | None:
    """
    Fallback-оценка S из снимка опционов:
    - Берём страйки и |delta| близкие к 0.5 (окно 0.2..0.8, вес 1/| |delta|-0.5 |)
    """
    num = 0.0
    den = 0.0

    def get_nested(d: Dict, keys: list[str], blocks=("details","greeks","day","underlying_asset")):
        for k in keys:
            if k in d:
                return d[k]
        for b in blocks:
            sub = d.get(b, {})
            if isinstance(sub, dict):
                for k in keys:
                    if k in sub:
                        return sub[k]
        return None

    for r in raw:
        K = get_nested(r, ["strike","k","strike_price","strikePrice"])
        dlt = get_nested(r, ["delta","dlt"])
        if K is None or dlt is None:
            continue
        try:
            K = float(K); dlt = float(dlt)
        except Exception:
            continue
        if 0.2 <= abs(dlt) <= 0.8 and K > 0:
            w = 1.0 / (abs(abs(dlt) - 0.5) + 1e-6)
            num += w * K
            den += w
    if den > 0:
        return num / den
    return None


def _get_api_key() -> str | None:
    key = None
    try:
        key = st.secrets.get("POLYGON_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        key = None
    if not key:
        key = os.getenv("POLYGON_API_KEY")
    return key


# --- UI: Controls -------------------------------------------------------------

api_key = _get_api_key()
if not api_key:
    st.error("POLYGON_API_KEY is not set in Streamlit Secrets or environment variables.")
    st.stop()



# --- Input state helpers ------------------------------------------------------
if "ticker" not in st.session_state:
    st.session_state["ticker"] = "SPY"

def _normalize_ticker():
    t = st.session_state.get("ticker", "")
    st.session_state["ticker"] = (t or "").strip().upper()

# --- Controls moved to sidebar ----------------------------------------------
with st.sidebar:
    st.image("logogamma.png", use_column_width=True)
    st.text_input("Ticker", key="ticker", on_change=_normalize_ticker)
    ticker = st.session_state.get("ticker", "")

    # Получаем список будущих экспираций под выбранный тикер
    expirations: list[str] = st.session_state.get(f"expirations:{ticker}", [])
    if not expirations and ticker:
        try:
            expirations = list_future_expirations(ticker, api_key)
            st.session_state[f"expirations:{ticker}"] = expirations
        except Exception as e:
            st.error(f"Unable to retrieve expiration dates: {e}")
            expirations = []

    if expirations:
        # по умолчанию ближайшая дата — первая в списке
        default_idx = 0
        sel = st.selectbox("Expiration date", options=expirations, index=default_idx, key=f"exp_sel:{ticker}")
        expiration = sel

        # --- Search trigger below expiration selection ---
        search_clicked = st.button("Search", key="btn_search", type="primary", use_container_width=True)
        if search_clicked:
            try:
                _normalize_ticker()
            except Exception:
                pass

        # --- Режим агрегации экспираций ---
        mode_exp = st.radio("Expiration mode", ["Single","Multi"], index=0, horizontal=True)
        selected_exps = []
        weight_mode = "equal"
        if mode_exp == "Single":
            selected_exps = [expiration]
        else:
            selected_exps = st.multiselect("Select expiration", options=expirations, default=expirations[:2])
            weight_mode = st.selectbox("Weighing", ["equal","1/T","1/√T"], index=2)
    else:
        expiration = ""
        st.warning("There are no expiration dates available for the ticker.")



# --- Empty state guard for Multi with no selected expirations ---
try:
    if ('mode_exp' in locals()) and (mode_exp == 'Multi') and (not selected_exps):
        st.info('Select expiration date')
        st.stop()
except Exception:
    pass

# --- Data fetch ---------------------------------------------------------------
raw_records: List[Dict] | None = None
snapshot_js: Dict | None = None

if ticker and expiration:
    try:
        snapshot_js = download_snapshot_json(ticker, expiration, api_key)
        raw_records = _coerce_results(snapshot_js)
        st.session_state["raw_records"] = raw_records
    except PolygonError as e:
        st.error(f"Error Polygon: {e}")
    except Exception as e:
        st.error(f"Error loading snapshot JSON: {e}")

# --- Sidebar: download raw provider JSON -------------------------------------
if snapshot_js:
    try:
        raw_bytes = json.dumps(snapshot_js, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        fname = f"{ticker}_{expiration}.json"
        st.sidebar.download_button(
            label="Download JSON",
            data=raw_bytes,
            file_name=fname,
            mime="application/json",
            use_container_width=True,
        )
    except Exception as e:
        st.sidebar.error(f"Failed to prepare JSON for download: {e}")
else:
    st.sidebar.info("Select the ticker and expiration date to download the snapshot and JSON.")

# --- Download tables button placeholder (below raw JSON) ---------------------
dl_tables_container = st.sidebar.empty()

# --- Spot price ---------------------------------------------------------------
S: float | None = None
if ticker:
    try:
        S = get_spot_snapshot(ticker, api_key)
        # snapshot-only spot
    except Exception as e:
        import streamlit as st
        # Graceful handling: invalid ticker or network error
        status = getattr(getattr(e, 'response', None), 'status_code', None)
        if status == 404:
            st.warning(f"Incorrect ticker: {ticker}. Check your spelling.")
            st.stop()
        else:
            st.error("Unable to retrieve spot price. Please check your connection or ticker.")
            st.stop()
# --- Run sanitize/window + show df_raw ---------------------------------------
if raw_records:
    try:
        res = sanitize_and_window_pipeline(raw_records, S=S)

        # --- TRUE MULTI-EXP PROCESSING: прогон по каждой выбранной экспирации и объединение ---
        try:
            if ("mode_exp" in locals()) and (mode_exp == "Multi") and selected_exps:

                # --- QA Sidebar setup for Multi ---
                qa_mode = st.sidebar.toggle("QA mode Multi", value=True)
                ok_exps, fail_exps = [], []
                _qa_rows = []
                st.sidebar.caption("Multi Diagnostics")
                try:
                    st.sidebar.write(f"Selected dates: **{len(selected_exps)}**")
                except Exception:
                    st.sidebar.write("Selected dates: ?")

                import pandas as pd
                df_corr_multi_list = []
                windows_multi = {}

                for _exp in selected_exps:
                    try:
                        js_e = download_snapshot_json(ticker, _exp, api_key)
                        recs_e = _coerce_results(js_e)
                        res_e = sanitize_and_window_pipeline(recs_e, S=S)
                        dfe = res_e.get("df_corr")
                        if dfe is not None and not getattr(dfe, "empty", True):
                            df_corr_multi_list.append(dfe)
                        win_e = res_e.get("windows") or {}

                        # QA: collect per-exp stats
                        try:
                            ok_exps.append(_exp)
                            rows_cnt = int(len(dfe)) if dfe is not None else 0
                            k_cnt = 0
                            k_min = float("nan"); k_max = float("nan")
                            dfw_e = res_e.get("df_weights")
                            if isinstance(dfw_e, pd.DataFrame) and not dfw_e.empty and isinstance(win_e, dict):
                                try:
                                    gW = dfw_e[dfw_e.get("exp")==_exp].sort_values("K").reset_index(drop=True)
                                    idxs = list(win_e.get(_exp, win_e.get("window", []))) if hasattr(win_e, "get") else []
                                    ks = []
                                    for ii in idxs:
                                        ii = int(ii)
                                        if 0 <= ii < len(gW):
                                            ks.append(float(gW.loc[ii, "K"]))
                                    if ks:
                                        k_cnt = len(ks)
                                        k_min = min(ks); k_max = max(ks)
                                except Exception:
                                    pass
                            _qa_rows.append({"exp": _exp, "status": "OK", "rows_df_corr": rows_cnt, "K_in_window": k_cnt, "K_min": k_min, "K_max": k_max})
                        except Exception:
                            pass

                        for k, v in win_e.items():
                            windows_multi[k] = v
                    except Exception as _exc:

                        # QA: record failure
                        try:
                            fail_exps.append((_exp, f"{type(_exc).__name__}: {_exc}"))
                            if qa_mode:
                                st.sidebar.error(f"{_exp}: {type(_exc).__name__}: {_exc}")
                        except Exception:
                            pass

                        # мягко пропускаем проблемные экспирации, чтобы не ломать UI
                        pass

                if df_corr_multi_list:
                    df_corr = pd.concat(df_corr_multi_list, ignore_index=True)
                if windows_multi:
                    windows = windows_multi
                    # --- Соберём промежуточные таблицы по каждой экспирации для ZIP-скачивания ---

                # QA: sidebar summary after per-exp processing
                try:
                    st.sidebar.write(f"Dates collected: **{len(ok_exps)}**")
                    if fail_exps:
                        st.sidebar.write("Issues: " + "; ".join(e for e, _ in fail_exps))
                    if _qa_rows:
                        try:
                            import pandas as _pd  # alias to avoid shadowing
                        except Exception:
                            pass
                        try:
                            _df_qa = pd.DataFrame(_qa_rows).set_index("exp")
                            st.sidebar.dataframe(_df_qa, use_container_width=True, height=220)
                        except Exception:
                            st.sidebar.write(_qa_rows)
                except Exception:
                    pass

                    try:
                        import io, zipfile
                        import pandas as pd

                        multi_exports = {}  # exp -> {name: DataFrame}
                        for _exp in selected_exps:
                            try:
                                js_e = download_snapshot_json(ticker, _exp, api_key)
                                recs_e = _coerce_results(js_e)
                                res_e = sanitize_and_window_pipeline(recs_e, S=S)

                                df_raw_e    = res_e.get("df_raw")
                                df_marked_e = res_e.get("df_marked")
                                df_corr_e   = res_e.get("df_corr")
                                df_weights_e= res_e.get("df_weights")
                                windows_e   = res_e.get("windows") or {}
                                window_raw_e= res_e.get("window_raw")
                                window_corr_e=res_e.get("window_corr")

                                # windows (табличный вид) для этой экспирации
                                df_windows_e = None
                                try:
                                    if windows_e and isinstance(windows_e, dict) and df_weights_e is not None:
                                        rows_w = []
                                        idxs = windows_e.get(_exp, [])
                                        if hasattr(df_weights_e, "empty") and not df_weights_e.empty:
                                            gW = df_weights_e[df_weights_e["exp"] == _exp].sort_values("K").reset_index(drop=True)
                                            for i in list(idxs):
                                                ii = int(i)
                                                if 0 <= ii < len(gW):
                                                    rows_w.append({
                                                        "exp": _exp,
                                                        "row_index": ii,
                                                        "K": float(gW.loc[ii, "K"]),
                                                        "w_blend": float(gW.loc[ii, "w_blend"]),
                                                    })
                                        if rows_w:
                                            df_windows_e = pd.DataFrame(rows_w, columns=["exp","row_index","K","w_blend"]).sort_values(["exp","K"])
                                except Exception:
                                    df_windows_e = None

                                multi_exports[_exp] = {
                                    "df_raw": df_raw_e,
                                    "df_marked": df_marked_e,
                                    "df_corr": df_corr_e,
                                    "df_weights": df_weights_e,
                                    "windows_table": df_windows_e,
                                    "window_raw": window_raw_e,
                                    "window_corr": window_corr_e,
                                }
                            except Exception:
                                # пропускаем проблемные даты, ZIP соберётся из удачных
                                pass

                        # Кнопка скачивания ZIP (только если есть хоть что-то)
                        
                        def _zip_multi_intermediate(exports: dict, final_sum_df=None) -> io.BytesIO:
                            bio = io.BytesIO()
                            with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                                # write per-exp intermediates
                                for exp_key, tbls in (exports or {}).items():
                                    for name, df in (tbls or {}).items():
                                        if df is None:
                                            continue
                                        try:
                                            if hasattr(df, "empty") and df.empty:
                                                continue
                                        except Exception:
                                            pass
                                        try:
                                            csv_bytes = df.to_csv(index=False).encode("utf-8")
                                        except Exception:
                                            csv_bytes = str(df).encode("utf-8")
                                        zf.writestr(f"{exp_key}/{name}.csv", csv_bytes)
                                # write per-exp finals
                                try:
                                    from lib.final_table import build_final_tables_from_corr, FinalTableConfig
                                    finals = build_final_tables_from_corr(df_corr, windows, cfg=FinalTableConfig())
                                    for exp_key, fin in (finals or {}).items():
                                        if fin is None or getattr(fin, "empty", True):
                                            continue
                                        zf.writestr(f"{exp_key}/final_table.csv", fin.to_csv(index=False).encode("utf-8"))
                                except Exception:
                                    pass
                                # aggregated final table for Multi (used by chart)
                                try:
                                    if 'df_final_multi' in locals() and df_final_multi is not None and not getattr(df_final_multi, 'empty', True):
                                        zf.writestr("FINAL_SUM.csv", df_final_multi.to_csv(index=False).encode("utf-8"))
                                except Exception:
                                    pass
                                # aggregated final table (sum) used by chart
                                try:
                                    if final_sum_df is not None and not getattr(final_sum_df, 'empty', True):
                                        zf.writestr("FINAL_SUM.csv", final_sum_df.to_csv(index=False).encode("utf-8"))
                                except Exception:
                                    pass


                            bio.seek(0)
                            return bio
                        if any(tbl is not None and (not getattr(tbl, "empty", True)) 
                               for tbls in multi_exports.values() for tbl in (tbls or {}).values()):
                            zip_bytes = _zip_multi_intermediate(multi_exports, df_final_multi if 'df_final_multi' in locals() else None)
                            fname = f"{ticker}_intermediate_{len(multi_exports)}exps.zip" if ticker else "intermediate_tables.zip"
                            dl_tables_container.download_button("Download tables",
                                data=zip_bytes.getvalue(),
                                file_name=fname,
                                mime="application/zip",
                                use_container_width=True,
                            )
                    except Exception as _zip_err:
                        st.warning("Failed to prepare ZIP with staging tables.")
                        st.exception(_zip_err)

        except Exception as _e_multi:
            st.warning("Multi-exp: Failed to merge series.")
            st.exception(_e_multi)
        df_raw = res.get("df_raw")
        if df_raw is None or getattr(df_raw, "empty", True):
            st.warning("df_raw is empty. Check the data format.")
        else:
            _st_hide_df(df_raw, use_container_width=True)
            # --- Ниже показываем остальные массивы по степени создания ---
            # 1) df_marked (df_raw + флаги аномалий)
            df_marked = res.get("df_marked")
            if df_marked is not None and not getattr(df_marked, "empty", True):
                _st_hide_subheader()
                _st_hide_df(df_marked, use_container_width=True, hide_index=True)

            # 2) df_corr (восстановленные IV/Greeks)
            # Guard: do not overwrite multi df_corr
            if ("mode_exp" in locals()) and (mode_exp == "Multi"):
                df_corr_single = res.get("df_corr")
            else:
                df_corr = res.get("df_corr")
            if df_corr is not None and not getattr(df_corr, "empty", True):
                _st_hide_subheader()
                _st_hide_df(df_corr, use_container_width=True, hide_index=True)

            # 3) df_weights (веса окна по страйку)
            df_weights = res.get("df_weights")
            if df_weights is not None and not getattr(df_weights, "empty", True):
                _st_hide_subheader()
                _st_hide_df(df_weights, use_container_width=True, hide_index=True)

            # 4) windows (выбранные страйки каждого окна) — преобразуем в таблицу
            # Guard: do not overwrite multi windows
            if ("mode_exp" in locals()) and (mode_exp == "Multi"):
                windows_single = res.get("windows")
            else:
                windows = res.get("windows")
            if windows and isinstance(windows, dict) and df_weights is not None:
                try:
                    rows = []
                    for exp, idx in windows.items():
                        # df_weights на эту экспирацию, отсортируем по K как в select_windows
                        g = df_weights[df_weights["exp"] == exp].sort_values("K").reset_index(drop=True)
                        for i in list(idx):
                            if 0 <= int(i) < len(g):
                                rows.append({"exp": exp, "row_index": int(i), "K": float(g.loc[int(i), "K"]),
                                             "w_blend": float(g.loc[int(i), "w_blend"])})
                    import pandas as pd  # локальный импорт безопасен
                    df_windows = pd.DataFrame(rows, columns=["exp","row_index","K","w_blend"]).sort_values(["exp","K"])
                    _st_hide_subheader()
                    # Приведём тип w_blend к float64 и зададим формат — уберёт предупреждение 2^53 в браузере
                    try:
                        df_windows = df_windows.astype({'w_blend': 'float64'})
                        _st_hide_df(
                            df_windows,
                            use_container_width=True,
                            hide_index=True,
                            column_config={'w_blend': st.column_config.NumberColumn('w_blend', format='%.6f')},
                        )
                    except Exception:
                        _st_hide_df(df_windows, use_container_width=True, hide_index=True)
                
                except Exception as _e:
                    st.warning("Failed to display windows in table view.")
                    st.exception(_e)

            # 5) window_raw (строки окна из исходных + флаги)
            window_raw = res.get("window_raw")
            if window_raw is not None and not getattr(window_raw, "empty", True):
                _st_hide_subheader()
                _st_hide_df(window_raw, use_container_width=True, hide_index=True)

            # 6) window_corr (строки окна из исправленных данных)
            window_corr = res.get("window_corr")
            if window_corr is not None and not getattr(window_corr, "empty", True):
                _st_hide_subheader()
                _st_hide_df(window_corr, use_container_width=True, hide_index=True)

            if 'mode_exp' in locals() and mode_exp != 'Multi':
                # Кнопка скачивания ZIP со всеми таблицами (single)
                try:
                    import io, zipfile
                    from lib.final_table import build_final_tables_from_corr, FinalTableConfig
                    def _zip_single_tables(res_dict, df_corr_single, windows_single, exp_str):
                        bio = io.BytesIO()
                        with zipfile.ZipFile(bio, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
                            for name in ['df_raw','df_marked','df_corr','df_weights','window_raw','window_corr']:
                                df = res_dict.get(name)
                                if df is None:
                                    continue
                                try:
                                    if hasattr(df, 'empty') and df.empty:
                                        continue
                                except Exception:
                                    pass
                                try:
                                    csv_bytes = df.to_csv(index=False).encode('utf-8')
                                except Exception:
                                    csv_bytes = str(df).encode('utf-8')
                                zf.writestr(f"{exp_str}/{name}.csv", csv_bytes)
                            # финальная таблица
                            try:
                                finals = build_final_tables_from_corr(df_corr_single, windows_single, cfg=FinalTableConfig())
                                if exp_str in finals and finals[exp_str] is not None and not getattr(finals[exp_str], 'empty', True):
                                    zf.writestr(f"{exp_str}/final_table.csv", finals[exp_str].to_csv(index=False).encode('utf-8'))
                            except Exception:
                                pass
                        bio.seek(0)
                        return bio
                    exp_str = expiration if 'expiration' in locals() else 'exp'
                    zip_bytes = _zip_single_tables(res, df_corr, windows, exp_str)
                    dl_tables_container.download_button('Download tables', data=zip_bytes.getvalue(), file_name=(f"{ticker}_{exp_str}_tables.zip" if ticker else 'tables.zip'), mime='application/zip', use_container_width=True)
                except Exception as _e_zip_single:
                    st.warning('Failed to prepare ZIP with tables (single).')
                    st.exception(_e_zip_single)
            # 7) Финальная таблица (по df_corr + windows)
            try:
                from lib.final_table import build_final_tables_from_corr, FinalTableConfig, _series_ctx_from_corr, build_final_sum_from_corr
                from lib.netgex_ag import compute_netgex_ag_per_expiry, NetGEXAGConfig
                from lib.power_zone_er import compute_power_zone
                import numpy as np
                import pandas as pd
                import math

                final_cfg = FinalTableConfig()
                scale_val = getattr(final_cfg, "scale_millions", 1_000_000)

                if df_corr is not None and windows:

                    # --- MULTI режим: взвешенная сумма по нескольким экспирациям ---
                    if ("mode_exp" in locals()) and (mode_exp == "Multi") and selected_exps:
                        # 1) веса по T
                        exp_list = [e for e in selected_exps if e in df_corr["exp"].unique().tolist()]
                        if not exp_list:
                            raise ValueError("There are no expirations selected for Multi mode.")
                        t_map = {}
                        for e in exp_list:
                            T_vals = df_corr.loc[df_corr["exp"]==e, "T"].dropna().values
                            T_med = float(np.nanmedian(T_vals)) if T_vals.size else float("nan")
                            if not (math.isfinite(T_med) and T_med>0):
                                T_med = 1.0/252.0  # защита от T→0/NaN
                            t_map[e] = T_med
                        w_raw = {}
                        for e in exp_list:
                            if weight_mode == "1/T":
                                w_raw[e] = 1.0 / max(t_map[e], 1.0/252.0)
                            elif weight_mode == "1/√T":
                                w_raw[e] = 1.0 / math.sqrt(max(t_map[e], 1.0/252.0))
                            else:
                                w_raw[e] = 1.0
                        w_sum = sum(w_raw.values()) or 1.0
                        weights = {e: w_raw[e]/w_sum for e in exp_list}

                        
                        # --- Суммарная финальная таблица (Multi) через final_table.build_final_sum_from_corr ---
                        df_final_multi = build_final_sum_from_corr(
                            df_corr=df_corr,
                            windows=windows,
                            selected_exps=exp_list,
                            weight_mode=weight_mode,
                            cfg=final_cfg,
                            s_override=S,
                        )

                        # --- QA: Sidebar Multi diagnostics for aggregated table ---
                        try:
                            union_k = int(df_final_multi["K"].nunique()) if ("K" in df_final_multi.columns) else int(len(df_final_multi))
                        except Exception:
                            union_k = int(len(df_final_multi)) if df_final_multi is not None else 0
                        try:
                            df_multi_rows = int(len(df_final_multi)) if df_final_multi is not None else 0
                        except Exception:
                            df_multi_rows = 0
                        # weights diagnostics
                        try:
                            weights_sum = float(sum((weights or {}).values())) if isinstance(weights, dict) else float("nan")
                        except Exception:
                            weights_sum = float("nan")
                        # invariants
                        try:
                            need_cols = [c for c in ["AG_1pct","NetGEX_1pct","call_oi","put_oi","K"] if c in getattr(df_final_multi, "columns", [])]
                            no_nan_critical = bool(df_final_multi[need_cols].notna().all().all()) if need_cols else True
                            k_sorted_unique = bool(df_final_multi["K"].is_monotonic_increasing and df_final_multi["K"].is_unique) if "K" in getattr(df_final_multi, "columns", []) else True
                        except Exception:
                            no_nan_critical = False
                            k_sorted_unique = False
                        # sidebar render
                        try:
                            st.sidebar.markdown("### Multi QA")
                            st.sidebar.metric("Selected dates", len(selected_exps))
                            try:
                                st.sidebar.metric("Dates collected", len(ok_exps))
                            except Exception:
                                pass
                            st.sidebar.metric("Strikes in the Union", union_k)
                            st.sidebar.metric("Rows in multi", df_multi_rows)
                            st.sidebar.markdown("**Invariants**")
                            checks = [
                                ("Sum of weights ≈ 1", abs((weights_sum or 0.0) - 1.0) < 1e-9),
                                ("No NaN in key columns", no_nan_critical),
                                ("K are sorted and unique", k_sorted_unique),
                                ("Union is not empty", df_multi_rows > 0),
                            ]
                            for label, ok in checks:
                                st.sidebar.write(("✅ " if ok else "❌ ") + label)
                            # weights table
                            try:
                                import pandas as pd
                                _w_rows = [{"exp": e, "T": t_map.get(e), "w": weights.get(e)} for e in (exp_list or [])]
                                _wdf = pd.DataFrame(_w_rows)
                                with st.sidebar.expander("Weights"):
                                    st.dataframe(_wdf, use_container_width=True, height=200)
                            except Exception:
                                pass
                        except Exception:
                            pass

                        # Диагностика и кнопка скачивания
                        if df_final_multi is None or getattr(df_final_multi, "empty", True):
                            reason = []
                            if not exp_list:
                                reason.append("expirations not selected")
                            if not windows or not any(len(windows.get(e, [])) for e in exp_list):
                                reason.append("no windows built")
                            if df_final_multi is None or getattr(df_final_multi, "empty", True):
                                reason.append("no data on the combined K grid")
                            st.sidebar.info("The summary table is empty: " + ", ".join(reason))
                        else:
                            _st_hide_subheader()
                            _st_hide_df(df_final_multi, use_container_width=True, hide_index=True)
                            try:
                                st.sidebar.download_button(
                                    "Download table (Multi)",
                                    data=df_final_multi.to_csv(index=False).encode("utf-8"),
                                    file_name=(f"{ticker}_FINAL_SUM.csv" if 'ticker' in locals() and ticker else "FINAL_SUM.csv"),
                                    mime="text/csv",
                                )
                            except Exception as _dl_e:
                                st.sidebar.warning("Failed to generate CSV summary table.")
                                st.sidebar.exception(_dl_e)


                        _st_hide_subheader()
                        _st_hide_df(df_final_multi, use_container_width=True, hide_index=True)
                        # --- Net GEX chart (Multi: aggregated) ---
                        try:
                            st.markdown("### Net GEX")

                            render_netgex_bars(df_final=df_final_multi, ticker=ticker, spot=S if 'S' in locals() else None, toggle_key='netgex_multi')
                        except Exception as _chart_em:
                            st.error('Unable to display Net GEX chart (Multi)')
                            st.exception(_chart_em)

                        # --- Key Levels chart (Multi) ---
                        try:
                            import pandas as pd, pytz
                            # Подбор столбца NetGEX и spot для G-Flip
                            _ycol_m = "NetGEX_1pct_M" if ("NetGEX_1pct_M" in df_final_multi.columns) else ("NetGEX_1pct" if "NetGEX_1pct" in df_final_multi.columns else None)
                            _spot_m = float(pd.to_numeric(df_final_multi.get("S"), errors="coerce").median()) if ("S" in df_final_multi.columns) else None
                            _gflip_m = _compute_gamma_flip_from_table(df_final_multi, y_col=_ycol_m or "NetGEX_1pct", spot=_spot_m)
                            # Привязываем G-Flip к ближайшему доступному страйку из df_final_multi['K']
                            try:
                                _Ks_m = pd.to_numeric(df_final_multi.get("K"), errors="coerce").dropna().tolist() if ("K" in df_final_multi.columns) else []
                                if (_gflip_m is not None) and _Ks_m:
                                    _gflip_m = float(min(_Ks_m, key=lambda x: abs(float(x) - float(_gflip_m))))
                            except Exception:
                                pass
                            # Дата сессии и прайс‑лента для Key Levels
                            _session_date_str_m = pd.Timestamp.now(pytz.timezone("America/New_York")).strftime("%Y-%m-%d")
                            _price_df_m = _load_session_price_df_for_key_levels(ticker, _session_date_str_m, st.secrets.get("POLYGON_API_KEY", ""))
                            # Рендер
                            st.markdown("### Key Levels")

                            render_key_levels(df_final=df_final_multi, ticker=ticker, g_flip=_gflip_m, price_df=_price_df_m, session_date=_session_date_str_m, toggle_key="key_levels_multi")
                            try:
                                _render_level_strength_heatmap(df_final_multi, _price_df_m, _gflip_m, S if 'S' in locals() else None)
                            except Exception as e:
                                import streamlit as st, traceback
                                st.error(f"Heatmap exception: int: 49757")
                                st.code(traceback.format_exc())

                            # --- Advanced Analysis Block (Multi) — placed under Key Levels ---
                            try:
                                if render_advanced_analysis_block is not None:
                                    try:
                                        import pandas as pd, pytz
                                        _session_date_str_m = pd.Timestamp.now(pytz.timezone("America/New_York")).strftime("%Y-%m-%d")
                                        _price_df_m = _load_session_price_df_for_key_levels(
                                            ticker, _session_date_str_m, st.secrets.get("POLYGON_API_KEY", "")
                                        )
                                    except Exception:
                                        _price_df_m = None
                                    render_advanced_analysis_block(
                                        ticker=ticker,
                                        df_final=df_final_multi,
                                        df_corr=df_corr if 'df_corr' in locals() else None,
                                        S=S if 'S' in locals() else None,
                                        price_df=_price_df_m,
                                        selected_exps=selected_exps if 'selected_exps' in locals() else None,
                                        weight_mode=weight_mode if 'weight_mode' in locals() else "1/√T",
                                        caption_suffix="Aggregated by selected expirations."
                                    )
                            except Exception as _aabm_e:
                                st.warning("Advanced block (Multi) failed")
                                st.exception(_aabm_e)
                        except Exception as _klm_e:
                            st.error('Failed to display Key Levels chart (Multi)')
                            st.exception(_klm_e)


                    else:
                        # --- SINGLE режим: как было ---
                        final_tables = build_final_tables_from_corr(df_corr, windows, cfg=final_cfg)
                        exps = list(final_tables.keys())
                        if exps:
                            exp_to_show = expiration if 'expiration' in locals() and expiration in final_tables else exps[0]
                            df_final = final_tables.get(exp_to_show)
                            if df_final is not None and not getattr(df_final, "empty", True):
                                _st_hide_subheader()
                                _st_hide_df(df_final, use_container_width=True, hide_index=True)
                                # --- Net GEX chart (under the final table) ---
                                try:
                                    st.markdown("### Net GEX")
                                    render_netgex_bars(df_final=df_final, ticker=ticker, spot=S if 'S' in locals() else None, toggle_key='netgex_main')
                                except Exception as _chart_e:
                                    st.error('Unable to display Net GEX chart')
                                    st.exception(_chart_e)
                                # --- Key Levels chart (uses same final table & G-Flip from netgex_chart) ---
                                try:
                                    # Определяем столбец NetGEX и spot так же, как в netgex_chart
                                    _ycol = "NetGEX_1pct_M" if "NetGEX_1pct_M" in df_final.columns else ("NetGEX_1pct" if "NetGEX_1pct" in df_final.columns else None)
                                    _spot_for_flip = float(pd.to_numeric(df_final.get("S"), errors="coerce").median()) if "S" in df_final.columns else None
                                    _gflip_val = _compute_gamma_flip_from_table(df_final, y_col=_ycol or "NetGEX_1pct", spot=_spot_for_flip)
                                    
                                    # --- Snap G-Flip to nearest available strike from df_final['K'] (no math rounding) ---
                                    try:
                                        import pandas as pd
                                        _Ks = pd.to_numeric(df_final.get("K"), errors="coerce").dropna().tolist() if ("K" in df_final.columns) else []
                                        if (_gflip_val is not None) and _Ks:
                                            _gflip_val = float(min(_Ks, key=lambda x: abs(float(x) - float(_gflip_val))))
                                    except Exception:
                                        pass
                                    import pytz, pandas as pd
                                    _session_date_str = pd.Timestamp.now(pytz.timezone("America/New_York")).strftime("%Y-%m-%d")
                                    _price_df = _load_session_price_df_for_key_levels(ticker, _session_date_str, st.secrets.get("POLYGON_API_KEY", ""))
                                    st.markdown("### Key Levels")
                                    render_key_levels(df_final=df_final, ticker=ticker, g_flip=_gflip_val, price_df=_price_df, session_date=_session_date_str, toggle_key="key_levels_main")
                                    try:
                                        _render_level_strength_heatmap(df_final, _price_df, _gflip_val, S if 'S' in locals() else None)
                                    except Exception as e:
                                        import streamlit as st, traceback
                                        st.error(f"Heatmap exception: int: 49757")
                                        st.code(traceback.format_exc())

                                    # --- Advanced Analysis Block (Single) — placed under Key Levels ---
                                    try:
                                        if render_advanced_analysis_block is not None:
                                            try:
                                                import pandas as pd, pytz
                                                _session_date_str = pd.Timestamp.now(pytz.timezone("America/New_York")).strftime("%Y-%m-%d")
                                                _price_df = _load_session_price_df_for_key_levels(
                                                    ticker, _session_date_str, st.secrets.get("POLYGON_API_KEY", "")
                                                )
                                            except Exception:
                                                _price_df = None
                                            render_advanced_analysis_block(
                                                ticker=ticker,
                                                df_final=df_final,
                                                df_corr=df_corr if 'df_corr' in locals() else None,
                                                S=S if 'S' in locals() else None,
                                                price_df=_price_df,
                                                selected_exps=selected_exps if 'selected_exps' in locals() else None,
                                                weight_mode=weight_mode if 'weight_mode' in locals() else "1/√T",
                                                caption_suffix="Aggregated by the selected expiration."
                                            )
                                    except Exception as _aabs_e:
                                        st.warning("Advanced block (Single) failed")
                                        st.exception(_aabs_e)
                                except Exception as _kl_e:
                                    st.error('Failed to display Key Levels chart')
                                    st.exception(_kl_e)

                            else:
                                st.info("The final table is empty for the selected expiration.")
            except Exception as _e:
                st.error("Error constructing final table.")
                st.exception(_e)
    except Exception as e:
            st.error("Pipeline sanitize/window error.")
            st.exception(e)
