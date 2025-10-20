
"""
sanitize_window.py — Робастная подготовка опционных данных:
1) Принимаем сырые записи от провайдера (Polygon/иные).
2) Строим "сырую" таблицу.
3) Помечаем аномалии (IV/греков/дельт/данных).
4) Восстанавливаем IV-кривую по каждой экспирации и пересчитываем грейки (BS).
5) Формируем "исправленные" данные.
6) Выбираем окно страйков (по бленд-весу: |C-P|, ликвидность, долларовая гамма).
7) Возвращаем "оконные" таблицы (raw/corrected) и служебные артефакты.

Без внешних зависимостей, кроме numpy/pandas.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd



def _parse_expiration_to_ts(exp_raw):
    """Parse expiration to unix timestamp (seconds, UTC), DST-aware.
    - If number-like UNIX ts (seconds), return as float.
    - If string date without time (YYYY-MM-DD), interpret as end-of-day market close
      at 16:00 America/New_York for that calendar date (DST-aware), then convert to UTC.
    - Otherwise try robust UTC parsing.
    """
    # Numeric epoch seconds (keep as-is)
    if isinstance(exp_raw, (int, float)) and exp_raw > 10_000:
        try:
            return float(exp_raw)
        except Exception:
            return None

    exp_str = str(exp_raw) if exp_raw is not None else None
    if not exp_str:
        return None

    # Try parsing generically first
    try:
        dt_utc = pd.to_datetime(exp_str, utc=True)
    except Exception:
        try:
            dt_utc = pd.to_datetime(exp_str).tz_localize("UTC")
        except Exception:
            dt_utc = None

    # If this is a pure date (YYYY-MM-DD), build 16:00 America/New_York and convert
    is_date_only = isinstance(exp_str, str) and len(exp_str) == 10 and exp_str[4] == "-" and exp_str[7] == "-"
    if is_date_only:
        from datetime import datetime, time
        try:
            from zoneinfo import ZoneInfo  # Python 3.9+
            ny = ZoneInfo("America/New_York")
            y, m, d = map(int, exp_str.split("-"))
            dt_ny = datetime(y, m, d, 16, 0, 0, tzinfo=ny)  # 16:00 ET (DST-aware)
            ts = float(dt_ny.astimezone(tz=None).timestamp())  # to local tz, then timestamp -> epoch in UTC
            # Better: convert to UTC explicitly to avoid ambiguity
            ts = float(dt_ny.astimezone(ZoneInfo("UTC")).timestamp())
            return ts
        except Exception:
            # Fallback: keep previous UTC parse if available; else assume 20:00 UTC
            if dt_utc is not None:
                # If parsed to midnight UTC, shift by 20:00 as a conservative default
                if dt_utc.hour == 0 and dt_utc.minute == 0 and dt_utc.second == 0:
                    return float(dt_utc.timestamp() + 20*3600.0)
                return float(dt_utc.timestamp())
            # Last resort: try naive parse and add 20:00 UTC
            try:
                naive = pd.to_datetime(exp_str)
                return float(naive.tz_localize("UTC").timestamp() + 20*3600.0)
            except Exception:
                return None

    # If not date-only, return the robust UTC parse if present
    if dt_utc is not None:
        return float(dt_utc.timestamp())

    # As a final fallback, try naive -> UTC
    try:
        return float(pd.to_datetime(exp_str).tz_localize("UTC").timestamp())
    except Exception:
        return None

_SEC_PER_YEAR = 365.25 * 24 * 3600.0

def _yearfrac(now_ts: float, exp_ts: float) -> float:
    if exp_ts is None:
        return float('nan')
    T = max(0.0, (exp_ts - now_ts) / _SEC_PER_YEAR)
    return T

def _phi(x: np.ndarray) -> np.ndarray:
    # Стандартная нормальная pdf
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def _ndist(x: np.ndarray) -> np.ndarray:
    """Стандартная нормальная CDF без np.erf (используем math.erf)."""
    x = np.asarray(x, float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))

def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, None))

def _bs_greeks_gamma_vega_delta(S: np.ndarray, K: np.ndarray, T: np.ndarray,
                                iv: np.ndarray, r: float, q: float,
                                is_call: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает (gamma, vega, delta) в допущениях BS.
    - gamma: 1 / (денежная единица подложки)
    - vega:  изменение цены опциона на 1 пункт волы (в долях), единицы — ден.ед.
    - delta: безразмерная (в долях)

    Все массивы одинаковой длины. Для T<=0 или iv<=0 -> NaN.
    """
    S = np.asarray(S, float)
    K = np.asarray(K, float)
    T = np.asarray(T, float)
    iv = np.asarray(iv, float)
    is_call = np.asarray(is_call, bool)

    out_gamma = np.full_like(S, np.nan, dtype=float)
    out_vega  = np.full_like(S, np.nan, dtype=float)
    out_delta = np.full_like(S, np.nan, dtype=float)

    valid = (S > 0) & (K > 0) & (T > 0) & (iv > 0)
    if not np.any(valid):
        return out_gamma, out_vega, out_delta

    Sv = S[valid]
    Kv = K[valid]
    Tv = T[valid]
    vv = iv[valid]

    sqrtT = np.sqrt(Tv)
    d1 = (_safe_log(Sv / Kv) + (r - q + 0.5 * vv * vv) * Tv) / (vv * sqrtT)
    d2 = d1 - vv * sqrtT

    pdf = _phi(d1)
    cdf_d1 = _ndist(d1)
    cdf_m_d1 = _ndist(-d1)

    # gamma (в 1/ден.ед. подложки)
    gamma = np.exp(-q * Tv) * pdf / (Sv * vv * sqrtT)

    # vega (ден.ед. на 1.0 волы)
    vega = Sv * np.exp(-q * Tv) * pdf * sqrtT

    # delta
    # call: e^{-qT} N(d1), put: -e^{-qT} N(-d1)
    delta = np.where(is_call, np.exp(-q * Tv) * cdf_d1, -np.exp(-q * Tv) * cdf_m_d1)

    out_gamma[valid] = gamma
    out_vega[valid] = vega
    out_delta[valid] = delta
    return out_gamma, out_vega, out_delta


# -----------------------
# МОДЕЛЬ ДАННЫХ
# -----------------------

@dataclass
class SanitizerConfig:
    r: float = 0.0
    q: float = 0.0
    contract_mult_default: int = 100

    # Фильтры валидности
    iv_min: float = 0.03   # годовая вола в долях
    iv_max: float = 5.00
    allow_negative_gamma: bool = False
    allow_negative_vega: bool = False

    # Выбор окна
    p_cover: float = 0.95
    nmin: int = 15
    nmax: int = 49

    # Бленд-вес
    alpha_w: float = 1.0   # |C-P|
    beta_w: float = 0.3    # min(C,P)
    gamma_w: float = 0.25  # долларовая гамма * OI (после нормировки)


def _get_first(d: Dict, keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _get_nested(d: dict, keys: list[str], default=None,
                blocks: tuple[str, ...] = ("details", "greeks", "day", "underlying_asset")):
    """Ищет значение сначала на верхнем уровне, затем во вложенных блоках.
    - keys: порядок приоритетов ключей (первый найденный непустой возвращается)
    - blocks: имена вложенных словарей, характерных для snapshot-провайдеров (Polygon).
    """
    v = _get_first(d, keys, default=None)
    if v is not None:
        return v
    for b in blocks:
        sub = d.get(b, {})
        if isinstance(sub, dict):
            vv = _get_first(sub, keys, default=None)
            if vv is not None:
                return vv
    return default


# -----------------------
# 1) СЫРАЯ ТАБЛИЦА
# -----------------------

def build_raw_table(
    raw: List[Dict],
    S: float,
    now: Optional[datetime] = None,
    shares_per_contract: Optional[int] = None,
    cfg: SanitizerConfig = SanitizerConfig(),
) -> pd.DataFrame:
    """
    Строит сырую таблицу из списка записей провайдера.
    Унифицирует поля к колонкам:
    exp_ts, exp (строка), side (C/P), K, S, T, oi, vol, iv, delta, gamma, vega, mult.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    now_ts = now.timestamp()

    rows = []
    for r in raw:
        side_raw = str(_get_nested(r, ["side", "option_type", "contract_type", "type"], "")).lower()
        side = "C" if side_raw.startswith("c") else ("P" if side_raw.startswith("p") else None)

        K = _get_nested(r, ["strike", "k", "strike_price", "strikePrice"])
        if K is None:
            # иногда строка символа содержит strike, но это лишнее
            continue

        exp_raw = _get_nested(r, ["expiration", "expiration_date", "expiry", "expDate", "t"])
        # Robust expiration parsing with market-close adjustment (20:00 UTC for date-only strings)
        exp_ts = _parse_expiration_to_ts(exp_raw)
        exp_str = datetime.utcfromtimestamp(exp_ts).strftime("%Y-%m-%d") if exp_ts is not None else (str(exp_raw) if exp_raw is not None else None)

        oi = _get_nested(r, ["open_interest", "openInterest", "oi"], 0) or 0
        vol = _get_nested(r, ["volume", "vol"], 0) or 0
        iv  = _get_nested(r, ["implied_volatility", "impliedVolatility", "iv"], None)
        dlt = _get_nested(r, ["delta", "dlt"], None)
        gmm = _get_nested(r, ["gamma", "gmm"], None)
        vga = _get_nested(r, ["vega", "vga"], None)

        mult = _get_nested(r, ["shares_per_contract", "contract_size", "contractMultiplier"], None)
        if mult is None:
            mult = shares_per_contract if shares_per_contract is not None else cfg.contract_mult_default

        rows.append({
            "exp_ts": exp_ts,
            "exp": exp_str,
            "side": side,
            "K": float(K),
            "S": float(S),
            "T": _yearfrac(now_ts, exp_ts) if exp_ts is not None else float('nan'),
            "oi": float(oi),
            "vol": float(vol),
            "iv": float(iv) if iv is not None else float('nan'),
            "delta": float(dlt) if dlt is not None else float('nan'),
            "gamma": float(gmm) if gmm is not None else float('nan'),
            "vega": float(vga) if vga is not None else float('nan'),
            "mult": int(mult),
        })

    df = pd.DataFrame(rows)
    # Нормализация side
    df["side"] = df["side"].map({"C": "C", "P": "P"})
    return df


# -----------------------
# 2) ПОМЕТКА АНОМАЛИЙ
# -----------------------

def mark_anomalies(df: pd.DataFrame, cfg: SanitizerConfig = SanitizerConfig()) -> pd.DataFrame:
    """Добавляет колонки с флагами аномалий и текстовой причиной."""
    out = df.copy()

    bad_iv   = (out["iv"].isna()) | (out["iv"] <= cfg.iv_min) | (out["iv"] > cfg.iv_max)
    bad_gmm  = out["gamma"].isna() | (out["gamma"] < 0 if not cfg.allow_negative_gamma else False)
    bad_vga  = out["vega"].isna()  | (out["vega"]  < 0 if not cfg.allow_negative_vega else False)
    bad_dltC = (out["side"] == "C") & ((out["delta"].isna()) | (out["delta"] < 0) | (out["delta"] > 1))
    bad_dltP = (out["side"] == "P") & ((out["delta"].isna()) | (out["delta"] < -1) | (out["delta"] > 0))
    bad_T    = out["T"].isna() | (out["T"] <= 0)

    out["bad_iv"]   = bad_iv
    out["bad_gamma"]= bad_gmm
    out["bad_vega"] = bad_vga
    out["bad_delta"]= bad_dltC | bad_dltP
    out["bad_T"]    = bad_T
    out["bad_any"]  = bad_iv | bad_gmm | bad_vga | bad_dltC | bad_dltP | bad_T

    # Причины (собираем текстом для аудита)
    reasons = []
    for i, row in out.iterrows():
        r = []
        if row["bad_iv"]:    r.append("IV")
        if row["bad_gamma"]: r.append("Gamma")
        if row["bad_vega"]:  r.append("Vega")
        if row["bad_delta"]: r.append("Delta")
        if row["bad_T"]:     r.append("T")
        reasons.append(",".join(r))
    out["anomaly_reason"] = reasons
    return out


# -----------------------
# 3) ВОССТАНОВЛЕНИЕ IV И ПЕРЕСЧЁТ ГРЕКОВ
# -----------------------

def _forward_price(S: float, T: float, r: float, q: float) -> float:
    return S * math.exp((r - q) * max(T, 0.0))

def _interp_by_moneyness(Ks: np.ndarray, vals: np.ndarray, F: float) -> np.ndarray:
    """Линейная интерполяция по x = ln(K/F). На краях — удержание ближайшего значения."""
    Ks = np.asarray(Ks, float)
    vals = np.asarray(vals, float)
    x = np.log(np.clip(Ks / F, 1e-12, None))
    # сортировка по x
    idx = np.argsort(x)
    xs = x[idx]
    vs = vals[idx]

    # индексы валидных значений
    valid = ~np.isnan(vs)
    if valid.sum() == 0:
        return np.full_like(vals, np.nan)
    xs_valid = xs[valid]
    vs_valid = vs[valid]

    # для интерполяции нужен хотя бы один
    interp_vals_sorted = np.interp(xs, xs_valid, vs_valid, left=vs_valid[0], right=vs_valid[-1])
    # вернуть в исходный порядок Ks
    out = np.empty_like(vals)
    out[idx] = interp_vals_sorted
    return out

def rebuild_iv_and_greeks(df_marked: pd.DataFrame, cfg: SanitizerConfig = SanitizerConfig()) -> pd.DataFrame:
    """
    Для каждой экспирации:
      - строим комбинированную IV по страйкам (предпочитая валидные значения; если у C битая, а у P норм — берём P).
      - по интерполяции ln(K/F) восстанавливаем пропуски.
      - пересчитываем грейки (gamma/vega/delta) от этой IV для обоих типов опционов.
    Возвращает таблицу с колонками iv_corr, gamma_corr, vega_corr, delta_corr.
    """
    df = df_marked.copy()

    # Готовим контейнеры для скорректированных значений
    df["iv_corr"]    = np.nan
    df["gamma_corr"] = np.nan
    df["vega_corr"]  = np.nan
    df["delta_corr"] = np.nan

    # Группируем по экспирации
    for exp, g in df.groupby("exp", sort=False):
        idx = g.index.values
        S_vec = g["S"].values
        K_vec = g["K"].values
        T_vec = g["T"].values

        # Текущая цена S предполагается одинаковой для всей группы
        S0 = float(np.nanmedian(S_vec)) if len(S_vec) else float("nan")
        Tm = np.nanmedian(T_vec) if len(T_vec) else float("nan")
        F0 = _forward_price(S0, Tm, cfg.r, cfg.q)

        # IV по сторонам
        iv_call = g.loc[g["side"]=="C", ["K","iv","bad_iv"]].rename(columns={"iv":"iv_c","bad_iv":"bad_c"})
        iv_put  = g.loc[g["side"]=="P", ["K","iv","bad_iv"]].rename(columns={"iv":"iv_p","bad_iv":"bad_p"})
        iv_merged = pd.merge(iv_call, iv_put, on="K", how="outer")

        # Комбинация на одном страйке: если у C битая, у P норм — берём P; если обе норм — среднее; если обе битые — NaN
        def _combine_row(row):
            vc = row.get("iv_c", np.nan)
            vp = row.get("iv_p", np.nan)
            bc = bool(row.get("bad_c", True))
            bp = bool(row.get("bad_p", True))
            if not bc and not bp:
                return 0.5 * (float(vc) + float(vp))
            if not bc and bp:
                return float(vc)
            if bc and not bp:
                return float(vp)
            return float("nan")

        iv_merged["iv_comb"] = iv_merged.apply(_combine_row, axis=1)

        # Проекция на вектор всех K из группы
        K_all = g["K"].values
        # Строим массив комбинированной IV на Ks группы (восстановление интерполяцией по ln(K/F))
        # 1) собрать значение iv_comb для K, которые есть в iv_merged
        iv_map = {float(k): float(v) for k, v in zip(iv_merged["K"].values, iv_merged["iv_comb"].values)}
        iv_vals = np.array([iv_map.get(float(k), np.nan) for k in K_all], dtype=float)

        # 2) интерполяция/экстраполяция по ln(K/F)
        iv_filled = _interp_by_moneyness(K_all, iv_vals, F0)

        # inplace записываем скорректированную IV и forward F
        df.loc[idx, "iv_corr"] = iv_filled
        df.loc[idx, "F"] = F0

        # Пересчёт греков по скорректированной IV
        is_call = (g["side"].values == "C")
        gamma_c, vega_c, delta_c = _bs_greeks_gamma_vega_delta(
            S=g["S"].values,
            K=g["K"].values,
            T=g["T"].values,
            iv=iv_filled,
            r=cfg.r,
            q=cfg.q,
            is_call=is_call
        )
        df.loc[idx, "gamma_corr"] = gamma_c
        df.loc[idx, "vega_corr"]  = vega_c
        df.loc[idx, "delta_corr"] = delta_c

    return df


# -----------------------
# 4) ВЕС ДЛЯ ОКНА И ВЫБОР ОКНА
# -----------------------

def _normalize_for_blend(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    x = np.where(~np.isfinite(x) | (x < 0), 0.0, x)
    med = np.median(x[x > 0]) if np.any(x > 0) else 1.0
    return x / max(med, 1e-12)

def compute_window_weights(df_corr: pd.DataFrame, cfg: SanitizerConfig = SanitizerConfig()) -> pd.DataFrame:
    """
    Агрегируем по страйку в разрезе экспирации:
      - call_oi / put_oi
      - ликвидность: min(COI, POI)
      - долларовая гамма на 1% движения: sum_side( gamma_corr * S^2 * 0.01 * mult * oi_side )
      - бленд: alpha*|C-P| + beta*min(C,P) + gamma_w*Gamma_$1% *нормировки*
    Возвращает таблицу weights с одной строкой на (exp, K).
    """
    df = df_corr.copy()

    # Разделить C/P по OI
    c = df[df["side"]=="C"].groupby(["exp","K"], as_index=False)["oi"].sum().rename(columns={"oi":"call_oi"})
    p = df[df["side"]=="P"].groupby(["exp","K"], as_index=False)["oi"].sum().rename(columns={"oi":"put_oi"})
    base = pd.merge(c, p, on=["exp","K"], how="outer").fillna(0.0)

    # Ликвидность
    base["liq"] = np.minimum(base["call_oi"], base["put_oi"])

    # Долларовая гамма (на 1% движения), суммируем по сторонам
    df["gamma_dollar_1pct"] = df["gamma_corr"] * (df["S"]**2) * 0.01 * df["mult"]
    # умножим на OI каждой строки и агрегируем
    df["gamma_dollar_1pct_oi"] = df["gamma_dollar_1pct"] * df["oi"]
    ggd = df.groupby(["exp","K"], as_index=False)["gamma_dollar_1pct_oi"].sum()
    base = pd.merge(base, ggd, on=["exp","K"], how="left").fillna(0.0)

    # Бленд-веса с нормировкой по медиане
    w0 = _normalize_for_blend(np.abs(base["call_oi"].values - base["put_oi"].values))
    wl = _normalize_for_blend(base["liq"].values)
    wg = _normalize_for_blend(base["gamma_dollar_1pct_oi"].values)

    base["w_blend"] = cfg.alpha_w * w0 + cfg.beta_w * wl + cfg.gamma_w * wg
    return base


def _select_window_for_exp(strikes: np.ndarray, weights: np.ndarray, S: float,
                           p: float, nmin: int, nmax: int) -> np.ndarray:
    """Симметричное расширение окна от ATM до покрытия p (по сумме весов) с ограничениями nmin/nmax."""
    strikes = np.asarray(strikes, float)
    weights = np.asarray(weights, float)
    n = len(strikes)
    if n == 0:
        return np.arange(0)

    # ATM-индекс
    i_atm = int(np.argmin(np.abs(strikes - float(S))))
    L = R = i_atm

    # safety
    w = np.where(~np.isfinite(weights) | (weights < 0), 0.0, weights)
    total = w.sum()
    if total <= 0:
        # Нет информации — вернём минимум окрест ATM
        L = max(0, i_atm - nmin//2)
        R = min(n-1, L + nmin - 1)
        return np.arange(L, R+1)

    covered = w[i_atm]
    while covered / total < p and (L > 0 or R < n-1):
        # приращение слева/справа
        inc_left = w[L-1] if L > 0 else -1.0
        inc_right= w[R+1] if R < n-1 else -1.0
        if inc_left > inc_right:
            L -= 1
            covered += w[L]
        else:
            R += 1
            covered += w[R]

        # ограничение на максимум
        if (R - L + 1) >= nmax:
            break

    # Гарантируем минимум ширины
    while (R - L + 1) < nmin:
        if L > 0: L -= 1
        if (R - L + 1) >= nmin: break
        if R < n-1: R += 1
        if L == 0 and R == n-1: break

    return np.arange(L, R+1)


def select_windows(df_weights: pd.DataFrame, df_corr: pd.DataFrame, cfg: SanitizerConfig = SanitizerConfig()) -> Dict[str, np.ndarray]:
    """
    Для каждой экспирации возвращает индексы выбранного окна по сортированным страйкам.
    """
    windows = {}
    for exp, g in df_weights.groupby("exp", sort=False):
        # сортируем по K
        gs = g.sort_values("K").reset_index(drop=True)
        strikes = gs["K"].values
        weights = gs["w_blend"].values

        # берём S как медиану S по этой экспирации
        S_exp = float(np.nanmedian(df_corr.loc[df_corr["exp"]==exp, "S"].values))
        idx = _select_window_for_exp(strikes, weights, S_exp, cfg.p_cover, cfg.nmin, cfg.nmax)
        windows[exp] = idx
    return windows


# -----------------------
# 5) ФОРМИРОВАНИЕ "ОКОННЫХ" ТАБЛИЦ
# -----------------------

def build_window_tables(
    df_raw: pd.DataFrame,
    df_marked: pd.DataFrame,
    df_corr: pd.DataFrame,
    df_weights: pd.DataFrame,
    windows: Dict[str, np.ndarray]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Возвращает:
      - window_raw: сырые строки (C и P) только по страйкам окна каждой экспирации + флаги аномалий
      - window_corr: исправленные строки по тем же фильтрам (с iv_corr/gamma_corr/...)
    """
    # Список (exp, K) в окне
    keep_pairs = []
    for exp, idx in windows.items():
        g = df_weights[df_weights["exp"] == exp].sort_values("K").reset_index(drop=True)
        Ks_keep = g.loc[idx, "K"].values if len(g) else np.array([])
        keep_pairs.extend([(exp, float(k)) for k in Ks_keep])

    keep_df = pd.DataFrame(keep_pairs, columns=["exp","K"]).drop_duplicates()

    # Мерджим с raw/marked/corr
    def _filter_on_pairs(base: pd.DataFrame) -> pd.DataFrame:
        return base.merge(keep_df, on=["exp","K"], how="inner")

    window_raw  = _filter_on_pairs(df_marked)  # содержит флаги bad_*
    window_corr = _filter_on_pairs(df_corr)    # содержит iv_corr/gamma_corr/...

    # Удобная сортировка
    window_raw  = window_raw.sort_values(["exp","K","side"]).reset_index(drop=True)
    window_corr = window_corr.sort_values(["exp","K","side"]).reset_index(drop=True)
    return window_raw, window_corr


# -----------------------
# 6) ПАЙПЛАЙН "ПОД КЛЮЧ"
# -----------------------

def sanitize_and_window_pipeline(
    raw: List[Dict],
    S: float,
    now: Optional[datetime] = None,
    shares_per_contract: Optional[int] = None,
    cfg: SanitizerConfig = SanitizerConfig()
) -> Dict[str, object]:
    """
    Полный цикл:
      raw -> df_raw -> df_marked -> df_corr -> weights -> windows -> window_raw/window_corr
    Возвращает словарь с ключами:
      df_raw, df_marked, df_corr, df_weights, windows, window_raw, window_corr
    """
    df_raw = build_raw_table(raw, S=S, now=now, shares_per_contract=shares_per_contract, cfg=cfg)
    df_marked = mark_anomalies(df_raw, cfg=cfg)
    df_corr = rebuild_iv_and_greeks(df_marked, cfg=cfg)
    df_weights = compute_window_weights(df_corr, cfg=cfg)
    windows = select_windows(df_weights, df_corr, cfg=cfg)
    window_raw, window_corr = build_window_tables(df_raw, df_marked, df_corr, df_weights, windows)

    return {
        "df_raw": df_raw,
        "df_marked": df_marked,
        "df_corr": df_corr,
        "df_weights": df_weights,
        "windows": windows,
        "window_raw": window_raw,
        "window_corr": window_corr,
    }


# -----------------------
# Пример использования (закомментирован)
# -----------------------
# if __name__ == "__main__":
#     import json
#     with open("SPY_polygon_raw.json","r") as f:
#         raw = json.load(f)
#     S = 650.0
#     res = sanitize_and_window_pipeline(raw, S)
#     for k,v in res.items():
#         if isinstance(v, pd.DataFrame):
#             print(k, v.shape, v.columns.tolist())
#         else:
#             print(k, type(v))


def build_window_panels(
    df_corr: pd.DataFrame,
    df_weights: pd.DataFrame,
    windows: Dict[str, np.ndarray]
) -> Dict[str, pd.DataFrame]:
    """
    Возвращает словарь {exp: DataFrame} с итоговой таблицей по страйкам окна.
    Таблица включает:
      - S, F, exp, K
      - call_oi, put_oi
      - call_vol, put_vol
      - iv_corr_call, iv_corr_put (по возможности; иначе NaN)
      - gamma_dollar_1pct_oi (сумма по сторонам)
      - w_blend (вес окна на страйке)
    """
    panels = {}

    # Подготовим агрегаты по OI/Vol/IV (по сторонам)
    agg_oi = df_corr.groupby(["exp","K","side"], as_index=False)["oi"].sum().rename(columns={"oi":"oi_side"})
    agg_vol= df_corr.groupby(["exp","K","side"], as_index=False)["vol"].sum().rename(columns={"vol":"vol_side"})
    # IV корр: возьмём медиану по строкам данной стороны
    agg_iv = df_corr.groupby(["exp","K","side"], as_index=False)["iv_corr"].median().rename(columns={"iv_corr":"iv_side"})

    # долларовая гамма * OI уже была в compute_window_weights, посчитаем ещё раз локально для устойчивости
    tmp = df_corr.copy()
    tmp["gamma_dollar_1pct"] = tmp["gamma_corr"] * (tmp["S"]**2) * 0.01 * tmp["mult"]
    tmp["gamma_dollar_1pct_oi"] = tmp["gamma_dollar_1pct"] * tmp["oi"]
    agg_g = tmp.groupby(["exp","K"], as_index=False)["gamma_dollar_1pct_oi"].sum()

    # Мержим агрегаты
    base = agg_oi.merge(agg_vol, on=["exp","K","side"], how="outer")
    base = base.merge(agg_iv, on=["exp","K","side"], how="outer")

    # Разворачиваем по сторонам в колонки
    panel_all = (
        base.pivot_table(index=["exp","K"], columns="side", values=["oi_side","vol_side","iv_side"], aggfunc="sum")
        .sort_index()
    )
    panel_all.columns = [f"{v}_{s.lower()}" for v,s in panel_all.columns]
    panel_all = panel_all.reset_index()

    # Подмешаем гамму и веса окна
    panel_all = panel_all.merge(agg_g, on=["exp","K"], how="left")
    panel_all = panel_all.merge(df_weights[["exp","K","w_blend"]], on=["exp","K"], how="left")

    # S и F одинаковы для всей экспирации — возьмём медиану из df_corr
    for exp, g in panel_all.groupby("exp"):
        S_exp = float(np.nanmedian(df_corr.loc[df_corr["exp"]==exp, "S"].values))
        F_exp = float(np.nanmedian(df_corr.loc[df_corr["exp"]==exp, "F"].values)) if "F" in df_corr.columns else float("nan")
        panel_all.loc[panel_all["exp"]==exp, "S"] = S_exp
        panel_all.loc[panel_all["exp"]==exp, "F"] = F_exp

    # Оставим только страйки окна
    panels = {}
    for exp, idx in windows.items():
        gW = df_weights[df_weights["exp"]==exp].sort_values("K").reset_index(drop=True)
        Ks_keep = gW.loc[idx, "K"].values if len(gW) else np.array([])
        panel = panel_all[ (panel_all["exp"]==exp) & (panel_all["K"].isin(Ks_keep)) ].copy()
        panel = panel.sort_values("K").reset_index(drop=True)
        panels[exp] = panel

    return panels
