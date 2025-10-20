
# -*- coding: utf-8 -*-
"""
power_zone_er.py — расчёт Power Zone (PZ) и Easy Reach (ER Up / ER Down)
по формулировке, использованной в проекте (функция compute_power_zone_and_er).

Вход:
    S : float — спотовая цена
    strikes_eval : Iterable[float] — сетка страйков (оценочная шкала, обычно окно)
    all_series_ctx : Iterable[dict] — список контекстов по экспирациям:
        каждый dict должен содержать ключи (как в проекте):
            "strikes"         : list[float]
            "gamma_abs_share" : array-like или {K: val}
            "gamma_net_share" : array-like или {K: val}
            "call_oi", "put_oi"   : {K: float}
            "call_vol","put_vol"  : {K: float}
            "iv_call","iv_put"    : {K: float}  (для оценки sigma_est)
            "T" : float — время до экспирации в годах

Опционально:
    day_high, day_low : float — дневной High/Low (для ограничения ширины ядра)
Параметры (оставлены по умолчанию как в проекте):
    beta0=1.0, eta=0.60, zeta=0.40, alpha_g=1.0, alpha_v=1.0,
    theta_short_gamma=0.5, c_vol_log=4.0, d_star=2.0, eps=1e-6

Выход:
    (pz_norm, er_up_norm, er_down_norm) — по одному массиву длины len(strikes_eval), нормированных к [0,1].
"""

from __future__ import annotations
import math
from typing import Iterable, Tuple, Dict, Any, List, Optional
import numpy as _np


def compute_power_zone(
    S: float,
    strikes_eval: Iterable[float],
    all_series_ctx: Iterable[Dict[str, Any]],
    day_high: Optional[float] = None,
    day_low: Optional[float] = None,
    *,
    beta0: float = 1.0,
    eta: float = 0.50,
    zeta: float = 0.40,
    alpha_g: float = 1.0,
    alpha_v: float = 1.0,
    theta_short_gamma: float = 0.5,
    c_vol_log: float = 4.0,
    d_star: float = 2.0,
    eps: float = 1e-6,
) -> _np.ndarray:
    """Вычисление Power Zone (PZ) по оценочной сетке. Возвращает массив длины len(strikes_eval)."""
    # Преобразуем вход
    strikes_eval = _np.asarray(list(strikes_eval), dtype=float)
    n_eval = len(strikes_eval)
    all_series_ctx = list(all_series_ctx)
    if n_eval == 0 or not all_series_ctx:
        return _np.zeros(0, dtype=float)

    # --- Веса по экспирациям (time weighting) ---
    # tau ~ позиция в сессии: 0.5 (mid-session) если реального времени нет
    tau = 0.5
    total_oi = _np.array([
        float(sum((s.get("call_oi") or {}).values()) + sum((s.get("put_oi") or {}).values()))
        for s in all_series_ctx
    ], dtype=float)
    T_arr = _np.array([float(s.get("T", 0.0)) for s in all_series_ctx], dtype=float)
    W_time = total_oi * (T_arr**(-eta)) * ((1.0 - tau)**zeta)
    W_time = (W_time / W_time.sum()) if W_time.sum() > 0 else _np.ones_like(W_time) / len(W_time)

    # --- Ядро и ширина окна (h) ---
    # Оценка sigma из всех доступных IV по сериям; fallback 0.25
    iv_vals: List[float] = []
    for s in all_series_ctx:
        for key in ("iv_call","iv_put"):
            ivd = s.get(key)
            if isinstance(ivd, dict):
                for v in ivd.values():
                    try:
                        vv = float(v)
                        if vv > 0 and not math.isnan(vv):
                            iv_vals.append(vv)
                    except Exception:
                        pass
    sigma_est = float(_np.median(iv_vals)) if iv_vals else 0.0
    # Интрадейный диапазон R
    if isinstance(day_high, (int, float)) and isinstance(day_low, (int, float)):
        R = float(day_high) - float(day_low)
    else:
        R = 0.01 * float(S)
    # «Средний шаг» по оценочной сетке
    def _median_step(arr: List[float]) -> float:
        if len(arr) < 2: return 1.0
        diffs = sorted([abs(arr[i+1]-arr[i]) for i in range(len(arr)-1)])
        mid = len(diffs)//2
        return diffs[mid] if len(diffs)%2==1 else 0.5*(diffs[mid-1]+diffs[mid])
    step_eval = _median_step(strikes_eval.tolist())

    # Базовая ширина ядра по волатильности (1/sqrt(252))
    T_med = float(_np.median(T_arr[T_arr > 0])) if _np.any(T_arr > 0) else (1.0/252.0)
    h_base = float(S) * float(sigma_est) * math.sqrt(T_med)
    h = max(step_eval, h_base)

    # --- Предобработка серий: сглаженные AG, стабильность и активность ---
    prep = []
    for s in all_series_ctx:
        Ks = _np.array(list(s.get("strikes") or []), dtype=float)
        if Ks.size == 0:
            continue
        Ks.sort()

        # g_abs/g_net могут прийти как дикты (по страйкам) или как массивы, выровненные с Ks
        g_raw_abs = s.get("gamma_abs_share")
        g_raw_net = s.get("gamma_net_share")
        if isinstance(g_raw_abs, dict):
            g_abs = _np.array([g_raw_abs.get(float(k), 0.0) for k in Ks], dtype=float)
        elif g_raw_abs is None:
            g_abs = _np.zeros_like(Ks, dtype=float)
        else:
            g_abs = _np.asarray(g_raw_abs, dtype=float)

        if isinstance(g_raw_net, dict):
            g_net = _np.array([g_raw_net.get(float(k), 0.0) for k in Ks], dtype=float)
        elif g_raw_net is None:
            g_net = _np.zeros_like(Ks, dtype=float)
        else:
            g_net = _np.asarray(g_raw_net, dtype=float)

        # Перевод в долларовые величины (как в проекте): * S / 1000
            AG_e = _np.asarray(g_abs, float)
            NG_e = _np.abs(_np.asarray(g_net, float))

        # Локальное треугольное сглаживание по окну h
        def _median_step2(xarr: List[float]) -> float:
            if len(xarr) < 2: return 1.0
            diffs = sorted([abs(xarr[i+1]-xarr[i]) for i in range(len(xarr)-1)])
            mid = len(diffs)//2
            return diffs[mid] if len(diffs)%2==1 else 0.5*(diffs[mid-1]+diffs[mid])
        step_i = _median_step2(list(Ks))
        radius = max(1, int(round(h / max(step_i, 1e-8))))
        n_s = len(Ks)
        AG_loc = _np.zeros(n_s, dtype=float)
        NG_loc = _np.zeros(n_s, dtype=float)
        for j in range(n_s):
            l = max(0, j - radius); rj = min(n_s - 1, j + radius)
            idx = _np.arange(l, rj + 1)
            w = 1.0 - _np.abs(idx - j) / radius
            w[w < 0] = 0.0
            W = w.sum() if w.sum() > 0 else 1.0
            AG_loc[j] = float((_np.sum(AG_e[idx] * w)) / W)
            NG_loc[j] = float((_np.sum(NG_e[idx] * w)) / W)

        # Стабильность: отношение AG к AG+NG
        Stab_e = AG_loc / (AG_loc + NG_loc + eps)

        # Активность: смесь объёма и OI с лог-компрессией и ослаблением вдали от ATM
        call_oi_dict = s.get("call_oi", {}) or {}
        put_oi_dict  = s.get("put_oi",  {}) or {}
        call_vol_dict= s.get("call_vol",{}) or {}
        put_vol_dict = s.get("put_vol", {}) or {}
        call_oi_vec = _np.array([call_oi_dict.get(k, 0.0) for k in Ks], dtype=float)
        put_oi_vec  = _np.array([put_oi_dict.get(k, 0.0)  for k in Ks], dtype=float)
        vol_vec     = _np.array([call_vol_dict.get(k, 0.0) + put_vol_dict.get(k, 0.0) for k in Ks], dtype=float)

        if c_vol_log > 0:
            Vol_eff = _np.log1p(c_vol_log * vol_vec) / _np.log1p(c_vol_log)
        else:
            Vol_eff = vol_vec.copy()

        OI_eff = _np.power(_np.maximum(call_oi_vec + put_oi_vec, 0.0), 0.90)

        atm_idx = int(_np.argmin(_np.abs(Ks - float(S))))
        idxs = _np.arange(len(Ks), dtype=int)
        dist_steps = _np.abs(idxs - atm_idx)
        w_blend = _np.clip(1.0 - dist_steps / d_star, 0.0, 1.0)
        vol_sum = _np.maximum(vol_vec, 0.0)
        oi_sum  = _np.maximum(call_oi_vec + put_oi_vec, 0.0)

        mxv = float(_np.nanmax(vol_sum)) if vol_sum.size else 0.0
        mxi = float(_np.nanmax(oi_sum))  if oi_sum.size  else 0.0

        vol_n = (vol_sum / mxv) if mxv > 0.0 else _np.zeros_like(vol_sum)
        oi_n  = (oi_sum  / mxi) if mxi > 0.0 else _np.zeros_like(oi_sum)

        Act_raw = _np.sqrt(vol_n * oi_n)

        # Нормировки к [0,1]
        def _norm01(arr: _np.ndarray) -> _np.ndarray:
            arr = _np.asarray(arr, dtype=float)
            mx = float(_np.nanmax(arr)) if arr.size else 0.0
            return (arr / mx) if mx > 0 else _np.zeros_like(arr)

        AG_hat = AG_loc
        Stab_hat = Stab_e
        Act_hat = Act_raw

        prep.append({
            "Ks": Ks,
            "AG_loc": AG_loc,
            "AG_hat": AG_hat,
            "Stab_hat": Stab_hat,
            "Act_hat": Act_hat,
            "vol_vec": vol_vec,
            "call_vol_dict": call_vol_dict,
            "put_vol_dict":  put_vol_dict,
            "call_oi_dict":  call_oi_dict,
            "put_oi_dict":   put_oi_dict,
        })

    if not prep:
        return (_np.zeros(n_eval, dtype=float), _np.zeros(n_eval, dtype=float), _np.zeros(n_eval, dtype=float))

    # --- Ядро по оценочной сетке (гауссово, центр в S) ---
    dist_eval = (strikes_eval - float(S)) / float(h if h > 0 else 1.0)
    Wd_eval = _np.exp(-0.5 * (dist_eval ** 2))

    # --- Power Zone (масса) ---
    mass_vals = _np.zeros(n_eval, dtype=float)
    for w_e, c in zip(W_time, prep):
        Ks = c["Ks"]
        idx_near = _np.searchsorted(Ks, strikes_eval).clip(1, len(Ks)-1)
        left = idx_near - 1; right = idx_near
        pick = _np.where(_np.abs(Ks[left] - strikes_eval) <= _np.abs(Ks[right] - strikes_eval), left, right)
        mass_vals += w_e * Wd_eval * (c["AG_hat"][pick] * c["Stab_hat"][pick] * c["Act_hat"][pick])
    return mass_vals
