#!/usr/bin/env python3
"""
TRIPLE SUPERTREND OPTIMIZED (10-1, 11-2, 12-3 — Exit-B Tahan Tren).
Menguji 3 penyempurnaan pada varian unggulan (1h MTF-A +4h dan 4h sendiri):
  1. Filter Anti-Sideways: ADX(14) > 20 saat sinyal entry terbentuk.
  2. Filter Asimetris Short: Long bebas, Short hanya jika close < EMA(200).
  3. Risk-Based Position Sizing (1% & 2% risiko per trade berdasarkan jarak SL) untuk menekan Max Drawdown.
"""
import numpy as np
import pandas as pd

import backtest as bt
import eventtree2 as e2
from zerolag import stats
from tv_metrics import generate_tv_report

ST_PARAMS = (
    (10, 1.0),
    (11, 2.0),
    (12, 3.0),
)


def rma(series: np.ndarray, length: int) -> np.ndarray:
    out = np.full(len(series), np.nan, dtype=float)
    if len(series) < length:
        return out
    out[length - 1] = np.mean(series[:length])
    alpha = 1.0 / length
    for i in range(length, len(series)):
        out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
    return out


def calc_adx(df: pd.DataFrame, length: int = 14) -> np.ndarray:
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)

    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))

    up_move = h - np.roll(h, 1)
    dn_move = np.roll(l, 1) - l
    up_move[0] = 0.0
    dn_move[0] = 0.0

    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)

    atr = rma(tr, length)
    plus_di = 100.0 * rma(plus_dm, length) / np.maximum(atr, 1e-8)
    minus_di = 100.0 * rma(minus_dm, length) / np.maximum(atr, 1e-8)

    dx = 100.0 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-8)
    adx = rma(np.nan_to_num(dx, nan=0.0), length)
    return np.nan_to_num(adx, nan=0.0)


def calc_supertrend(df: pd.DataFrame, atr_len: int, mult: float):
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(c)

    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr_raw = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = rma(tr_raw, atr_len)

    hl2 = (h + l) / 2.0
    basic_ub = hl2 + mult * atr
    basic_lb = hl2 - mult * atr

    ub = np.copy(basic_ub)
    lb = np.copy(basic_lb)
    trend = np.zeros(n, dtype=int)

    start = atr_len
    if start >= n:
        return ub, lb, trend

    trend[start] = 1 if c[start] >= hl2[start] else -1

    for i in range(start + 1, n):
        ub[i] = basic_ub[i] if (basic_ub[i] < ub[i - 1] or c[i - 1] > ub[i - 1]) else ub[i - 1]
        lb[i] = basic_lb[i] if (basic_lb[i] > lb[i - 1] or c[i - 1] < lb[i - 1]) else lb[i - 1]

        if trend[i - 1] == 1:
            trend[i] = -1 if c[i] < lb[i] else 1
        else:
            trend[i] = 1 if c[i] > ub[i] else -1

    return ub, lb, trend


def prepare_tf(df: pd.DataFrame):
    t = pd.to_datetime(df["ts"]).astype("int64").to_numpy()
    o = df["open"].to_numpy(float)
    c = df["close"].to_numpy(float)

    sts = [calc_supertrend(df, length, mult) for length, mult in ST_PARAMS]
    _, _, tr1 = sts[0]
    _, _, tr2 = sts[1]
    ub3, lb3, tr3 = sts[2]

    conf = np.where((tr1 == 1) & (tr2 == 1) & (tr3 == 1), 1,
           np.where((tr1 == -1) & (tr2 == -1) & (tr3 == -1), -1, 0))

    # Exit-B (Tahan Tren): entry saat konfluensi penuh, bertahan selama 12-3 belum flip
    hold_B = np.zeros(len(conf), dtype=int)
    cur = 0
    for i in range(1, len(conf)):
        if conf[i] != 0 and conf[i] != cur:
            cur = conf[i]
        elif cur == 1 and tr3[i] != 1:
            cur = 0
        elif cur == -1 and tr3[i] != -1:
            cur = 0
        hold_B[i] = cur

    adx14 = calc_adx(df, 14)
    ema200 = pd.Series(c).ewm(span=200, adjust=False).mean().to_numpy(float)

    return {
        "df": df,
        "t": t,
        "step": int(np.median(np.diff(t))),
        "o": o,
        "c": c,
        "conf": conf,
        "hold_B": hold_B,
        "up3": ub3,
        "dn3": lb3,
        "adx": adx14,
        "ema200": ema200,
        "yr": df["ts"].dt.year.to_numpy(),
        "months": (df["ts"].iloc[-1] - df["ts"].iloc[0]).days / 30,
    }


def align_conf(e, h):
    idx = np.searchsorted(h["t"] + h["step"], e["t"] + e["step"], side="right") - 1
    return np.where(idx >= 0, h["conf"][np.clip(idx, 0, None)], 0)


def build_filtered_target(e, use_mtf_4h: bool, D4h, use_adx: bool, use_asym_short: bool):
    conf = e["conf"]
    hold = e["hold_B"]
    c = e["c"]
    adx = e["adx"]
    ema200 = e["ema200"]
    n = len(c)

    mtf_dir = align_conf(e, D4h) if use_mtf_4h else hold

    tg = np.zeros(n, dtype=int)
    cur = 0
    for i in range(200, n):
        # Cek apakah sinyal bertahan Exit-B sudah selesai
        if cur != 0:
            if hold[i] == 0:
                cur = 0
            elif hold[i] != cur:
                # Balik arah langsung hanya jika konfluensi baru lolos seluruh filter
                new_dir = hold[i]
                pass_mtf = (mtf_dir[i] == new_dir) if use_mtf_4h else True
                pass_adx = (adx[i] > 20.0) if use_adx else True
                pass_short = (c[i] < ema200[i]) if (use_asym_short and new_dir == -1) else True
                cur = new_dir if (conf[i] == new_dir and pass_mtf and pass_adx and pass_short) else 0

        # Jika flat, cek trigger konfluensi baru
        if cur == 0 and conf[i] != 0 and conf[i] != conf[i - 1]:
            cand = conf[i]
            pass_mtf = (mtf_dir[i] == cand) if use_mtf_4h else True
            pass_adx = (adx[i] > 20.0) if use_adx else True
            pass_short = (c[i] < ema200[i]) if (use_asym_short and cand == -1) else True
            if pass_mtf and pass_adx and pass_short:
                cur = cand

        tg[i] = cur
    return tg


def run_trades(e, tg):
    o, c, up, dn = e["o"], e["c"], e["up3"], e["dn3"]
    out_stats, out_tv = [], []
    pos, ep, rd, je = 0, 0.0, 0.0, 0
    for i in range(1, len(c) - 1):
        if tg[i] != tg[i - 1]:
            px = o[i + 1]
            if pos:
                reason = "flip" if tg[i] == -pos else "lepas"
                out_stats.append((je, pos, ep, px, rd, reason))
                out_tv.append((je, i + 1, pos, ep, px, rd, reason))
            pos = tg[i]
            if pos:
                ep, je = px, i + 1
                rd = max(abs((px - dn[i]) / px) if pos == 1 else abs((up[i] - px) / px), 1e-4)
    if pos:
        out_stats.append((je, pos, ep, c[-1], rd, "end"))
        out_tv.append((je, len(c) - 1, pos, ep, c[-1], rd, "end"))
    return out_stats, out_tv


def sim_risk_sizing(tr_stats, risk_pct: float, cost_pct: float = 0.0022):
    """Simulasi compounding dengan ukuran lot berbasis risiko tetap per trade (1% atau 2% modal)."""
    eq = 10000.0
    peak = eq
    max_dd = 0.0
    for _, pos, ep, px, rd, _ in tr_stats:
        gross_ret = ((px - ep) / ep) if pos == 1 else ((ep - px) / ep)
        net_ret = gross_ret - cost_pct
        r_mult = net_ret / max(rd, 1e-4)
        # Batasi leverage efektif maksimal 5x agar realistis
        lev = min(risk_pct / max(rd, 1e-4), 5.0)
        pnl_pct = lev * net_ret
        eq = max(eq * (1.0 + pnl_pct), 0.0)
        if eq > peak:
            peak = eq
        dd = ((peak - eq) / peak) * 100.0 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    roi = ((eq - 10000.0) / 10000.0) * 100.0
    return roi, max_dd, eq


def run(ex):
    d_raw = {tf: e2.load(ex, tf) for tf in ("1h", "4h")}
    D = {tf: prepare_tf(df) for tf, df in d_raw.items()}

    setups = [
        ("4h", False, "Baseline (Tanpa Filter)", False, False),
        ("4h", False, "+ ADX(14) > 20", True, False),
        ("4h", False, "+ Short < EMA200", False, True),
        ("4h", False, "+ ADX>20 & Short<EMA200", True, True),
        ("1h", True, "MTF+4h Baseline", False, False),
        ("1h", True, "MTF+4h + ADX(14) > 20", True, False),
        ("1h", True, "MTF+4h + Short < EMA200", False, True),
        ("1h", True, "MTF+4h + ADX>20 & Short<EMA200", True, True),
    ]

    rows, sizing_rows, yrs, tv_blocks = [], [], [], []
    for tf, use_mtf, label, use_adx, use_asym in setups:
        e = D[tf]
        tg = build_filtered_target(e, use_mtf, D["4h"], use_adx, use_asym)
        tr_stats, tr_tv = run_trades(e, tg)
        r, y = stats(tr_stats, e["yr"], tf, label, e["months"])
        if r:
            rows.append(r)
            yrs += y
            roi_1, dd_1, eq_1 = sim_risk_sizing(tr_stats, 0.01)
            roi_2, dd_2, eq_2 = sim_risk_sizing(tr_stats, 0.02)
            sizing_rows.append({
                "TF": tf,
                "varian": label,
                "bersihR": r["bersihR_total"],
                "ROI_1%Risk": f"{roi_1:.1f}%",
                "MaxDD_1%Risk": f"{dd_1:.1f}%",
                "ROI_2%Risk": f"{roi_2:.1f}%",
                "MaxDD_2%Risk": f"{dd_2:.1f}%",
                "EndCap_2%": f"${eq_2:,.0f}",
            })
            if r["bersihR_total"] > 0:
                tv_blocks.append(generate_tv_report(tr_tv, e["df"], tf, label))

    fence = "`" * 3
    bt._emit(
        "\n### Triple Supertrend Exit-B Optimization (ADX>20 + Asymmetric Short < EMA200)\n"
        + fence + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + fence
    )
    bt._emit(
        "\n### Simulasi Risk-Based Position Sizing (1% & 2% Risk per Trade, Max Lev 5x)\n"
        + fence + "\n" + pd.DataFrame(sizing_rows).to_string(index=False) + "\n" + fence
    )
    for blk in tv_blocks:
        bt._emit(blk)
    bt._emit(
        "\n### Per tahun (bersihR)\n"
        + fence + "\n" + pd.DataFrame(yrs).to_string(index=False) + "\n" + fence
    )


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
