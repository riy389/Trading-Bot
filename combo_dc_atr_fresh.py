#!/usr/bin/env python3
"""
1H COMBO_DC_ATR — FRESH CROSSOVER + EMA200 + ADX FILTER.
Menggabungkan selektivitas sinyal Fresh Breakout (tidak re-entry berulang selama harga
masih di atas saluran) dengan filter tren makro EMA200 dan variasi R:R 1:2 hingga 1:3.
100% Non-Repaint: sinyal dievaluasi di close bar i, eksekusi di open bar i+1.
"""
import numpy as np
import pandas as pd

import backtest as bt
import eventtree2 as e2
from zerolag import stats
from tv_metrics import generate_tv_report

MAX_HOLD_BARS = 72

VARIANTS = [
    {"name": "Fresh Base (TP3.0% / SL1.5%)", "tp": 0.030, "sl": 0.015, "ema_filt": False, "adx_min": 0.0},
    {"name": "Fresh + EMA200 (TP3.0% / SL1.5%)", "tp": 0.030, "sl": 0.015, "ema_filt": True, "adx_min": 0.0},
    {"name": "Fresh + EMA200 (TP4.5% / SL1.5%)", "tp": 0.045, "sl": 0.015, "ema_filt": True, "adx_min": 0.0},
    {"name": "Fresh + EMA200 (TP4.0% / SL2.0%)", "tp": 0.040, "sl": 0.020, "ema_filt": True, "adx_min": 0.0},
    {"name": "Fresh + EMA200 + ADX20 (TP4.5% / SL1.5%)", "tp": 0.045, "sl": 0.015, "ema_filt": True, "adx_min": 20.0},
]


def calc_rma(series: np.ndarray, length: int) -> np.ndarray:
    """Wilder's Smoothing (RMA) identik TradingView."""
    out = np.full_like(series, np.nan, dtype=float)
    if len(series) < length:
        return out
    out[length - 1] = np.mean(series[:length])
    alpha = 1.0 / length
    for i in range(length, len(series)):
        out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
    return out


def calc_adx(h: np.ndarray, l: np.ndarray, c: np.ndarray, length: int = 14) -> np.ndarray:
    prev_h = np.roll(h, 1)
    prev_l = np.roll(l, 1)
    prev_c = np.roll(c, 1)
    prev_h[0], prev_l[0], prev_c[0] = h[0], l[0], c[0]

    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    up_move = h - prev_h
    dn_move = prev_l - l

    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)

    atr = calc_rma(tr, length)
    plus_di = 100.0 * calc_rma(plus_dm, length) / np.where(atr == 0, np.nan, atr)
    minus_di = 100.0 * calc_rma(minus_dm, length) / np.where(atr == 0, np.nan, atr)

    dx = 100.0 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, np.nan, (plus_di + minus_di))
    return calc_rma(np.nan_to_num(dx, nan=0.0), length)


def prepare_indicators(df: pd.DataFrame):
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)

    # Donchian Channel 20 (bebas lookahead)
    s_h = pd.Series(h)
    s_l = pd.Series(l)
    dc_hi = s_h.rolling(20).max().shift(1).to_numpy(float)
    dc_lo = s_l.rolling(20).min().shift(1).to_numpy(float)

    # Fresh Crossover: bar i tembus batas Donchian, sedangkan bar i-1 masih di dalam saluran
    prev_c = np.roll(c, 1)
    prev_dc_hi = np.roll(dc_hi, 1)
    prev_dc_lo = np.roll(dc_lo, 1)
    fresh_up = (c > dc_hi) & (prev_c <= prev_dc_hi)
    fresh_dn = (c < dc_lo) & (prev_c >= prev_dc_lo)

    # ATR(14) RMA & ekspansi volatilitas
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr14 = calc_rma(tr, 14)
    atr_ma = pd.Series(atr14).rolling(20).mean().to_numpy(float)
    atr_ok = (atr14 > atr_ma) & ((h - l) >= 0.8 * atr14)

    ema200 = pd.Series(c).ewm(span=200, adjust=False).mean().to_numpy(float)
    adx14 = calc_adx(h, l, c, 14)

    long_sig = fresh_up & atr_ok
    short_sig = fresh_dn & atr_ok

    long_sig[:205] = False
    short_sig[:205] = False

    return {
        "o": o,
        "h": h,
        "l": l,
        "c": c,
        "ema200": ema200,
        "adx": adx14,
        "long_sig": long_sig,
        "short_sig": short_sig,
        "yr": df["ts"].dt.year.to_numpy(),
        "months": (df["ts"].iloc[-1] - df["ts"].iloc[0]).days / 30.0,
        "df": df,
    }


def simulate_variant(data: dict, cfg: dict):
    o, h, l, c = data["o"], data["h"], data["l"], data["c"]
    ema200, adx = data["ema200"], data["adx"]
    long_sig, short_sig = data["long_sig"], data["short_sig"]
    n = len(c)

    tp_pct = cfg["tp"]
    sl_pct = cfg["sl"]
    use_ema = cfg["ema_filt"]
    adx_min = cfg["adx_min"]

    tr_stats = []
    tr_tv = []

    pos = 0
    ep = 0.0
    sl_px = 0.0
    tp_px = 0.0
    rd = sl_pct
    je = 0

    for i in range(205, n - 1):
        if pos == 1:
            if l[i] <= sl_px:
                exit_px = min(o[i], sl_px)
                tr_stats.append((je, 1, ep, exit_px, rd, "sl"))
                tr_tv.append((je, i, 1, ep, exit_px, rd, "sl"))
                pos = 0
            elif h[i] >= tp_px:
                exit_px = max(o[i], tp_px)
                tr_stats.append((je, 1, ep, exit_px, rd, "tp"))
                tr_tv.append((je, i, 1, ep, exit_px, rd, "tp"))
                pos = 0
            elif (i - je) >= MAX_HOLD_BARS:
                tr_stats.append((je, 1, ep, o[i + 1], rd, "lepas"))
                tr_tv.append((je, i + 1, 1, ep, o[i + 1], rd, "lepas"))
                pos = 0

        elif pos == -1:
            if h[i] >= sl_px:
                exit_px = max(o[i], sl_px)
                tr_stats.append((je, -1, ep, exit_px, rd, "sl"))
                tr_tv.append((je, i, -1, ep, exit_px, rd, "sl"))
                pos = 0
            elif l[i] <= tp_px:
                exit_px = min(o[i], tp_px)
                tr_stats.append((je, -1, ep, exit_px, rd, "tp"))
                tr_tv.append((je, i, -1, ep, exit_px, rd, "tp"))
                pos = 0
            elif (i - je) >= MAX_HOLD_BARS:
                tr_stats.append((je, -1, ep, o[i + 1], rd, "lepas"))
                tr_tv.append((je, i + 1, -1, ep, o[i + 1], rd, "lepas"))
                pos = 0

        if pos == 0:
            adx_ok = adx[i] >= adx_min
            ok_l = long_sig[i] and adx_ok and (not use_ema or c[i] > ema200[i])
            ok_s = short_sig[i] and adx_ok and (not use_ema or c[i] < ema200[i])

            if ok_l:
                pos = 1
                ep = o[i + 1]
                sl_px = ep * (1.0 - sl_pct)
                tp_px = ep * (1.0 + tp_pct)
                rd = sl_pct
                je = i + 1
            elif ok_s:
                pos = -1
                ep = o[i + 1]
                sl_px = ep * (1.0 + sl_pct)
                tp_px = ep * (1.0 - tp_pct)
                rd = sl_pct
                je = i + 1

    if pos != 0:
        tr_stats.append((je, pos, ep, c[-1], rd, "end"))
        tr_tv.append((je, n - 1, pos, ep, c[-1], rd, "end"))

    return tr_stats, tr_tv


def sim_small_account(tr_stats, fee_rt: float, leverage: float, init_eq: float = 6.0):
    """Simulasi akun $6 Cross Margin (margin $3 / max 50% equity, 1 posisi aktif)."""
    eq = init_eq
    peak = init_eq
    max_dd = 0.0
    for _, pos, ep, px, _, _ in tr_stats:
        if eq <= 0.05:
            eq = 0.0
            max_dd = 100.0
            break
        margin = min(3.0, eq * 0.5)
        notional = margin * leverage
        raw_ret = ((px - ep) / ep) if pos == 1 else ((ep - px) / ep)
        pnl = notional * (raw_ret - fee_rt)
        eq += pnl
        if eq <= 0.0:
            eq = 0.0
            max_dd = 100.0
            break
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100.0
        max_dd = max(max_dd, dd)
    return round(eq, 2), round(max_dd, 1)


def run(ex):
    df_1h = e2.load(ex, "1h")
    data = prepare_indicators(df_1h)

    rows = []
    yrs = []
    small_rows = []
    best_tv_candidates = []

    for cfg in VARIANTS:
        nm = cfg["name"]
        tr_stats, tr_tv = simulate_variant(data, cfg)
        r, y = stats(tr_stats, data["yr"], "1h", nm, data["months"])
        if r:
            rows.append(r)
            yrs += y
            best_tv_candidates.append((r.get("bersihR_total", -999), nm, tr_tv))

        rets_maker = []
        for _, pos, ep, px, rd, _ in tr_stats:
            raw_ret = ((px - ep) / ep) if pos == 1 else ((ep - px) / ep)
            rets_maker.append((raw_ret - 0.0010) / rd)
        maker_r_tot = round(float(np.sum(rets_maker)), 1) if rets_maker else 0.0

        eq5_mkt, dd5_mkt = sim_small_account(tr_stats, 0.0022, 5.0)
        eq10_mkt, dd10_mkt = sim_small_account(tr_stats, 0.0022, 10.0)
        eq5_mkr, dd5_mkr = sim_small_account(tr_stats, 0.0010, 5.0)
        eq10_mkr, dd10_mkr = sim_small_account(tr_stats, 0.0010, 10.0)

        small_rows.append({
            "varian": nm,
            "trade": len(tr_stats),
            "R_mkt(0.22%)": r["bersihR_total"] if r else 0,
            "R_mkr(0.10%)": maker_r_tot,
            "$6_5x_Mkt": f"${eq5_mkt} (DD {dd5_mkt}%)",
            "$6_10x_Mkt": f"${eq10_mkt} (DD {dd10_mkt}%)",
            "$6_5x_Mkr": f"${eq5_mkr} (DD {dd5_mkr}%)",
            "$6_10x_Mkr": f"${eq10_mkr} (DD {dd10_mkr}%)",
        })

    fence = "`" * 3
    bt._emit(
        "\n### 1h Combo_DC_ATR Fresh Breakout + EMA200 (fee 0,12% + slippage 0,10% = 0,22%)\n"
        + fence + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + fence
    )
    bt._emit(
        "\n### Simulasi Biaya Maker (0,10%) & Akun $6 Cross Margin (Margin $3, Lev 5x & 10x)\n"
        + fence + "\n" + pd.DataFrame(small_rows).to_string(index=False) + "\n" + fence
    )
    best_tv_candidates.sort(key=lambda x: x[0], reverse=True)
    for _, nm, tr_tv in best_tv_candidates[:2]:
        bt._emit(generate_tv_report(tr_tv, data["df"], "1h", nm))
    bt._emit(
        "\n### Per tahun (1h Combo_DC_ATR Fresh - bersihR)\n"
        + fence + "\n" + pd.DataFrame(yrs).to_string(index=False) + "\n" + fence
    )


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
