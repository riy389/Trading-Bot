#!/usr/bin/env python3
"""
1H REGIME FILTER TEST (ADX + Kaufman Efficiency Ratio).
Membandingkan performa 1h Combo_DC_ATR Fresh + EMA200 (TP3.0% / SL1.5%):
  1. Tanpa Regime Filter (Baseline)
  2. Mode A (Standby): Breakout aktif hanya saat Trending (ADX & ER tinggi), PAUSE saat Choppy.
  3. Mode B (Dual-Regime): Breakout saat Trending + Mean Reversion BB/RSI saat Choppy/Sideways.
100% Non-Repaint: sinyal dievaluasi di close bar i, eksekusi di open bar i+1.
"""
import numpy as np
import pandas as pd

import backtest as bt
import eventtree2 as e2
from zerolag import stats
from tv_metrics import generate_tv_report

MAX_HOLD_TREND = 72
MAX_HOLD_CHOP = 24


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


def calc_efficiency_ratio(c: np.ndarray, length: int = 20) -> np.ndarray:
    """Kaufman Efficiency Ratio (ER): 1.0 = tren lurus bersih, 0.0 = choppy bolak-balik."""
    s = pd.Series(c)
    direction = (s - s.shift(length)).abs()
    volatility = s.diff().abs().rolling(length).sum()
    er = direction / volatility.replace(0, np.nan)
    return er.fillna(0.0).to_numpy(float)


def calc_rsi(c: np.ndarray, length: int = 14) -> np.ndarray:
    diff = np.diff(c, prepend=c[0])
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    avg_g = calc_rma(gain, length)
    avg_l = calc_rma(loss, length)
    rs = avg_g / np.where(avg_l == 0, np.nan, avg_l)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return np.nan_to_num(rsi, nan=50.0)


def prepare_data(df: pd.DataFrame):
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)

    # Donchian 20 (bebas lookahead)
    s_h = pd.Series(h)
    s_l = pd.Series(l)
    dc_hi = s_h.rolling(20).max().shift(1).to_numpy(float)
    dc_lo = s_l.rolling(20).min().shift(1).to_numpy(float)

    prev_c = np.roll(c, 1)
    prev_dc_hi = np.roll(dc_hi, 1)
    prev_dc_lo = np.roll(dc_lo, 1)
    fresh_up = (c > dc_hi) & (prev_c <= prev_dc_hi)
    fresh_dn = (c < dc_lo) & (prev_c >= prev_dc_lo)

    # ATR(14) RMA
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr14 = calc_rma(tr, 14)
    atr_ma = pd.Series(atr14).rolling(20).mean().to_numpy(float)
    atr_ok = (atr14 > atr_ma) & ((h - l) >= 0.8 * atr14)

    # EMA 200
    ema200 = pd.Series(c).ewm(span=200, adjust=False).mean().to_numpy(float)

    # Regime Metrics: ADX(14) & Kaufman ER(20)
    adx14 = calc_adx(h, l, c, 14)
    er20 = calc_efficiency_ratio(c, 20)

    # Bollinger Bands (20, 2) & RSI(14) untuk Mean Reversion saat Choppy
    s_c = pd.Series(c)
    bb_mid = s_c.rolling(20).mean().to_numpy(float)
    bb_std = s_c.rolling(20).std().to_numpy(float)
    bb_up = bb_mid + 2.0 * bb_std
    bb_lo = bb_mid - 2.0 * bb_std
    rsi14 = calc_rsi(c, 14)

    trend_long = fresh_up & atr_ok & (c > ema200)
    trend_short = fresh_dn & atr_ok & (c < ema200)

    # Mean reversion: harga menyentuh/menembus pita BB + RSI ekstrem
    mr_long = (c < bb_lo) & (rsi14 < 32.0)
    mr_short = (c > bb_up) & (rsi14 > 68.0)

    trend_long[:205] = False
    trend_short[:205] = False
    mr_long[:205] = False
    mr_short[:205] = False

    return {
        "o": o,
        "h": h,
        "l": l,
        "c": c,
        "adx": adx14,
        "er": er20,
        "bb_mid": bb_mid,
        "trend_long": trend_long,
        "trend_short": trend_short,
        "mr_long": mr_long,
        "mr_short": mr_short,
        "yr": df["ts"].dt.year.to_numpy(),
        "months": (df["ts"].iloc[-1] - df["ts"].iloc[0]).days / 30.0,
        "df": df,
    }


VARIANTS = [
    {"name": "1. Base Fresh+EMA200 (Tanpa Regime)", "mode": "base", "adx_tr": 0.0, "er_tr": 0.0},
    {"name": "2. Regime Standby (ADX>22 & ER>0.22)", "mode": "standby", "adx_tr": 22.0, "er_tr": 0.22},
    {"name": "3. Regime Standby Ketat (ADX>25 & ER>0.28)", "mode": "standby", "adx_tr": 25.0, "er_tr": 0.28},
    {"name": "4. Dual-Regime (Trend + MR BB/RSI)", "mode": "dual", "adx_tr": 22.0, "er_tr": 0.22},
]


def simulate(data: dict, cfg: dict):
    o, h, l, c = data["o"], data["h"], data["l"], data["c"]
    adx, er, bb_mid = data["adx"], data["er"], data["bb_mid"]
    trend_long, trend_short = data["trend_long"], data["trend_short"]
    mr_long, mr_short = data["mr_long"], data["mr_short"]
    n = len(c)

    mode = cfg["mode"]
    adx_tr = cfg["adx_tr"]
    er_tr = cfg["er_tr"]

    tr_stats = []
    tr_tv = []

    pos = 0
    strat_type = ""  # "trend" atau "mr"
    ep = 0.0
    sl_px = 0.0
    tp_px = 0.0
    rd = 0.015
    je = 0

    for i in range(205, n - 1):
        # 1. Kelola posisi aktif
        if pos == 1:
            max_hold = MAX_HOLD_TREND if strat_type == "trend" else MAX_HOLD_CHOP
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
            elif strat_type == "mr" and c[i] >= bb_mid[i]:
                # Mean reversion keluar saat harga kembali ke rata-rata (Middle BB)
                tr_stats.append((je, 1, ep, o[i + 1], rd, "tp_mid"))
                tr_tv.append((je, i + 1, 1, ep, o[i + 1], rd, "tp_mid"))
                pos = 0
            elif (i - je) >= max_hold:
                tr_stats.append((je, 1, ep, o[i + 1], rd, "lepas"))
                tr_tv.append((je, i + 1, 1, ep, o[i + 1], rd, "lepas"))
                pos = 0

        elif pos == -1:
            max_hold = MAX_HOLD_TREND if strat_type == "trend" else MAX_HOLD_CHOP
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
            elif strat_type == "mr" and c[i] <= bb_mid[i]:
                tr_stats.append((je, -1, ep, o[i + 1], rd, "tp_mid"))
                tr_tv.append((je, i + 1, -1, ep, o[i + 1], rd, "tp_mid"))
                pos = 0
            elif (i - je) >= max_hold:
                tr_stats.append((je, -1, ep, o[i + 1], rd, "lepas"))
                tr_tv.append((je, i + 1, -1, ep, o[i + 1], rd, "lepas"))
                pos = 0

        # 2. Entry baru berdasarkan deteksi regime di close bar i
        if pos == 0:
            is_trending = True if mode == "base" else ((adx[i] >= adx_tr) and (er[i] >= er_tr))
            is_choppy = (adx[i] < 20.0) and (er[i] < 0.20)

            if is_trending:
                if trend_long[i]:
                    pos = 1
                    strat_type = "trend"
                    ep = o[i + 1]
                    rd = 0.015
                    sl_px = ep * (1.0 - rd)
                    tp_px = ep * 1.030
                    je = i + 1
                elif trend_short[i]:
                    pos = -1
                    strat_type = "trend"
                    ep = o[i + 1]
                    rd = 0.015
                    sl_px = ep * (1.0 + rd)
                    tp_px = ep * 0.970
                    je = i + 1

            elif mode == "dual" and is_choppy:
                if mr_long[i]:
                    pos = 1
                    strat_type = "mr"
                    ep = o[i + 1]
                    rd = 0.015
                    sl_px = ep * (1.0 - rd)
                    tp_px = ep * 1.022
                    je = i + 1
                elif mr_short[i]:
                    pos = -1
                    strat_type = "mr"
                    ep = o[i + 1]
                    rd = 0.015
                    sl_px = ep * (1.0 + rd)
                    tp_px = ep * 0.978
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
    data = prepare_data(df_1h)

    rows = []
    yrs = []
    small_rows = []
    best_tv = []

    for cfg in VARIANTS:
        nm = cfg["name"]
        tr_stats, tr_tv = simulate(data, cfg)
        r, y = stats(tr_stats, data["yr"], "1h", nm, data["months"])
        if r:
            rows.append(r)
            yrs += y
            best_tv.append((r.get("bersihR_total", -999), nm, tr_tv))

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
        "\n### 1h Regime Filter Test: Trending vs Choppy (fee 0,12% + slippage 0,10% = 0,22%)\n"
        + fence + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + fence
    )
    bt._emit(
        "\n### Simulasi Biaya Maker (0,10%) & Akun $6 Cross Margin (Margin $3, Lev 5x & 10x)\n"
        + fence + "\n" + pd.DataFrame(small_rows).to_string(index=False) + "\n" + fence
    )
    best_tv.sort(key=lambda x: x[0], reverse=True)
    for _, nm, tr_tv in best_tv[:2]:
        bt._emit(generate_tv_report(tr_tv, data["df"], "1h", nm))
    bt._emit(
        "\n### Per tahun (1h Regime Filter - bersihR)\n"
        + fence + "\n" + pd.DataFrame(yrs).to_string(index=False) + "\n" + fence
    )


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
