#!/usr/bin/env python3
"""
TAHAP 1: FADE FALSE BREAKOUT (CHOPPY) + TREND BREAKOUT (TRENDING) — BTC 1H.
Membandingkan:
  1. Trend-Only (Standby Ketat: ADX>25 & ER>0.28) sebagai tolok ukur pemenang sebelumnya.
  2. Fade-Only (Limit Order di dc_hi / dc_lo saat Choppy: ADX<20 & ER<0.20).
  3. Combined (Trend Breakout saat Trending + Fade Fakeout Limit Order saat Choppy).
100% Non-Repaint: level Donchian & kondisi regime dihitung dari bar sebelumnya (sudah close).
Limit order terisi intrabar jika high/low menyentuh level dc_hi/dc_lo (biaya Maker/Hybrid 0,10%).
"""
import numpy as np
import pandas as pd

import backtest as bt
import eventtree2 as e2
from zerolag import stats
from tv_metrics import generate_tv_report

MAX_HOLD_TREND = 72
MAX_HOLD_FADE = 18


def calc_rma(series: np.ndarray, length: int) -> np.ndarray:
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
    s = pd.Series(c)
    direction = (s - s.shift(length)).abs()
    volatility = s.diff().abs().rolling(length).sum()
    er = direction / volatility.replace(0, np.nan)
    return er.fillna(0.0).to_numpy(float)


def prepare_data(df: pd.DataFrame):
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)

    # Donchian 20 (bebas lookahead: pakai max/min 20 bar sebelum bar i)
    s_h = pd.Series(h)
    s_l = pd.Series(l)
    dc_hi = s_h.rolling(20).max().shift(1).to_numpy(float)
    dc_lo = s_l.rolling(20).min().shift(1).to_numpy(float)

    prev_c = np.roll(c, 1)
    prev_dc_hi = np.roll(dc_hi, 1)
    prev_dc_lo = np.roll(dc_lo, 1)
    fresh_up = (c > dc_hi) & (prev_c <= prev_dc_hi)
    fresh_dn = (c < dc_lo) & (prev_c >= prev_dc_lo)

    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr14 = calc_rma(tr, 14)
    atr_ma = pd.Series(atr14).rolling(20).mean().to_numpy(float)
    atr_ok = (atr14 > atr_ma) & ((h - l) >= 0.8 * atr14)

    ema200 = pd.Series(c).ewm(span=200, adjust=False).mean().to_numpy(float)
    adx14 = calc_adx(h, l, c, 14)
    er20 = calc_efficiency_ratio(c, 20)

    trend_long = fresh_up & atr_ok & (c > ema200)
    trend_short = fresh_dn & atr_ok & (c < ema200)
    trend_long[:205] = False
    trend_short[:205] = False

    return {
        "o": o,
        "h": h,
        "l": l,
        "c": c,
        "dc_hi": dc_hi,
        "dc_lo": dc_lo,
        "adx": adx14,
        "er": er20,
        "trend_long": trend_long,
        "trend_short": trend_short,
        "yr": df["ts"].dt.year.to_numpy(),
        "months": (df["ts"].iloc[-1] - df["ts"].iloc[0]).days / 30.0,
        "df": df,
    }


VARIANTS = [
    {"name": "1. Trend-Only (ADX>25 & ER>0.28)", "use_trend": True, "use_fade": False, "fade_tp": 0.012, "fade_sl": 0.008},
    {"name": "2. Fade-Only (TP1.2% / SL0.8%)", "use_trend": False, "use_fade": True, "fade_tp": 0.012, "fade_sl": 0.008},
    {"name": "3. Fade-Only (TP1.5% / SL1.0%)", "use_trend": False, "use_fade": True, "fade_tp": 0.015, "fade_sl": 0.010},
    {"name": "4. Trend + Fade (TP1.2% / SL0.8%)", "use_trend": True, "use_fade": True, "fade_tp": 0.012, "fade_sl": 0.008},
    {"name": "5. Trend + Fade (TP1.5% / SL1.0%)", "use_trend": True, "use_fade": True, "fade_tp": 0.015, "fade_sl": 0.010},
]


def simulate(data: dict, cfg: dict):
    o, h, l, c = data["o"], data["h"], data["l"], data["c"]
    dc_hi, dc_lo = data["dc_hi"], data["dc_lo"]
    adx, er = data["adx"], data["er"]
    trend_long, trend_short = data["trend_long"], data["trend_short"]
    n = len(c)

    use_trend = cfg["use_trend"]
    use_fade = cfg["use_fade"]
    fade_tp = cfg["fade_tp"]
    fade_sl = cfg["fade_sl"]

    tr_stats = []
    tr_tv = []

    pos = 0
    mode_active = ""  # "trend" atau "fade"
    ep = 0.0
    sl_px = 0.0
    tp_px = 0.0
    rd = 0.015
    je = 0

    for i in range(205, n - 1):
        # 1. Kelola posisi aktif pada bar i
        if pos == 1:
            max_hold = MAX_HOLD_TREND if mode_active == "trend" else MAX_HOLD_FADE
            if l[i] <= sl_px:
                exit_px = min(o[i], sl_px)
                tr_stats.append((je, 1, ep, exit_px, rd, f"sl_{mode_active}"))
                tr_tv.append((je, i, 1, ep, exit_px, rd, f"sl_{mode_active}"))
                pos = 0
            elif h[i] >= tp_px:
                exit_px = max(o[i], tp_px)
                tr_stats.append((je, 1, ep, exit_px, rd, f"tp_{mode_active}"))
                tr_tv.append((je, i, 1, ep, exit_px, rd, f"tp_{mode_active}"))
                pos = 0
            elif (i - je) >= max_hold:
                tr_stats.append((je, 1, ep, o[i + 1], rd, f"lepas_{mode_active}"))
                tr_tv.append((je, i + 1, 1, ep, o[i + 1], rd, f"lepas_{mode_active}"))
                pos = 0

        elif pos == -1:
            max_hold = MAX_HOLD_TREND if mode_active == "trend" else MAX_HOLD_FADE
            if h[i] >= sl_px:
                exit_px = max(o[i], sl_px)
                tr_stats.append((je, -1, ep, exit_px, rd, f"sl_{mode_active}"))
                tr_tv.append((je, i, -1, ep, exit_px, rd, f"sl_{mode_active}"))
                pos = 0
            elif l[i] <= tp_px:
                exit_px = min(o[i], tp_px)
                tr_stats.append((je, -1, ep, exit_px, rd, f"tp_{mode_active}"))
                tr_tv.append((je, i, -1, ep, exit_px, rd, f"tp_{mode_active}"))
                pos = 0
            elif (i - je) >= max_hold:
                tr_stats.append((je, -1, ep, o[i + 1], rd, f"lepas_{mode_active}"))
                tr_tv.append((je, i + 1, -1, ep, o[i + 1], rd, f"lepas_{mode_active}"))
                pos = 0

        # 2. Evaluasi sinyal di close bar i -> eksekusi di bar i+1
        if pos == 0:
            is_trending = (adx[i] > 25.0) and (er[i] > 0.28)
            is_choppy = (adx[i] < 20.0) and (er[i] < 0.20)

            # A. Trend Breakout (eksekusi di open bar i+1)
            if use_trend and is_trending:
                if trend_long[i]:
                    pos = 1
                    mode_active = "trend"
                    ep = o[i + 1]
                    rd = 0.015
                    sl_px = ep * (1.0 - rd)
                    tp_px = ep * 1.030
                    je = i + 1
                    continue
                elif trend_short[i]:
                    pos = -1
                    mode_active = "trend"
                    ep = o[i + 1]
                    rd = 0.015
                    sl_px = ep * (1.0 + rd)
                    tp_px = ep * 0.970
                    je = i + 1
                    continue

            # B. Fade False Breakout (Limit Order ditaruh di dc_hi / dc_lo untuk bar i+1)
            if use_fade and is_choppy:
                # Level Donchian yang berlaku saat bar i+1 berjalan (dihitung dari 20 bar s/d bar i)
                lvl_hi = max(dc_hi[i], h[i])
                lvl_lo = min(dc_lo[i], l[i])
                # Pastikan lebar saluran minimal 1.2% agar bukan kompresi ekstrem
                if (lvl_hi - lvl_lo) / lvl_lo >= 0.012:
                    # Cek apakah di bar i+1 harga menyentuh limit sell di lvl_hi atau limit buy di lvl_lo
                    hit_hi = (o[i + 1] < lvl_hi) and (h[i + 1] >= lvl_hi)
                    hit_lo = (o[i + 1] > lvl_lo) and (l[i + 1] <= lvl_lo)

                    if hit_hi and not hit_lo:
                        # Limit Sell terisi di lvl_hi; cek langsung apakah bar i+1 juga menyentuh SL intrabar
                        ep = lvl_hi
                        rd = fade_sl
                        sl_px = ep * (1.0 + fade_sl)
                        tp_px = ep * (1.0 - fade_tp)
                        je = i + 1
                        if h[i + 1] >= sl_px:
                            tr_stats.append((je, -1, ep, sl_px, rd, "sl_fade"))
                            tr_tv.append((je, i + 1, -1, ep, sl_px, rd, "sl_fade"))
                        else:
                            pos = -1
                            mode_active = "fade"
                    elif hit_lo and not hit_hi:
                        # Limit Buy terisi di lvl_lo; cek langsung apakah bar i+1 juga menyentuh SL intrabar
                        ep = lvl_lo
                        rd = fade_sl
                        sl_px = ep * (1.0 - fade_sl)
                        tp_px = ep * (1.0 + fade_tp)
                        je = i + 1
                        if l[i + 1] <= sl_px:
                            tr_stats.append((je, 1, ep, sl_px, rd, "sl_fade"))
                            tr_tv.append((je, i + 1, 1, ep, sl_px, rd, "sl_fade"))
                        else:
                            pos = 1
                            mode_active = "fade"

    if pos != 0:
        tr_stats.append((je, pos, ep, c[-1], rd, "end"))
        tr_tv.append((je, n - 1, pos, ep, c[-1], rd, "end"))

    return tr_stats, tr_tv


def sim_small_account(tr_stats, fee_trend: float, fee_fade: float, leverage: float, init_eq: float = 6.0):
    """Simulasi akun $6 Cross Margin (margin $3 / max 50% equity, 1 posisi aktif)."""
    eq = init_eq
    peak = init_eq
    max_dd = 0.0
    for _, pos, ep, px, _, tag in tr_stats:
        if eq <= 0.05:
            eq = 0.0
            max_dd = 100.0
            break
        margin = min(3.0, eq * 0.5)
        notional = margin * leverage
        raw_ret = ((px - ep) / ep) if pos == 1 else ((ep - px) / ep)
        fee_used = fee_fade if "fade" in tag else fee_trend
        pnl = notional * (raw_ret - fee_used)
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

        # Hitung R dengan biaya Maker realistis:
        # Trend: 0.10% (Limit TP/Market SL) atau 0.22% (Market)
        # Fade : 0.06% (Post-Only Limit Entry + Limit TP, sesekali Market SL)
        rets_realistic = []
        for _, pos, ep, px, rd, tag in tr_stats:
            raw_ret = ((px - ep) / ep) if pos == 1 else ((ep - px) / ep)
            fee_u = 0.0006 if "fade" in tag else 0.0010
            rets_realistic.append((raw_ret - fee_u) / rd)
        r_mkr_tot = round(float(np.sum(rets_realistic)), 1) if rets_realistic else 0.0
        best_tv.append((r_mkr_tot, nm, tr_tv))

        eq5_mkt, dd5_mkt = sim_small_account(tr_stats, 0.0022, 0.0010, 5.0)
        eq10_mkt, dd10_mkt = sim_small_account(tr_stats, 0.0022, 0.0010, 10.0)
        eq5_mkr, dd5_mkr = sim_small_account(tr_stats, 0.0010, 0.0006, 5.0)
        eq10_mkr, dd10_mkr = sim_small_account(tr_stats, 0.0010, 0.0006, 10.0)

        small_rows.append({
            "varian": nm,
            "trade": len(tr_stats),
            "R_std(0.22%)": r["bersihR_total"] if r else 0,
            "R_real_Mkr": r_mkr_tot,
            "$6_5x_Mkt": f"${eq5_mkt} (DD {dd5_mkt}%)",
            "$6_10x_Mkt": f"${eq10_mkt} (DD {dd10_mkt}%)",
            "$6_5x_Mkr": f"${eq5_mkr} (DD {dd5_mkr}%)",
            "$6_10x_Mkr": f"${eq10_mkr} (DD {dd10_mkr}%)",
        })

    fence = "`" * 3
    bt._emit(
        "\n### Tahap 1: Fade False Breakout (Choppy) + Trend Breakout (Trending) — BTC 1h\n"
        + fence + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + fence
    )
    bt._emit(
        "\n### Simulasi Biaya Maker Realistis & Akun $6 Cross Margin (Margin $3, Lev 5x & 10x)\n"
        + fence + "\n" + pd.DataFrame(small_rows).to_string(index=False) + "\n" + fence
    )
    best_tv.sort(key=lambda x: x[0], reverse=True)
    for _, nm, tr_tv in best_tv[:2]:
        bt._emit(generate_tv_report(tr_tv, data["df"], "1h", nm))
    bt._emit(
        "\n### Per tahun (Tahap 1 - bersihR standar 0.22%)\n"
        + fence + "\n" + pd.DataFrame(yrs).to_string(index=False) + "\n" + fence
    )


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
