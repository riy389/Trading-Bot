#!/usr/bin/env python3
"""
UJI INDIKATOR NON-LAGGING & LOW-LAGGING (SATUAN & KONFLUENSI).
Dilengkapi filter Volume & Volatilitas (ATR), target gerak harga 1.5% & 3.0% (R:R 1:2),
serta simulasi akun riil $6 Cross Margin ($3 Initial Margin, Lev 5x & 10x).
"""
import numpy as np
import pandas as pd

import backtest as bt
import eventtree2 as e2
from zerolag import stats
from tv_metrics import generate_tv_report


def rma(series: np.ndarray, length: int) -> np.ndarray:
    out = np.full(len(series), np.nan, dtype=float)
    if len(series) < length:
        return out
    out[length - 1] = np.mean(series[:length])
    alpha = 1.0 / length
    for i in range(length, len(series)):
        out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
    return out


def calc_rsi(c: np.ndarray, period: int = 14) -> np.ndarray:
    diff = np.diff(c, prepend=c[0])
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    avg_g = rma(gain, period)
    avg_l = rma(loss, period)
    rs = avg_g / np.maximum(avg_l, 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return np.nan_to_num(rsi, nan=50.0)


def calc_zlema(c: np.ndarray, period: int = 21) -> np.ndarray:
    lag = int((period - 1) / 2)
    de_lagged = np.copy(c)
    if lag > 0:
        de_lagged[lag:] = c[lag:] + (c[lag:] - c[:-lag])
    return pd.Series(de_lagged).ewm(span=period, adjust=False).mean().to_numpy(float)


def prepare_indicators(df: pd.DataFrame, min_atr_pct: float):
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float) if "volume" in df.columns else np.ones_like(c)
    n = len(c)

    # 1. Volatilitas (ATR 14) & Volume Filter
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr14 = rma(tr, 14)
    atr_pct = np.nan_to_num(atr14 / np.maximum(c, 1e-8), nan=0.0)
    atr_ma = pd.Series(atr14).rolling(20).mean().bfill().to_numpy(float)

    vol_ma20 = pd.Series(v).rolling(20).mean().bfill().to_numpy(float)
    vol_ok = v > (vol_ma20 * 1.30)
    volat_ok = (atr_pct >= min_atr_pct) & (atr14 >= atr_ma * 0.90)
    base_filter = vol_ok & volat_ok

    # 2. Donchian Channel (20) - Pure Price Action Breakout (0-lag)
    dc_hi = pd.Series(h).shift(1).rolling(20).max().to_numpy(float)
    dc_lo = pd.Series(l).shift(1).rolling(20).min().to_numpy(float)
    dc_long = (c > dc_hi) & (np.roll(c, 1) <= np.roll(dc_hi, 1))
    dc_short = (c < dc_lo) & (np.roll(c, 1) >= np.roll(dc_lo, 1))

    # 3. Liquidity Sweep + Rejection (15-bar swing sweep)
    sw_hi = pd.Series(h).shift(1).rolling(15).max().to_numpy(float)
    sw_lo = pd.Series(l).shift(1).rolling(15).min().to_numpy(float)
    sweep_long = (l < sw_lo) & (c > sw_lo) & (c > o)
    sweep_short = (h > sw_hi) & (c < sw_hi) & (c < o)

    # 4. Bollinger Bands (20, 2) Squeeze + Impulse Breakout
    bb_mid = pd.Series(c).rolling(20).mean().to_numpy(float)
    bb_std = pd.Series(c).rolling(20).std().to_numpy(float)
    bb_up = bb_mid + 2.0 * bb_std
    bb_lo = bb_mid - 2.0 * bb_std
    bb_width = (bb_up - bb_lo) / np.maximum(bb_mid, 1e-8)
    bb_w_ma = pd.Series(bb_width).shift(1).rolling(20).mean().to_numpy(float)
    was_squeezed = bb_width < bb_w_ma
    body = np.abs(c - o)
    impulse = body >= (1.2 * np.nan_to_num(atr14, nan=0.0))
    bb_long = was_squeezed & (c > bb_up) & impulse
    bb_short = was_squeezed & (c < bb_lo) & impulse

    # 5. Zero-Lag EMA (21) Slope Flip
    zlema = calc_zlema(c, 21)
    zl_up = zlema > np.roll(zlema, 1)
    zl_dn = zlema < np.roll(zlema, 1)
    zl_long = zl_up & (~np.roll(zl_up, 1)) & (c > zlema)
    zl_short = zl_dn & (~np.roll(zl_dn, 1)) & (c < zlema)

    # 6. RSI (14) Momentum Breakout (55 / 45)
    rsi = calc_rsi(c, 14)
    rsi_long = (rsi > 55.0) & (np.roll(rsi, 1) <= 55.0)
    rsi_short = (rsi < 45.0) & (np.roll(rsi, 1) >= 45.0)

    # Konfluensi Gabungan
    signals = {
        "1_Donchian_20": (dc_long & base_filter, dc_short & base_filter),
        "2_LiqSweep": (sweep_long & base_filter, sweep_short & base_filter),
        "3_BBSqueeze": (bb_long & base_filter, bb_short & base_filter),
        "4_ZLEMA_21": (zl_long & base_filter, zl_short & base_filter),
        "5_RSI_Mom": (rsi_long & base_filter, rsi_short & base_filter),
        "Combo_BB_RSI": (
            (c > bb_up) & (rsi > 55.0) & impulse & base_filter,
            (c < bb_lo) & (rsi < 45.0) & impulse & base_filter,
        ),
        "Combo_DC_ATR": (
            dc_long & (atr14 > atr_ma * 1.10) & base_filter,
            dc_short & (atr14 > atr_ma * 1.10) & base_filter,
        ),
        "Combo_Sweep_ZL": (
            sweep_long & zl_up & base_filter,
            sweep_short & zl_dn & base_filter,
        ),
        "Combo_All_Conf": (
            (c > bb_up) & zl_up & (rsi > 55.0) & base_filter,
            (c < bb_lo) & zl_dn & (rsi < 45.0) & base_filter,
        ),
    }

    # Bersihkan bar warmup 50 candle awal
    for k in signals:
        lg, sh = signals[k]
        lg[:50] = False
        sh[:50] = False
        signals[k] = (lg, sh)

    return {
        "df": df,
        "o": o,
        "h": h,
        "l": l,
        "c": c,
        "signals": signals,
        "yr": df["ts"].dt.year.to_numpy(),
        "months": (df["ts"].iloc[-1] - df["ts"].iloc[0]).days / 30,
    }


def simulate_fixed_pct(e, long_sig, short_sig, tp_pct: float, sl_pct: float, max_hold_bars: int = 48):
    """
    Eksekusi non-repaint: sinyal dievaluasi saat close bar i -> masuk di open bar i+1.
    Hard SL dan TP dicek intrabar (high/low), dengan batas waktu max_hold_bars agar modal tidak nyangkut.
    """
    o, h, l, c = e["o"], e["h"], e["l"], e["c"]
    n = len(c)

    out_stats, out_tv = [], []
    pos = 0
    je = 0
    ep = 0.0
    sl_px = 0.0
    tp_px = 0.0

    for i in range(50, n - 1):
        if pos == 1:
            if l[i] <= sl_px:
                px = min(o[i], sl_px)
                out_stats.append((je, 1, ep, px, sl_pct, "sl"))
                out_tv.append((je, i, 1, ep, px, sl_pct, "sl"))
                pos = 0
            elif h[i] >= tp_px:
                px = max(o[i], tp_px)
                out_stats.append((je, 1, ep, px, sl_pct, "tp"))
                out_tv.append((je, i, 1, ep, px, sl_pct, "tp"))
                pos = 0
            elif (i - je) >= max_hold_bars:
                out_stats.append((je, 1, ep, o[i + 1], sl_pct, "lepas"))
                out_tv.append((je, i + 1, 1, ep, o[i + 1], sl_pct, "lepas"))
                pos = 0

        elif pos == -1:
            if h[i] >= sl_px:
                px = max(o[i], sl_px)
                out_stats.append((je, -1, ep, px, sl_pct, "sl"))
                out_tv.append((je, i, -1, ep, px, sl_pct, "sl"))
                pos = 0
            elif l[i] <= tp_px:
                px = min(o[i], tp_px)
                out_stats.append((je, -1, ep, px, sl_pct, "tp"))
                out_tv.append((je, i, -1, ep, px, sl_pct, "tp"))
                pos = 0
            elif (i - je) >= max_hold_bars:
                out_stats.append((je, -1, ep, o[i + 1], sl_pct, "lepas"))
                out_tv.append((je, i + 1, -1, ep, o[i + 1], sl_pct, "lepas"))
                pos = 0

        if pos == 0:
            if long_sig[i] and not short_sig[i]:
                pos = 1
                je = i + 1
                ep = o[i + 1]
                sl_px = ep * (1.0 - sl_pct)
                tp_px = ep * (1.0 + tp_pct)
            elif short_sig[i] and not long_sig[i]:
                pos = -1
                je = i + 1
                ep = o[i + 1]
                sl_px = ep * (1.0 + sl_pct)
                tp_px = ep * (1.0 - tp_pct)

    if pos != 0:
        out_stats.append((je, pos, ep, c[-1], sl_pct, "end"))
        out_tv.append((je, n - 1, pos, ep, c[-1], sl_pct, "end"))

    return out_stats, out_tv


def sim_cross_6usd(tr_stats, lev: float, init_eq: float = 6.0, init_margin: float = 3.0, cost_pct: float = 0.0022):
    eq = init_eq
    peak = init_eq
    max_dd = 0.0
    is_mc = False

    for _, pos, ep, px, rd, _ in tr_stats:
        if eq < 1.0:
            is_mc = True
            eq = 0.0
            max_dd = 100.0
            break

        margin = min(init_margin, eq * 0.5)
        notional = margin * lev

        worst_pnl = notional * (-rd - cost_pct * 0.5)
        if eq + worst_pnl <= notional * 0.005:
            is_mc = True
            eq = 0.0
            max_dd = 100.0
            break

        gross_ret = ((px - ep) / ep) if pos == 1 else ((ep - px) / ep)
        net_ret = gross_ret - cost_pct
        eq += notional * net_ret

        if eq <= notional * 0.005:
            is_mc = True
            eq = 0.0
            max_dd = 100.0
            break

        if eq > peak:
            peak = eq
        dd = ((peak - eq) / peak) * 100.0 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    end_str = "MC!" if is_mc else f"${eq:.2f}"
    return end_str, f"{max_dd:.1f}%"


def run(ex):
    tf_configs = [
        ("15m", 0.0035, 64),  # min ATR 0.35%, max hold 16 jam (64 bar)
        ("1h", 0.0060, 48),   # min ATR 0.60%, max hold 48 jam (48 bar)
    ]
    targets = [
        ("TP1.5%_SL0.75%", 0.015, 0.0075),
        ("TP3.0%_SL1.50%", 0.030, 0.0150),
    ]

    fence = "`" * 3
    for tf, min_atr, max_hold in tf_configs:
        df_raw = e2.load(ex, tf)
        e = prepare_indicators(df_raw, min_atr)
        rows, sim_rows, yrs, tv_candidates = [], [], [], []

        for sig_name, (lg, sh) in e["signals"].items():
            for tgt_label, tp_pct, sl_pct in targets:
                label = f"{sig_name} | {tgt_label}"
                tr_stats, tr_tv = simulate_fixed_pct(e, lg, sh, tp_pct, sl_pct, max_hold)
                r, y = stats(tr_stats, e["yr"], tf, label, e["months"])
                if r:
                    rows.append(r)
                    yrs += y
                    end5, dd5 = sim_cross_6usd(tr_stats, 5.0)
                    end10, dd10 = sim_cross_6usd(tr_stats, 10.0)
                    sim_rows.append({
                        "TF": tf,
                        "varian": label,
                        "trade": r["trade"],
                        "bln": r["per_bulan"],
                        "hit%": r["hit%"],
                        "bersihR": r["bersihR_total"],
                        "End_5x": end5,
                        "DD_5x": dd5,
                        "End_10x": end10,
                        "DD_10x": dd10,
                    })
                    if r["bersihR_total"] > 0:
                        tv_candidates.append((r["bersihR_total"], tr_tv, df_raw, tf, label))

        bt._emit(
            f"\n### Non-Lagging & Confluence Test — TF {tf} (Vol > 1.3x MA20 + ATR Filter | Fee 0.22%)\n"
            + fence + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + fence
        )
        bt._emit(
            f"\n### Simulasi Akun $6 Cross Margin ($3 Initial Margin | 5x & 10x) — TF {tf}\n"
            + fence + "\n" + pd.DataFrame(sim_rows).to_string(index=False) + "\n" + fence
        )
        for _, tr_tv, dfr, t_f, lbl in sorted(tv_candidates, key=lambda x: x[0], reverse=True)[:3]:
            bt._emit(generate_tv_report(tr_tv, dfr, t_f, lbl))
        bt._emit(
            f"\n### Per tahun ({tf} - bersihR)\n"
            + fence + "\n" + pd.DataFrame(yrs).to_string(index=False) + "\n" + fence
        )


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
