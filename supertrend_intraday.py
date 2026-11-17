#!/usr/bin/env python3
"""
TRIPLE SUPERTREND INTRADAY & SCALPING (10-1, 11-2, 12-3).
Pullback Re-Alignment + Hard SL Cap + Fixed TP (1.5R / 2.0R / 2.5R) + Auto-Breakeven di +1.0R.
Simulasi akun riil: $6 Equity | $3 Initial Margin | Cross Margin | Leverage 5x & 10x.
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
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)

    sts = [calc_supertrend(df, length, mult) for length, mult in ST_PARAMS]
    _, _, tr1 = sts[0]
    _, _, tr2 = sts[1]
    ub3, lb3, tr3 = sts[2]

    conf = np.where((tr1 == 1) & (tr2 == 1) & (tr3 == 1), 1,
           np.where((tr1 == -1) & (tr2 == -1) & (tr3 == -1), -1, 0))

    adx14 = calc_adx(df, 14)
    ema200 = pd.Series(c).ewm(span=200, adjust=False).mean().to_numpy(float)

    return {
        "df": df,
        "t": t,
        "step": int(np.median(np.diff(t))),
        "o": o,
        "h": h,
        "l": l,
        "c": c,
        "tr1": tr1,
        "tr3": tr3,
        "conf": conf,
        "up3": ub3,
        "dn3": lb3,
        "adx": adx14,
        "ema200": ema200,
        "yr": df["ts"].dt.year.to_numpy(),
        "months": (df["ts"].iloc[-1] - df["ts"].iloc[0]).days / 30,
    }


def align_htf(e, h):
    idx = np.searchsorted(h["t"] + h["step"], e["t"] + e["step"], side="right") - 1
    clipped = np.clip(idx, 0, None)
    valid = idx >= 0
    h_conf = np.where(valid, h["conf"][clipped], 0)
    h_short_ok = np.where(valid, h["c"][clipped] < h["ema200"][clipped], False)
    return h_conf, h_short_ok


def run_intraday_trades(e, filt_dir, filt_short_ok, sl_min_pct: float, sl_max_pct: float, tp_rr: float):
    o, h, l, c = e["o"], e["h"], e["l"], e["c"]
    tr1, tr3 = e["tr1"], e["tr3"]
    up3, dn3 = e["up3"], e["dn3"]
    adx, ema200 = e["adx"], e["ema200"]
    n = len(c)

    out_stats, out_tv = [], []
    pos = 0
    je = 0
    ep = 0.0
    sl_px = 0.0
    tp_px = 0.0
    rd_pct = 0.0
    rd_dist = 0.0
    be_moved = False

    for i in range(200, n - 1):
        # 1. Evaluasi posisi terbuka di bar i
        if pos == 1:
            if l[i] <= sl_px:
                px = min(o[i], sl_px)
                reason = "be" if be_moved else "sl"
                out_stats.append((je, 1, ep, px, rd_pct, reason))
                out_tv.append((je, i, 1, ep, px, rd_pct, reason))
                pos = 0
            elif h[i] >= tp_px:
                px = max(o[i], tp_px)
                out_stats.append((je, 1, ep, px, rd_pct, "tp"))
                out_tv.append((je, i, 1, ep, px, rd_pct, "tp"))
                pos = 0
            else:
                if (not be_moved) and h[i] >= ep + rd_dist:
                    sl_px = ep * 1.0025  # Auto-Breakeven +0.25% tutup fee + slippage
                    be_moved = True
                if tr3[i] == -1:
                    out_stats.append((je, 1, ep, o[i + 1], rd_pct, "lepas"))
                    out_tv.append((je, i + 1, 1, ep, o[i + 1], rd_pct, "lepas"))
                    pos = 0

        elif pos == -1:
            if h[i] >= sl_px:
                px = max(o[i], sl_px)
                reason = "be" if be_moved else "sl"
                out_stats.append((je, -1, ep, px, rd_pct, reason))
                out_tv.append((je, i, -1, ep, px, rd_pct, reason))
                pos = 0
            elif l[i] <= tp_px:
                px = min(o[i], tp_px)
                out_stats.append((je, -1, ep, px, rd_pct, "tp"))
                out_tv.append((je, i, -1, ep, px, rd_pct, "tp"))
                pos = 0
            else:
                if (not be_moved) and l[i] <= ep - rd_dist:
                    sl_px = ep * 0.9975  # Auto-Breakeven -0.25% tutup fee + slippage
                    be_moved = True
                if tr3[i] == 1:
                    out_stats.append((je, -1, ep, o[i + 1], rd_pct, "lepas"))
                    out_tv.append((je, i + 1, -1, ep, o[i + 1], rd_pct, "lepas"))
                    pos = 0

        # 2. Cari sinyal baru saat flat (evaluasi close bar i -> eksekusi open bar i+1)
        if pos == 0 and adx[i] > 20.0:
            long_sig = (
                filt_dir[i] == 1
                and tr3[i] == 1
                and tr1[i] == 1
                and tr1[i - 1] == -1
            )
            short_sig = (
                filt_dir[i] == -1
                and filt_short_ok[i]
                and c[i] < ema200[i]
                and tr3[i] == -1
                and tr1[i] == -1
                and tr1[i - 1] == 1
            )

            if long_sig:
                cand_ep = o[i + 1]
                cand_sl = dn3[i]
                cand_dist = cand_ep - cand_sl
                cand_rd = cand_dist / cand_ep if cand_ep > 0 else 0.0
                if sl_min_pct <= cand_rd <= sl_max_pct:
                    pos = 1
                    je = i + 1
                    ep = cand_ep
                    sl_px = cand_sl
                    rd_dist = cand_dist
                    rd_pct = max(cand_rd, 1e-4)
                    tp_px = ep + tp_rr * rd_dist
                    be_moved = False

            elif short_sig:
                cand_ep = o[i + 1]
                cand_sl = up3[i]
                cand_dist = cand_sl - cand_ep
                cand_rd = cand_dist / cand_ep if cand_ep > 0 else 0.0
                if sl_min_pct <= cand_rd <= sl_max_pct:
                    pos = -1
                    je = i + 1
                    ep = cand_ep
                    sl_px = cand_sl
                    rd_dist = cand_dist
                    rd_pct = max(cand_rd, 1e-4)
                    tp_px = ep - tp_rr * rd_dist
                    be_moved = False

    if pos != 0:
        out_stats.append((je, pos, ep, c[-1], rd_pct, "end"))
        out_tv.append((je, n - 1, pos, ep, c[-1], rd_pct, "end"))

    return out_stats, out_tv


def sim_cross_6usd(tr_stats, lev: float, init_eq: float = 6.0, init_margin: float = 3.0, cost_pct: float = 0.0022):
    """
    Simulasi akun Cross Margin:
      - Modal awal $6.00 (seluruh equity jadi jaminan / collateral).
      - Initial margin $3.00 (atau 50% equity jika saldo turun di bawah $6).
      - Notional posisi = margin * leverage (5x = $15, 10x = $30).
      - Cek likuidasi jika equity + worst_loss <= notional * 0.5% (MMR).
    """
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

        # Cek skenario terburuk menyentuh SL sebelum exit
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
    d_raw = {tf: e2.load(ex, tf) for tf in ("5m", "15m", "1h", "4h")}
    D = {tf: prepare_tf(df) for tf, df in d_raw.items()}

    f_5m_15m, s_5m_15m = align_htf(D["5m"], D["15m"])
    f_5m_1h, s_5m_1h = align_htf(D["5m"], D["1h"])
    f_5m_both = np.where((f_5m_15m == f_5m_1h) & (f_5m_15m != 0), f_5m_15m, 0)
    s_5m_both = s_5m_15m & s_5m_1h

    f_15m_1h, s_15m_1h = align_htf(D["15m"], D["1h"])
    f_15m_4h, s_15m_4h = align_htf(D["15m"], D["4h"])
    f_15m_both = np.where((f_15m_1h == f_15m_4h) & (f_15m_1h != 0), f_15m_1h, 0)
    s_15m_both = s_15m_1h & s_15m_4h

    f_1h_4h, s_1h_4h = align_htf(D["1h"], D["4h"])

    setups = [
        ("5m", "F:15m", f_5m_15m, s_5m_15m, 0.0050, 0.0150),
        ("5m", "F:1h", f_5m_1h, s_5m_1h, 0.0050, 0.0150),
        ("5m", "F:15m+1h", f_5m_both, s_5m_both, 0.0050, 0.0150),
        ("15m", "F:1h", f_15m_1h, s_15m_1h, 0.0070, 0.0200),
        ("15m", "F:4h", f_15m_4h, s_15m_4h, 0.0070, 0.0200),
        ("15m", "F:1h+4h", f_15m_both, s_15m_both, 0.0070, 0.0200),
        ("1h", "F:4h", f_1h_4h, s_1h_4h, 0.0080, 0.0250),
    ]

    rows, sim_rows, yrs, tv_candidates = [], [], [], []
    for tf, flabel, fd, fs, smin, smax in setups:
        e = D[tf]
        for tp_rr in (1.5, 2.0, 2.5):
            label = f"{flabel} | TP:{tp_rr}R+BE"
            tr_stats, tr_tv = run_intraday_trades(e, fd, fs, smin, smax, tp_rr)
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
                    "hit%": r["hit%"],
                    "bersihR": r["bersihR_total"],
                    "SL%": r["jarak_SL%"],
                    "End_5x": end5,
                    "MaxDD_5x": dd5,
                    "End_10x": end10,
                    "MaxDD_10x": dd10,
                })
                if r["bersihR_total"] > 0:
                    tv_candidates.append((r["bersihR_total"], tr_tv, e["df"], tf, label))

    fence = "`" * 3
    bt._emit(
        "\n### Triple Supertrend Intraday & 5m Scalping (Pullback Re-Alignment + SL Cap + Fixed TP + BE)\n"
        + fence + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + fence
    )
    bt._emit(
        "\n### Simulasi Akun Riil: $6 Equity | $3 Initial Margin | Cross Margin | Lev 5x vs 10x\n"
        + fence + "\n" + pd.DataFrame(sim_rows).to_string(index=False) + "\n" + fence
    )
    for _, tr_tv, df_raw, tf, label in sorted(tv_candidates, key=lambda x: x[0], reverse=True)[:4]:
        bt._emit(generate_tv_report(tr_tv, df_raw, tf, label))
    bt._emit(
        "\n### Per tahun (Intraday - bersihR)\n"
        + fence + "\n" + pd.DataFrame(yrs).to_string(index=False) + "\n" + fence
    )


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
