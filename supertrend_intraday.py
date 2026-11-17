import os
import numpy as np
import pandas as pd
import eventtree2 as e2
import backtest as bt
from tv_metrics import generate_tv_report

FEE_RT = 0.0022  # 0.12% fee + 0.10% slippage round-trip


def calc_rma(arr, period):
    out = np.full_like(arr, np.nan, dtype=np.float64)
    if len(arr) < period:
        return out
    out[period - 1] = np.mean(arr[:period])
    alpha = 1.0 / period
    for i in range(period, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def calc_ema(arr, period):
    out = np.full_like(arr, np.nan, dtype=np.float64)
    if len(arr) < period:
        return out
    out[period - 1] = np.mean(arr[:period])
    alpha = 2.0 / (period + 1.0)
    for i in range(period, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def calc_adx(h, l, c, period=14):
    n = len(c)
    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        up_move = h[i] - h[i - 1]
        down_move = l[i - 1] - l[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    atr = calc_rma(tr, period)
    p_di = 100.0 * calc_rma(plus_dm, period) / np.where(atr == 0, np.nan, atr)
    m_di = 100.0 * calc_rma(minus_dm, period) / np.where(atr == 0, np.nan, atr)
    denom = p_di + m_di
    dx = 100.0 * np.abs(p_di - m_di) / np.where(denom == 0, np.nan, denom)
    return calc_rma(dx, period)


def calc_supertrend(h, l, c, atr_len, mult):
    n = len(c)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = calc_rma(tr, atr_len)
    hl2 = 0.5 * (h + l)

    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    trend = np.zeros(n, dtype=np.int8)

    for i in range(atr_len - 1, n):
        bu = hl2[i] + mult * atr[i]
        bl = hl2[i] - mult * atr[i]
        if i == atr_len - 1 or np.isnan(upper[i - 1]):
            upper[i] = bu
            lower[i] = bl
            trend[i] = 1 if c[i] >= hl2[i] else -1
            continue

        prev_u = upper[i - 1]
        prev_l = lower[i - 1]
        prev_c = c[i - 1]
        prev_t = trend[i - 1]

        lower[i] = bl if (bl > prev_l or prev_c < prev_l) else prev_l
        upper[i] = bu if (bu < prev_u or prev_c > prev_u) else prev_u

        if prev_t == -1 and c[i] > prev_u:
            trend[i] = 1
        elif prev_t == 1 and c[i] < prev_l:
            trend[i] = -1
        else:
            trend[i] = prev_t

    return trend, lower, upper


def prepare_tf(df):
    ts = df["timestamp"].to_numpy(dtype=np.int64)
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)

    t1, lo1, up1 = calc_supertrend(h, l, c, 10, 1.0)
    t2, lo2, up2 = calc_supertrend(h, l, c, 11, 2.0)
    t3, lo3, up3 = calc_supertrend(h, l, c, 12, 3.0)
    adx = calc_adx(h, l, c, 14)
    ema200 = calc_ema(c, 200)

    all_bull = (t1 == 1) & (t2 == 1) & (t3 == 1)
    all_bear = (t1 == -1) & (t2 == -1) & (t3 == -1)

    dt = pd.to_datetime(ts, unit="ms", utc=True)
    yr = dt.year.to_numpy()
    months = max(1.0, (ts[-1] - ts[0]) / (1000.0 * 86400.0 * 30.4375))

    return {
        "ts": ts, "o": o, "h": h, "l": l, "c": c,
        "t1": t1, "t2": t2, "t3": t3,
        "lo3": lo3, "up3": up3,
        "adx": adx, "ema200": ema200,
        "all_bull": all_bull, "all_bear": all_bear,
        "yr": yr, "months": months,
    }


def align_mtf(exec_ts, exec_step_ms, htf_data, htf_step_ms):
    n = len(exec_ts)
    dir_arr = np.zeros(n, dtype=np.int8)
    ema_ok_short = np.zeros(n, dtype=bool)

    htf_close_ts = htf_data["ts"] + htf_step_ms
    idx = np.searchsorted(htf_close_ts, exec_ts + exec_step_ms, side="right") - 1

    valid = idx >= 0
    v_idx = idx[valid]

    bull = htf_data["all_bull"][v_idx]
    bear = htf_data["all_bear"][v_idx]
    d_sub = np.zeros(len(v_idx), dtype=np.int8)
    d_sub[bull] = 1
    d_sub[bear] = -1
    dir_arr[valid] = d_sub

    c_htf = htf_data["c"][v_idx]
    e_htf = htf_data["ema200"][v_idx]
    ema_ok_short[valid] = (~np.isnan(e_htf)) & (c_htf < e_htf)

    return dir_arr, ema_ok_short


def simulate_trades(e, filt_dir, filt_short_ok, sl_min_pct, sl_max_pct, tp_rr, use_be=True):
    o, h, l, c = e["o"], e["h"], e["l"], e["c"]
    t1, t3 = e["t1"], e["t3"]
    lo3, up3 = e["lo3"], e["up3"]
    adx, ema200 = e["adx"], e["ema200"]
    n = len(c)

    pos = 0
    je = 0
    ep = 0.0
    sl = 0.0
    tp = 0.0
    rd = 0.0
    be_moved = False
    trades = []

    for i in range(205, n - 1):
        # 1. Kelola posisi aktif di bar i (menggunakan high/low bar i)
        if pos == 1:
            # Cek SL terlebih dahulu (konservatif bila SL & TP kena di bar yang sama)
            if l[i] <= sl:
                px = min(o[i], sl)
                reason = "be" if be_moved else "sl"
                trades.append((je, i, 1, ep, px, rd, reason))
                pos = 0
            elif h[i] >= tp:
                px = max(o[i], tp)
                trades.append((je, i, 1, ep, px, rd, "tp"))
                pos = 0
            else:
                if use_be and (not be_moved) and h[i] >= ep + 1.0 * rd:
                    sl = ep * 1.0025  # Breakeven +0.25% tutup fee & slippage
                    be_moved = True
                if t3[i] == -1:
                    # Garis lambat 12-3 flip berlawanan -> keluar di open i+1
                    trades.append((je, i + 1, 1, ep, o[i + 1], rd, "flip12"))
                    pos = 0

        elif pos == -1:
            if h[i] >= sl:
                px = max(o[i], sl)
                reason = "be" if be_moved else "sl"
                trades.append((je, i, -1, ep, px, rd, reason))
                pos = 0
            elif l[i] <= tp:
                px = min(o[i], tp)
                trades.append((je, i, -1, ep, px, rd, "tp"))
                pos = 0
            else:
                if use_be and (not be_moved) and l[i] <= ep - 1.0 * rd:
                    sl = ep * 0.9975  # Breakeven -0.25% tutup fee & slippage
                    be_moved = True
                if t3[i] == 1:
                    trades.append((je, i + 1, -1, ep, o[i + 1], rd, "flip12"))
                    pos = 0

        # 2. Cari sinyal baru saat flat (evaluasi close bar i -> entry open bar i+1)
        if pos == 0:
            if np.isnan(adx[i]) or adx[i] <= 20.0:
                continue

            # Pullback re-alignment: 12-3 sudah searah tren, 10-1 baru flip kembali searah
            long_sig = (
                filt_dir[i] == 1
                and t3[i] == 1
                and t1[i] == 1
                and t1[i - 1] == -1
            )
            short_sig = (
                filt_dir[i] == -1
                and filt_short_ok[i]
                and (not np.isnan(ema200[i]))
                and c[i] < ema200[i]
                and t3[i] == -1
                and t1[i] == -1
                and t1[i - 1] == 1
            )

            if long_sig:
                cand_ep = o[i + 1]
                cand_sl = lo3[i]
                cand_rd = cand_ep - cand_sl
                sl_pct = (cand_rd / cand_ep) * 100.0 if cand_ep > 0 else 0.0
                if sl_min_pct <= sl_pct <= sl_max_pct:
                    pos = 1
                    je = i + 1
                    ep = cand_ep
                    sl = cand_sl
                    rd = cand_rd
                    tp = ep + tp_rr * rd
                    be_moved = False

            elif short_sig:
                cand_ep = o[i + 1]
                cand_sl = up3[i]
                cand_rd = cand_sl - cand_ep
                sl_pct = (cand_rd / cand_ep) * 100.0 if cand_ep > 0 else 0.0
                if sl_min_pct <= sl_pct <= sl_max_pct:
                    pos = -1
                    je = i + 1
                    ep = cand_ep
                    sl = cand_sl
                    rd = cand_rd
                    tp = ep - tp_rr * rd
                    be_moved = False

    if pos != 0:
        trades.append((je, n - 1, pos, ep, c[-1], rd, "end"))

    return trades


def sim_cross_account(trades, init_eq=6.0, init_margin=3.0, lev=10.0):
    """
    Simulasi Cross Margin:
    - Seluruh equity menjadi jaminan (collateral).
    - Ukuran posisi (notional) = min(init_margin, equity * 0.5) * lev.
    - Likuidasi terjadi jika equity + unrealized_worst <= notional * 0.005 (MMR 0.5%).
    """
    eq = init_eq
    peak = init_eq
    max_dd = 0.0
    mc_count = 0

    for tr in trades:
        if eq < 1.0:
            mc_count += 1
            break
        margin_used = min(init_margin, eq * 0.5)
        notional = margin_used * lev

        _, _, pos, ep, px, rd, _ = tr
        raw_ret = ((px - ep) / ep) if pos == 1 else ((ep - px) / ep)

        # Cek skenario terburuk: posisi menyentuh SL sebelum exit
        worst_ret = -(rd / ep)
        worst_pnl = notional * (worst_ret - FEE_RT * 0.5)
        if eq + worst_pnl <= notional * 0.005:
            mc_count += 1
            eq = 0.0
            max_dd = 100.0
            break

        pnl = notional * (raw_ret - FEE_RT)
        eq += pnl
        if eq <= notional * 0.005:
            mc_count += 1
            eq = 0.0
            max_dd = 100.0
            break

        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100.0 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    return eq, max_dd, mc_count


def summarize(trades, e, tf, variant):
    if not trades:
        return None
    kotor_r = []
    bersih_r = []
    sl_pcts = []
    wins = 0
    exits = {}

    for je, jx, pos, ep, px, rd, reason in trades:
        exits[reason] = exits.get(reason, 0) + 1
        pnl_price = (px - ep) if pos == 1 else (ep - px)
        kr = pnl_price / rd
        fee_r = (ep * FEE_RT) / rd
        br = kr - fee_r
        kotor_r.append(kr)
        bersih_r.append(br)
        sl_pcts.append((rd / ep) * 100.0)
        if br > 0:
            wins += 1

    n_tr = len(trades)
    hit = wins / n_tr * 100.0
    per_bln = n_tr / e["months"]
    kr_avg = float(np.mean(kotor_r))
    br_avg = float(np.mean(bersih_r))
    br_tot = float(np.sum(bersih_r))
    sl_avg = float(np.mean(sl_pcts))

    eq5, dd5, mc5 = sim_cross_account(trades, 6.0, 3.0, 5.0)
    eq10, dd10, mc10 = sim_cross_account(trades, 6.0, 3.0, 10.0)

    exit_str = " ".join(f"{k}:{v}" for k, v in sorted(exits.items(), key=lambda x: -x[1]))

    return {
        "TF": tf,
        "varian": variant,
        "trade": n_tr,
        "bln": round(per_bln, 1),
        "hit%": round(hit, 1),
        "kotorR": round(kr_avg, 3),
        "bersihR": round(br_avg, 3),
        "totR": round(br_tot, 1),
        "SL%": round(sl_avg, 2),
        "End5x": f"${eq5:.2f}" if mc5 == 0 else "MC!",
        "DD5x%": round(dd5, 1),
        "End10x": f"${eq10:.2f}" if mc10 == 0 else "MC!",
        "DD10x%": round(dd10, 1),
        "keluar": exit_str,
        "_trades": trades,
    }


def main():
    ex = os.environ.get("EXCHANGE", "bitget")
    tfs = ["5m", "15m", "1h", "4h"]
    raw = {}
    prep = {}

    for tf in tfs:
        df = e2.load(ex, tf)
        raw[tf] = df
        prep[tf] = prepare_tf(df)

    step_ms = {
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
    }

    # Bangun filter MTF
    f_5m_15m, s_5m_15m = align_mtf(prep["5m"]["ts"], step_ms["5m"], prep["15m"], step_ms["15m"])
    f_5m_1h, s_5m_1h = align_mtf(prep["5m"]["ts"], step_ms["5m"], prep["1h"], step_ms["1h"])

    f_5m_both = np.where((f_5m_15m == f_5m_1h) & (f_5m_15m != 0), f_5m_15m, 0)
    s_5m_both = s_5m_15m & s_5m_1h

    f_15m_1h, s_15m_1h = align_mtf(prep["15m"]["ts"], step_ms["15m"], prep["1h"], step_ms["1h"])
    f_15m_4h, s_15m_4h = align_mtf(prep["15m"]["ts"], step_ms["15m"], prep["4h"], step_ms["4h"])
    f_15m_both = np.where((f_15m_1h == f_15m_4h) & (f_15m_1h != 0), f_15m_1h, 0)
    s_15m_both = s_15m_1h & s_15m_4h

    f_1h_4h, s_1h_4h = align_mtf(prep["1h"]["ts"], step_ms["1h"], prep["4h"], step_ms["4h"])

    configs = [
        # (exec_tf, label, f_dir, f_short, sl_min, sl_max)
        ("5m", "F:15m", f_5m_15m, s_5m_15m, 0.50, 1.50),
        ("5m", "F:1h", f_5m_1h, s_5m_1h, 0.50, 1.50),
        ("5m", "F:15m+1h", f_5m_both, s_5m_both, 0.50, 1.50),
        ("15m", "F:1h", f_15m_1h, s_15m_1h, 0.70, 2.00),
        ("15m", "F:4h", f_15m_4h, s_15m_4h, 0.70, 2.00),
        ("15m", "F:1h+4h", f_15m_both, s_15m_both, 0.70, 2.00),
        ("1h", "F:4h", f_1h_4h, s_1h_4h, 0.80, 2.50),
    ]

    tp_list = [1.5, 2.0, 2.5]
    rows = []

    for exec_tf, lbl, fd, fs, smin, smax in configs:
        for tp_rr in tp_list:
            vname = f"{lbl} | TP:{tp_rr}R+BE"
            tr = simulate_trades(prep[exec_tf], fd, fs, smin, smax, tp_rr, use_be=True)
            res = summarize(tr, prep[exec_tf], exec_tf, vname)
            if res:
                rows.append(res)

    bt._emit("### Intraday & Scalping Pullback Triple Supertrend ($6 Cross Margin | $3 Initial Margin | 5x & 10x)\n")
    df_out = pd.DataFrame([{k: v for k, v in r.items() if k != "_trades"} for r in rows])
    bt._emit(df_out.to_markdown(index=False))
    bt._emit("\n")

    # Cetak detail TradingView report untuk top 3 varian dengan totR tertinggi (jika positif)
    top_rows = sorted([r for r in rows if r["totR"] > 0], key=lambda x: x["totR"], reverse=True)[:3]
    for r in top_rows:
        bt._emit(generate_tv_report(r["_trades"], raw[r["TF"]], r["TF"], r["varian"]))


if __name__ == "__main__":
    main()
