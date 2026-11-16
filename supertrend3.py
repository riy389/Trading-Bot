#!/usr/bin/env python3
"""
SUPERTREND3: 3 Supertrend (ATR period-multiplier 10-1, 11-2, 12-3), BTC. Spesifikasi: Issue #1, Checkpoint 16 Nov 2026.

- Arah: entry saat ketiga Supertrend BARU jadi searah (hijau semua = long, merah semua = short) di close candle i,
  masuk di OPEN candle i+1.
- SL awal: long = di bawah swing low terakhir sebelum entry, short = di atas swing high terakhir (pivot SWING_N kiri/kanan,
  baru dipakai setelah terkonfirmasi, tanpa lookahead). Tanpa buffer. 1R = jarak entry ke SL. Entry dilewati kalau swing
  tidak valid atau jarak SL < MIN_SL_PCT.
- Varian exit: flip = SL swing + keluar saat salah satu Supertrend balik arah (di open candle berikutnya).
  lock1R = SL swing; setelah +1R (high/low 5m sebelumnya) SL dikunci di +1R, lalu trailing garis Supertrend 10-1
  (candle TF yang sudah close) kalau lebih menguntungkan. Sinyal berlawanan penuh menutup posisi (reverse).
- Filter volatilitas (varian _vol): entry hanya kalau ATR(14)% >= persentil 70 dari 500 candle sebelumnya DAN lebar Bollinger(20,2)
  >= rata-rata 100 candle sebelumnya. Dikunci sebelum lihat hasil.
- Semua exit dicek di candle 5m. SL dan level kunci di candle 5m yang sama -> SL duluan (dihitung di tie5m).
- Pembanding: entry acak (jumlah sama dengan ST / ST_vol; random_vol hanya di bar yang lolos filter), aturan SL dan kunci sama.
- Biaya dipotong mesin saat trade ditutup (bt.apply_costs), net dan gross.
"""

import numpy as np
import pandas as pd

import backtest as bt
from utswing import swing_levels

SYMBOL = "BTC/USDT:USDT"
ENTRY_TFS = ["1h", "15m"]
ST_PARAMS = [(10, 1.0), (11, 2.0), (12, 3.0)]   # (periode ATR, multiplier); yang pertama dipakai untuk trailing
SWING_N = 5                                      # dikunci sebelum lihat hasil
LOCK_R = 1.0
ATR_N, ATR_LOOK, ATR_PCTL = 14, 500, 0.70
BB_N, BB_LOOK = 20, 100
SWEEP_BARS = 288                                 # sapuan = SL kena lalu harga balik ke entry dalam 24 jam (candle 5m)
OUTPUT_TRADES = "trades_supertrend3.csv"


def load(ex, tf):
    d = bt.fetch_ohlcv_cached(ex, SYMBOL, tf, bt.DAYS_BACK_TF[tf])
    if d is None or len(d) < 500 or any(g[2] > bt.MAX_GAP_CANDLES for g in bt.find_ohlcv_gaps(d, tf)):
        raise SystemExit(f"[!] Data {tf} tidak layak.")
    return d.iloc[:-1].reset_index(drop=True)   # candle terakhir mungkin belum close


def st_state(df):
    """state per candle (di close): 1 = ketiga hijau, -1 = ketiga merah, 0 = lainnya. line1 = garis Supertrend 10-1."""
    res = [bt.supertrend(df, m, p) for p, m in ST_PARAMS]
    up = np.all([r[1] == -1 for r in res], axis=0)
    dn = np.all([r[1] == 1 for r in res], axis=0)
    return np.where(up, 1, np.where(dn, -1, 0)), res[0][0]


def vol_filter(df):
    h, l, c = (df[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    atrp = pd.Series(bt._rma_atr(h, l, c, ATR_N) / c)
    ok1 = atrp >= atrp.shift(1).rolling(ATR_LOOK, min_periods=ATR_LOOK).quantile(ATR_PCTL)
    cs = df["close"].astype(float)
    bbw = 4 * cs.rolling(BB_N).std(ddof=0) / cs.rolling(BB_N).mean()
    ok2 = bbw >= bbw.shift(1).rolling(BB_LOOK, min_periods=BB_LOOK).mean()
    return (ok1 & ok2).to_numpy()


def prep(df, tf, t5, base):
    """Data per TF entry yang dipetakan ke candle 5m (semua dari candle TF yang sudah close, tanpa lookahead)."""
    t = df["ts"].dt.tz_convert(None).to_numpy("datetime64[ns]")
    step = pd.Timedelta(tf).to_timedelta64()
    state, line1 = st_state(df)
    sh, sl = swing_levels(df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float), SWING_N)
    k5 = np.searchsorted(t, t5, side="right") - 1
    prev = np.clip(k5 - 1, 0, None)
    n5 = len(t5)
    js = np.searchsorted(t5, t + step)
    ok = (js < n5) & (t5[np.minimum(js, n5 - 1)] == t + step)
    return {**base, "state": state, "flt": vol_filter(df), "sh": sh, "sl": sl, "js": js, "ok": ok,
            "newbar": np.r_[True, k5[1:] != k5[:-1]],
            "state5": np.where(k5 >= 1, state[prev], 0), "line5": np.where(k5 >= 1, line1[prev], np.nan)}


def st_signals(S, vol):
    st = S["state"]
    idx = np.flatnonzero((st != 0) & (st != np.r_[0, st[:-1]]) & S["ok"] & (S["flt"] if vol else True))
    return [(int(S["js"][i]), int(st[i]), int(i)) for i in idx]


def random_signals(S, n, rng, vol):
    cand = np.flatnonzero(S["ok"] & (S["flt"] if vol else True))
    idx = np.sort(rng.choice(cand, size=min(n, len(cand)), replace=False))
    return [(int(S["js"][i]), 1 if rng.random() < 0.5 else -1, int(i)) for i in idx]


def simulate(S, sigs, mode, use_state):
    o, h, l, c = S["o"], S["h"], S["l"], S["c"]
    newbar, state5, line5 = S["newbar"], S["state5"], S["line5"]
    n5 = len(o)
    out, free, ties = [], 0, 0
    for js, sign, i in sigs:
        if js < free:
            continue
        e = o[js]
        sw = S["sl"][i] if sign == 1 else S["sh"][i]
        nl = sign * (e - sw)
        if not (sw == sw and nl > 0 and nl / e >= bt.MIN_SL_PCT):
            continue
        locked, mfe, ex = False, 0.0, None
        for j in range(js, n5):
            if use_state and j > js and newbar[j]:
                s = state5[j]
                if (mode == "flip" and s != sign) or (mode == "lock" and s == -sign):
                    ex = (j, o[j], "reverse" if s == -sign else "flip", j)
                    break
            lvl, why = sw, "stop_loss"
            if locked:
                lvl, why = e + sign * nl * LOCK_R, "lock_stop"
                tl = line5[j]
                if tl == tl and sign * (tl - lvl) > 0:
                    lvl, why = tl, "trail_win"
            ext = h[j] if sign == 1 else l[j]
            if (l[j] <= lvl) if sign == 1 else (h[j] >= lvl):
                if mode == "lock" and not locked and sign * (ext - e) >= LOCK_R * nl:
                    ties += 1
                ex = (j, min(o[j], lvl) if sign == 1 else max(o[j], lvl), why, j + 1)
                break
            mfe = max(mfe, sign * (ext - e) / nl)
            if mode == "lock" and mfe >= LOCK_R:
                locked = True
        if ex is None:
            ex = (n5 - 1, c[-1], "end_data", n5)
        j, px, why, free = ex
        swept = False
        if why == "stop_loss" and j + 1 < n5:
            w = slice(j + 1, j + 1 + SWEEP_BARS)
            swept = bool(S["H"][w].max() >= e) if sign == 1 else bool(S["L"][w].min() <= e)
        gr = sign * (px - e) / e
        out.append({"side": "long" if sign == 1 else "short", "entry_ts": str(S["ts"][js]), "exit_ts": str(S["ts"][j]),
                    "entry_price": e, "exit_price": px, "sl_pct": nl / e, "size": bt.RISK_PER_TRADE / (nl / e),
                    "gross_ret": gr, "gross_R": gr / (nl / e), "exit_reason": why, "mfe_R": mfe, "swept": swept})
    return out, ties


def make_row(tf, name, trades, ties, n_sig, cname, cost, split_ts, days):
    costed = bt.apply_costs(trades, cost)
    cap = [x for x in bt.capital_sim(costed, name, SYMBOL, tf) if x["risk_pct"] == 2.0 and x["policy"] == "skip"][0]
    r = bt.fs_row(tf, 0, 0, name, costed, split_ts, cap, days)
    r.update({"cost": cname, "n_sig": n_sig})
    if costed:
        pct = lambda rs: round(float(np.mean([t["exit_reason"] in rs for t in costed]) * 100), 1)
        r["pct_lock"], r["pct_trail"], r["pct_flip"] = pct(("lock_stop",)), pct(("trail_win",)), pct(("flip", "reverse"))
        sl_tr = [t for t in costed if t["exit_reason"] == "stop_loss"]
        r["sweep%"] = round(float(np.mean([t["swept"] for t in sl_tr]) * 100), 1) if sl_tr else None
        r["tie5m"] = ties
        if cname == "net" and r.get("pf_IS") is not None and r.get("pf_OOS") is not None:
            r["edge"] = int(r["pf_IS"] >= 1.2 and r["pf_OOS"] >= 1.2 and r["netR"] > 0 and r["n_IS"] >= 30 and r["n_OOS"] >= 30)
    return r, costed


def run(d5, dfs):
    t5 = d5["ts"].dt.tz_convert(None).to_numpy("datetime64[ns]")
    base = {"o": d5["open"].to_numpy(float).tolist(), "h": d5["high"].to_numpy(float).tolist(),
            "l": d5["low"].to_numpy(float).tolist(), "c": d5["close"].to_numpy(float).tolist(),
            "H": d5["high"].to_numpy(float), "L": d5["low"].to_numpy(float), "ts": d5["ts"].tolist()}
    split_ts = d5["ts"].iloc[0] + (d5["ts"].iloc[-1] - d5["ts"].iloc[0]) / 2
    days = max((d5["ts"].iloc[-1] - d5["ts"].iloc[0]).days, 1)
    cost_net = bt.net_cost(SYMBOL)
    rows, trade_rows = [], []
    for tf, df in dfs.items():
        S = prep(df, tf, t5, base)
        s_all, s_vol = st_signals(S, False), st_signals(S, True)
        rng = np.random.default_rng(1)
        sets = [("ST", s_all, ("flip", "lock"), True), ("ST_vol", s_vol, ("flip", "lock"), True),
                ("random", random_signals(S, len(s_all), rng, False), ("lock",), False),
                ("random_vol", random_signals(S, len(s_vol), rng, True), ("lock",), False)]
        for sname, sigs, modes, use_state in sets:
            for mode in modes:
                name = f"{sname}_{'flip' if mode == 'flip' else 'lock1R'}"
                trades, ties = simulate(S, sigs, mode, use_state)
                for cname, cost in (("net", cost_net), ("gross", 0.0)):
                    r, costed = make_row(tf, name, trades, ties, len(sigs), cname, cost, split_ts, days)
                    rows.append(r)
                    if cname == "net":
                        trade_rows += [{"tf": tf, "variant": name, **t} for t in costed]
                print(f"  {tf} {name}: {len(trades)} trade dari {len(sigs)} sinyal")
    pd.DataFrame(trade_rows).to_csv(OUTPUT_TRADES, index=False)
    fence = "`" * 3
    bt._emit("\n### SUPERTREND3 (10-1, 11-2, 12-3 searah), BTC, SL di swing terakhir (pivot " + f"{SWING_N}" + " kiri/kanan, tanpa buffer), long dan short. "
             "flip = keluar saat salah satu ST balik arah. lock1R = SL dikunci di +1R lalu trailing garis ST 10-1. _vol = hanya saat ATR% >= persentil 70 "
             "dan lebar Bollinger >= rata-rata. random = entry acak (jumlah = n_sig) dengan aturan SL dan kunci sama; random_vol = acak hanya di bar lolos filter. "
             f"net = biaya {cost_net * 100:.2f}%, gross = tanpa biaya. n_sig = sinyal, n = trade terisi. sweep% = % SL kena lalu harga balik ke entry dalam 24 jam. "
             "tie5m = candle 5m yang kena SL dan +1R sekaligus (dianggap SL duluan). edge = PF IS dan OOS >= 1,2, netR > 0, n >= 30 per paruh\n"
             + fence + "\n" + pd.DataFrame(rows).drop(columns=["A", "ATR", "pct_tp"], errors="ignore").to_string(index=False) + "\n" + fence)


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    d5 = load(ex, "5m")
    run(d5, {tf: load(ex, tf) for tf in ENTRY_TFS})


if __name__ == "__main__":
    main()
