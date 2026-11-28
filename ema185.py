#!/usr/bin/env python3
"""
UJI EMA185 4h (Binance, candle 4h dari cache 1h, sebelum holdout 2026-05-01). Aturan dari Riyanto, tidak di-tune:
BUY : candle hijau open < EMA185 dan close > EMA185 = candle 1; close candle 2,3,4 semuanya > EMA185; entry open candle 5.
SELL: candle merah open > EMA185 dan close < EMA185 = candle 1; entry langsung open candle 2 (tanpa konfirmasi).
SL = swing pivot 5 terakhir sebelum entry (long: swing low di bawah entry, short: swing high di atas entry), skip kalau jarak > 8%.
Exit (semua varian): SL, TP (1.5R/2R/3R/tanpa TP), atau close 4h menembus EMA185 melawan posisi (exit di open berikutnya), mana yang kena duluan.
Candle cross yang menutup posisi tetap dibaca sebagai sinyal baru (cross merah -> SELL di open berikutnya; cross hijau -> hitung 4 candle).
Varian "noWE": tanpa entry baru kalau open candle entry jatuh Sabtu/Minggu UTC (posisi terbuka tidak ditutup, sinyal dilewati).
SL/TP dicek intrabar (gap fill di open); kalau SL dan TP kena di candle 4h yang sama, urutan dibaca dari candle 1h (sama di 1h = SL dulu).
Satu posisi per koin. R = jarak entry ke SL.
Biaya 0,22% per trade (dalam R = 0,22% / jarak SL) + funding nyata. 13 koin selain BTC = satu portofolio, BTC referensi.
Gate: N_TRIED = 38 (29 + 8 varian + EMAexit long), per varian.
LULUS (dikunci sebelum lihat hasil): gate LOLOS, PF net IS dan OOS >= 1,2, net/trade IS dan OOS > 0, >= 30 trade per paruh.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import backtest as bt
import derivtree as dt
import gates
import utswing4h as u
from datacoins import SYMS
from derivrule1h import COST

N_TRIED, WARM, EMA_N, WAIT, PIV, MAXSL = 38, 200, 185, 3, 5, 0.08
H4 = u.H4
VAR = {}
for _wk in (False, True):
    for _nm, _m in (("TP1.5R+EMA", 1.5), ("TP2R+EMA", 2.0), ("TP3R+EMA", 3.0), ("EMAexit", None)):
        VAR[_nm + (" noWE" if _wk else "")] = (_m, _wk)
VAR["EMAexit long"] = (None, False)  # long-only: sinyal SELL diabaikan


def swing(flag, val, s, e, d):
    lo = max(0, s - 100 - PIV - 1)
    for j in (np.flatnonzero(flag[lo:s - PIV]) + lo)[::-1]:
        if (val[j] < e) if d > 0 else (val[j] > e):
            return val[j]
    return None


def first(k1, t, d, sl, tp):
    ht, hh, hl = k1
    p = np.searchsorted(ht, t)
    for a, b in zip(hh[p:p + 4], hl[p:p + 4]):
        if (b <= sl) if d > 0 else (a >= sl):
            return "sl"
        if (a >= tp) if d > 0 else (b <= tp):
            return "tp"
    return "sl"


def trades(name, k4, k1, f, m, wk, lo=False):
    o, h, l, c = (k4[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    ts = k4.index.to_numpy().astype("datetime64[ns]")
    dow = (ts.astype("datetime64[D]").astype("int64") + 3) % 7
    ema = k4["close"].ewm(span=EMA_N, adjust=False).mean().to_numpy()
    n = len(c)
    up = (c > o) & (o < ema) & (c > ema)
    dn = (c < o) & (o > ema) & (c < ema)
    pl = pd.Series(l).rolling(2 * PIV + 1, center=True).min().to_numpy() == l
    ph = pd.Series(h).rolling(2 * PIV + 1, center=True).max().to_numpy() == h
    out, i = [], WARM
    while i < n - 1:
        d, s = 0, 0
        if up[i] and i + WAIT + 1 < n and (c[i + 1:i + WAIT + 1] > ema[i + 1:i + WAIT + 1]).all():
            d, s = 1, i + WAIT + 1
        elif dn[i] and not lo:
            d, s = -1, i + 1
        if d == 0 or (wk and dow[s] >= 5):
            i += 1
            continue
        e = o[s]
        sl = swing(pl if d > 0 else ph, l if d > 0 else h, s, e, d)
        if sl is None or abs(e - sl) / e > MAXSL:
            i += 1
            continue
        r = abs(e - sl)
        tp = None if m is None else e + d * m * r
        x, q = None, s
        while q < n:
            if (d > 0 and o[q] <= sl) or (d < 0 and o[q] >= sl):
                x, tx = o[q], ts[q]
                break
            if tp is not None and ((d > 0 and o[q] >= tp) or (d < 0 and o[q] <= tp)):
                x, tx = o[q], ts[q]
                break
            hs = (l[q] <= sl) if d > 0 else (h[q] >= sl)
            hp = tp is not None and ((h[q] >= tp) if d > 0 else (l[q] <= tp))
            if hs and (not hp or first(k1, ts[q], d, sl, tp) == "sl"):
                x, tx = sl, ts[q] + H4
                break
            if hp:
                x, tx = tp, ts[q] + H4
                break
            if q + 1 < n and ((d > 0 and c[q] < ema[q]) or (d < 0 and c[q] > ema[q])):
                x, tx = o[q + 1], ts[q + 1]
                break
            q += 1
        if x is None:
            q = n - 1
            x, tx = c[q], ts[q] + H4
        out.append((ts[s], tx, d, e, x, r))
        i = max(q, i + 1)
    if not out:
        return pd.DataFrame(columns=["coin", "t", "side", "g", "fee", "fund"])
    t0, tx, d, e, x, r = (np.array(z) for z in zip(*out))
    ft = f.index.to_numpy().astype("datetime64[ns]")
    cs = np.r_[0.0, np.cumsum(f["rate"].fillna(0).to_numpy())]
    fs = cs[np.searchsorted(ft, tx.astype("datetime64[ns]"))] - cs[np.searchsorted(ft, t0.astype("datetime64[ns]"))]
    sp = r.astype(float) / e.astype(float)
    return pd.DataFrame({"coin": name, "t": t0, "side": d, "g": d * (x - e) / r, "fee": COST / sp, "fund": d * fs / sp})


def main():
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    months = sorted({(dt.START + timedelta(days=i)).isoformat()[:7] for i in range((end - dt.START).days + 1)})[:-1]
    K = {}
    for s in SYMS:
        k, f = u.load(s, months)
        k1 = k[k.index < u.HOLDOUT]
        K[s] = (u.to4h(k), (k1.index.to_numpy().astype("datetime64[ns]"), k1["high"].to_numpy(float), k1["low"].to_numpy(float)), f)
    core = [s for s in SYMS if s != "BTC"]
    R = {v: {s: trades(s, K[s][0], K[s][1], K[s][2], m, wk, v.endswith(" long")) for s in SYMS} for v, (m, wk) in VAR.items()}
    P = {v: pd.concat([R[v][s] for s in core]).sort_values("t", kind="stable").reset_index(drop=True) for v in VAR}
    rows = []
    for v, T in P.items():
        half = T["t"].iloc[0] + (T["t"].iloc[-1] - T["t"].iloc[0]) / 2
        sm = u.summ(T, half)
        gt = gates.run_gates((T["g"] - T["fund"]).to_numpy(), T["fee"].to_numpy(), N_TRIED)
        ok = (gt["LOLOS"] and sm["PF_IS"] >= 1.2 and sm["PF_OOS"] >= 1.2 and sm["net/tr IS"] > 0 and sm["net/tr OOS"] > 0
              and min(sm["n_IS"], sm["n_OOS"]) >= 30)
        rows.append({"exit": v, **sm, "p_IS": gt["p_IS"], "p_OOS": gt["p_OOS"], "fold": gt["fold_pos"],
                     "gate": "LOLOS" if gt["LOLOS"] else "GUGUR", "VONIS": "LULUS" if ok else "GUGUR"})
    ls = []
    for v, T in P.items():
        nt = T["g"] - T["fee"] - T["fund"]
        for sd, nm in ((1, "long"), (-1, "short")):
            mk = T["side"] == sd
            ls.append({"exit": v, "sisi": nm, "trade": int(mk.sum()), "gross/tr": round(T.loc[mk, "g"].mean(), 3),
                       "net/tr": round(nt[mk].mean(), 3), "net_R": round(nt[mk].sum(), 1)})
    base = [v for v, (m, wk) in VAR.items() if not wk]
    pk = []
    for s in SYMS:
        row = {"koin": s + (" (ref)" if s == "BTC" else "")}
        for v in base:
            t = R[v][s]
            row[v] = round(float((t["g"] - t["fee"] - t["fund"]).mean()), 3) if len(t) else float("nan")
        pk.append(row)
    f3 = "`" * 3
    bt._emit(f"\n### EMA185 4h (Binance, sebelum {u.HOLDOUT}, N_TRIED={N_TRIED}). Biaya 0,22% + funding nyata. 13 koin satu portofolio\n"
             + f3 + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + f3)
    bt._emit("\n### Long vs short per varian\n" + f3 + "\n" + pd.DataFrame(ls).to_string(index=False) + "\n" + f3)
    bt._emit("\n### Net/trade per koin, varian tanpa weekend-block (BTC = referensi)\n" + f3 + "\n" + pd.DataFrame(pk).to_string(index=False) + "\n" + f3)


if __name__ == "__main__":
    main()
