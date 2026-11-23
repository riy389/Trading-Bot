#!/usr/bin/env python3
"""
UJI UT BOT SWING 4h (kandidat terbaik checkpoint 31 Okt) + FUNDING NYATA + GATE. Aturan beku, tidak di-tune:
UT Bot Alerts (Pine v4) ATR 10, A=6 (tetangga A=5 dan 7 hanya cek plateau), candle 4h dari resample cache 1h Binance.
Entry open candle berikutnya hanya searah EMA200 4h (close sinyal vs EMA200), dilewati kalau open sudah melewati garis UT.
R = A x ATR candle sinyal. SL awal = entry -/+ R. Stop efektif = yang lebih dekat antara SL awal dan garis UT candle sebelumnya
(dicek intrabar, gap fill di open). Sinyal balik menutup di open berikutnya. Tanpa TP, tanpa batas waktu.
Biaya = 0,22% per trade (dalam R = 0,22% / jarak SL). Funding dari riwayat Binance (cache DataCoins): long bayar kalau rate > 0.
Gate: 13 koin selain BTC = satu portofolio, N_TRIED = 25, trade diurutkan menurut waktu entry; funding dihitung sebagai bagian P&L.
LULUS (dikunci sebelum lihat hasil): gate A=6 LOLOS, PF net >= 1,2 dan net/trade > 0 di IS dan OOS, >= 30 trade per paruh,
dan net/trade > 0 juga di A=5 dan A=7. Data setelah 2026-04-30 tidak dipakai (holdout). BTC hanya referensi.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import backtest as bt
import derivtree as dt
import gates
from datacoins import SYMS, parse_k1h
from derivrule1h import COST

N_TRIED, HOLDOUT, WARM, UT_C, MAIN, AS = 25, "2026-05-01", 200, 10, 6, (5, 6, 7)
H4 = np.timedelta64(4, "h")
AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}


def load(s, months):
    p, mk = s + "USDT", (lambda t: t.strftime("%Y-%m"))
    k = dt.load_set(f"{s}_klines1h", f"{dt.B}/monthly/klines/{p}/1h/{p}-1h-{{d}}.zip", parse_k1h, months, mk)
    f = dt.load_set(f"{s}_funding", f"{dt.B}/monthly/fundingRate/{p}/{p}-fundingRate-{{d}}.zip", dt.parse_f, months, mk)
    return k, f


def to4h(k):
    r = k[k.index < HOLDOUT].resample("4h")
    return r.agg(AGG)[r["close"].count() == 4].dropna()


def ut(c, h, l, a):
    n, pc = len(c), np.r_[np.nan, c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    tr[0] = h[0] - l[0]
    atr = np.full(n, np.nan)
    atr[UT_C - 1] = tr[:UT_C].mean()
    for i in range(UT_C, n):
        atr[i] = (atr[i - 1] * (UT_C - 1) + tr[i]) / UT_C
    nl, stop, sig, prev = a * atr, np.full(n, np.nan), np.zeros(n, int), 0.0
    for i in range(UT_C - 1, n):
        s, p = c[i], c[i - 1]
        if s > prev and p > prev:
            st = max(prev, s - nl[i])
        elif s < prev and p < prev:
            st = min(prev, s + nl[i])
        else:
            st = s - nl[i] if s > prev else s + nl[i]
        if s > st and p <= prev:
            sig[i] = 1
        elif s < st and prev <= p:
            sig[i] = -1
        stop[i], prev = st, st
    return stop, nl, sig


def sim(o, h, l, c, stop, nl, sig, okl, oks, ts):
    n, out, i = len(c), [], 0
    while i < n - 1:
        d = sig[i]
        if d == 0 or not (okl[i] if d > 0 else oks[i]) or (d > 0 and o[i + 1] <= stop[i]) or (d < 0 and o[i + 1] >= stop[i]):
            i += 1
            continue
        s, e, r = i + 1, o[i + 1], nl[i]
        sl, x, q = e - d * r, None, s
        while q < n:
            eff = max(sl, stop[q - 1]) if d > 0 else min(sl, stop[q - 1])
            if (d > 0 and o[q] <= eff) or (d < 0 and o[q] >= eff):
                x, tx = o[q], ts[q]
                break
            if (d > 0 and l[q] <= eff) or (d < 0 and h[q] >= eff):
                x, tx = eff, ts[q] + H4
                break
            if sig[q] == -d and q + 1 < n:
                x, tx = o[q + 1], ts[q + 1]
                break
            q += 1
        if x is None:
            q = n - 1
            x, tx = c[q], ts[q] + H4
        out.append((ts[s], tx, d, e, x, r))
        i = max(q, i + 1)
    return out


def trades(name, k4, f, a):
    o, h, l, c = (k4[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    ts = k4.index.to_numpy().astype("datetime64[ns]")
    ema = k4["close"].ewm(span=200, adjust=False).mean().to_numpy()
    w = np.arange(len(c)) >= WARM
    stop, nl, sig = ut(c, h, l, a)
    out = sim(o, h, l, c, stop, nl, sig, (c > ema) & w, (c < ema) & w, ts)
    if not out:
        return pd.DataFrame(columns=["coin", "t", "g", "fee", "fund"])
    t0, tx, d, e, x, r = (np.array(z) for z in zip(*out))
    ft, cs = f.index.to_numpy().astype("datetime64[ns]"), np.r_[0.0, np.cumsum(f["rate"].fillna(0).to_numpy())]
    fs = cs[np.searchsorted(ft, tx.astype("datetime64[ns]"))] - cs[np.searchsorted(ft, t0.astype("datetime64[ns]"))]
    sp = r.astype(float) / e.astype(float)
    return pd.DataFrame({"coin": name, "t": t0, "g": d * (x - e) / r, "fee": COST / sp, "fund": d * fs / sp})


def pf(x):
    m = -x[x < 0].sum()
    return round(float(x[x > 0].sum() / m), 2) if m > 0 else float("nan")


def summ(T, half):
    g, fee, fu = T["g"].to_numpy(), T["fee"].to_numpy(), T["fund"].to_numpy()
    nt, early = g - fee - fu, (T["t"] < half).to_numpy()
    return {"trade": len(T), "gross/tr": round(g.mean(), 3), "fee/tr": round(fee.mean(), 3), "fund/tr": round(fu.mean(), 3), "net/tr": round(nt.mean(), 3),
            "net_R": round(nt.sum(), 1), "n_IS": int(early.sum()), "n_OOS": int((~early).sum()), "net/tr IS": round(nt[early].mean(), 3),
            "net/tr OOS": round(nt[~early].mean(), 3), "PF_IS": pf(nt[early]), "PF_OOS": pf(nt[~early])}


def main():
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    months = sorted({(dt.START + timedelta(days=i)).isoformat()[:7] for i in range((end - dt.START).days + 1)})[:-1]
    K = {s: load(s, months) for s in SYMS}
    K = {s: (to4h(k), f) for s, (k, f) in K.items()}
    core = [s for s in SYMS if s != "BTC"]
    R = {a: {s: trades(s, K[s][0], K[s][1], a) for s in SYMS} for a in AS}
    P = {a: pd.concat([R[a][s] for s in core]).sort_values("t", kind="stable").reset_index(drop=True) for a in AS}
    half = P[MAIN]["t"].iloc[0] + (P[MAIN]["t"].iloc[-1] - P[MAIN]["t"].iloc[0]) / 2
    rows = {a: summ(P[a], half) for a in AS}
    T = P[MAIN]
    gt = gates.run_gates((T["g"] - T["fund"]).to_numpy(), T["fee"].to_numpy(), N_TRIED)
    m = rows[MAIN]
    ok = (gt["LOLOS"] and all(rows[MAIN][k] >= 1.2 for k in ("PF_IS", "PF_OOS")) and m["net/tr IS"] > 0 and m["net/tr OOS"] > 0
          and min(m["n_IS"], m["n_OOS"]) >= 30 and rows[5]["net/tr"] > 0 and rows[7]["net/tr"] > 0)
    tab = pd.DataFrame([{"A": a, **rows[a]} for a in AS])
    tab.loc[tab["A"] == MAIN, ["p_IS", "p_OOS", "fold", "gate"]] = [gt["p_IS"], gt["p_OOS"], gt["fold_pos"], "LOLOS" if gt["LOLOS"] else "GUGUR"]
    pk = []
    for s in SYMS:
        t = R[MAIN][s]
        n_ = (t["g"] - t["fee"] - t["fund"]).to_numpy()
        pk.append({"koin": s + (" (ref)" if s == "BTC" else ""), "trade": len(t), "gross/tr": round(t["g"].mean(), 3), "fund/tr": round(t["fund"].mean(), 3),
                   "net/tr": round(n_.mean(), 3), "net_R": round(n_.sum(), 1), "PF": pf(n_)})
    yr = T.assign(thn=T["t"].dt.year, net=T["g"] - T["fee"] - T["fund"]).groupby("thn").agg(
        trade=("g", "size"), gross_R=("g", "sum"), fund_R=("fund", "sum"), net_R=("net", "sum")).round(1).reset_index()
    f = "`" * 3
    bt._emit(f"\n### UT Bot swing 4h, aturan beku (ATR {UT_C}, EMA200 4h, Binance, sebelum {HOLDOUT}, N_TRIED={N_TRIED}, alpha_adj={gt['alpha_adj']:.5f}). "
             f"Biaya 0,22% + funding nyata. Positif per koin: {sum(r['net/tr'] > 0 for r in pk if 'ref' not in r['koin'])}/{len(core)}. "
             f"VONIS kriteria terkunci: {'LULUS' if ok else 'GUGUR'}\n" + f + "\n" + tab.to_string(index=False) + "\n" + f)
    bt._emit(f"\n### Per koin (A={MAIN}, BTC = referensi)\n" + f + "\n" + pd.DataFrame(pk).to_string(index=False) + "\n" + f)
    bt._emit(f"\n### Portofolio A={MAIN} per tahun (2020-2021 tidak dipakai saat memilih A)\n" + f + "\n" + yr.to_string(index=False) + "\n" + f)


if __name__ == "__main__":
    main()
