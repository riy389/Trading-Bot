#!/usr/bin/env python3
"""
REPLIKASI video "Cara Tahu Kapan Beli dan Jual Bitcoin": EMA185 4h (close), LONG ONLY, tanpa SL/TP.
BUY : saat flat, 4 candle berturut-turut close > EMA185 -> entry di open candle berikutnya.
EXIT: 1 candle close < EMA185 -> keluar di open candle berikutnya.
Hasil dalam % harga per trade (bukan R), compounding all-in per koin, dibanding buy and hold periode sama.
Biaya 0,22% per trade (standar proyek); kolom fut juga memotong funding nyata. Sebelum holdout. Tes replikasi, bukan kandidat (tanpa gate).
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import backtest as bt
import derivtree as dt
import utswing4h as u
from datacoins import SYMS
from derivrule1h import COST

EMA_N, CONF, WARM = 185, 4, 200


def mdd(path):
    return float((path / np.maximum.accumulate(path) - 1).min())


def run(k4, f):
    o = k4["open"].to_numpy(float)
    c = k4["close"].to_numpy(float)
    ts = k4.index.to_numpy().astype("datetime64[ns]")
    ema = k4["close"].ewm(span=EMA_N, adjust=False).mean().to_numpy()
    n = len(c)
    above = c > ema
    ft = f.index.to_numpy().astype("datetime64[ns]")
    cs = np.r_[0.0, np.cumsum(f["rate"].fillna(0).to_numpy())]
    r = np.zeros(n)
    tr = []
    k = WARM
    while k < n - 2:
        if above[k - CONF + 1:k + 1].all():
            s = k + 1
            idx = np.flatnonzero(c[s:n - 1] < ema[s:n - 1])
            x = s + int(idx[0]) + 1 if len(idx) else n - 1
            fund = cs[np.searchsorted(ft, ts[x])] - cs[np.searchsorted(ft, ts[s])]
            tr.append((ts[s], o[x] / o[s] - 1, fund))
            r[s:x] = o[s + 1:x + 1] / o[s:x] - 1
            r[x - 1] = (1 + r[x - 1]) * (1 - COST) - 1
            k = x
        else:
            k += 1
    t = pd.DataFrame(tr, columns=["t", "g", "fund"])
    eq = np.r_[1.0, np.cumprod(1 + r[WARM:n - 1])]
    bh = np.r_[1.0, o[WARM + 1:n] / o[WARM]]
    ex = {"strat%": round(100 * (eq[-1] - 1), 1), "B&H%": round(100 * (bh[-1] - 1), 1),
          "DD%": round(100 * mdd(eq), 1), "DD B&H%": round(100 * mdd(bh), 1), "start": str(ts[WARM])[:10], "end": str(ts[-1])[:10]}
    return t, ex


def stats(t):
    if len(t) == 0:
        return {"n": 0}
    ns, nf = t["g"] - COST, t["g"] - COST - t["fund"]
    w, l = ns[ns > 0], ns[ns <= 0]
    return {"n": len(t), "win%": round(100 * float((ns > 0).mean()), 1),
            "avgW%": round(100 * float(w.mean()), 2) if len(w) else 0.0,
            "avgL%": round(100 * float(l.mean()), 2) if len(l) else 0.0,
            "PF": round(float(w.sum() / -l.sum()), 2) if len(l) and l.sum() < 0 else float("inf"),
            "net/tr% spot": round(100 * float(ns.mean()), 2), "net/tr% fut": round(100 * float(nf.mean()), 2)}


def halves(name, t):
    if len(t) == 0:
        return []
    half = t["t"].iloc[0] + (t["t"].iloc[-1] - t["t"].iloc[0]) / 2
    return [{"set": name, **stats(t)}, {"set": name + " IS", **stats(t[t["t"] < half])},
            {"set": name + " OOS", **stats(t[t["t"] >= half])}]


def main():
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    months = sorted({(dt.START + timedelta(days=i)).isoformat()[:7] for i in range((end - dt.START).days + 1)})[:-1]
    rows, T, per = [], {}, {}
    for s in SYMS:
        k, f = u.load(s, months)
        k4 = u.to4h(k)
        k4 = k4[k4.index < u.HOLDOUT]
        t, ex = run(k4, f)
        T[s] = t
        per[s] = ex
        rows.append({"koin": s, **stats(t), **{k_: v for k_, v in ex.items() if k_ not in ("start", "end")}})
    core = pd.concat([T[s] for s in SYMS if s != "BTC"]).sort_values("t", kind="stable").reset_index(drop=True)
    pool = pd.DataFrame(halves("BTC", T["BTC"]) + halves("13 koin", core))
    f3 = "`" * 3
    bt._emit(f"\n### EMA185 4h LONG ONLY tanpa SL/TP (replikasi video), Binance, sebelum {u.HOLDOUT}. BTC: {per['BTC']['start']} s/d {per['BTC']['end']}. "
             "Biaya 0,22%; spot = tanpa funding, fut = dengan funding. strat% = compounding all-in, DD = max drawdown\n"
             + f3 + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + f3)
    bt._emit("\n### BTC dan 13 koin gabungan (IS/OOS = paruh waktu pertama/kedua)\n" + f3 + "\n" + pool.to_string(index=False) + "\n" + f3)


if __name__ == "__main__":
    main()
