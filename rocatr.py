#!/usr/bin/env python3
"""
ROC + ATR long/short, BTC 1h (dari artikel Daniro/Horizon, disesuaikan).
Long : ROC(10)>0, close>EMA50, (close - close sebelumnya) > 0.8*ATR(14). Short: kebalikannya.
Entry di open candle berikutnya. SL awal 2*ATR (ATR candle sinyal) = 1R, lalu trailing 2*ATR searah profit (tidak mundur).
Exit juga kalau ROC berbalik (di open candle berikutnya). Sinyal berlawanan = tutup lalu buka arah sebaliknya.
Satu posisi. Fee 0,12% + slippage 0,10% dihitung engine saat close, dalam R. Tidak ada TP.
"""
import numpy as np
import pandas as pd

import backtest as bt
import eventtree2 as e2

FEE, SLIP, K, WARM, EMA_N, ATR_N = 0.0012, 0.0010, 2.0, 200, 50, 14
VARIANTS = [
    ("BASE ROC10 m0.8 EMA50", {}),
    ("ROC 8", {"roc_n": 8}), ("ROC 12", {"roc_n": 12}),
    ("m 0.6", {"m": 0.6}), ("m 1.0", {"m": 1.0}),
    ("tanpa filter ATR", {"m": 0.0}), ("tanpa filter EMA", {"ema": False}),
    ("long saja", {"sides": (1,)}), ("short saja", {"sides": (-1,)}),
]


def signals(c, atr, roc_n=10, m=0.8, ema=True, sides=(1, -1)):
    roc = (c / c.shift(roc_n) - 1).to_numpy()
    cc = c.to_numpy()
    mv = (c - c.shift(1)).to_numpy()
    lg, sh = (roc > 0) & (mv > m * atr), (roc < 0) & (mv < -m * atr)
    if ema:
        e = c.ewm(span=EMA_N, adjust=False).mean().to_numpy()
        lg, sh = lg & (cc > e), sh & (cc < e)
    sig = np.zeros(len(cc), dtype=int)
    if 1 in sides:
        sig[lg] = 1
    if -1 in sides:
        sig[sh] = -1
    sig[:WARM] = 0
    return sig, roc


def sim(o, h, l, c, atr, sig, roc):
    n, out, i = len(c), [], 0
    while i < n - 1:
        d = sig[i]
        if d == 0:
            i += 1
            continue
        s = i + 1
        e = o[s]
        rf = K * atr[i] / e
        stop = e - d * K * atr[i]
        x, qe = c[n - 1], n - 1
        for q in range(s, n):
            if d > 0 and l[q] <= stop:
                x, qe = min(stop, o[q]), q
                break
            if d < 0 and h[q] >= stop:
                x, qe = max(stop, o[q]), q
                break
            stop = max(stop, h[q] - K * atr[q]) if d > 0 else min(stop, l[q] + K * atr[q])
            if q < n - 1 and roc[q] * d < 0:
                x, qe = o[q + 1], q
                break
        out.append((s, qe, d, e, x, rf))
        i = max(qe, i + 1)
    return out


def stats(tr, ts, half):
    s, q, d, e, x, rf = (np.array(z, dtype=float) for z in zip(*tr))
    gross = d * (x / e - 1) / rf
    net = gross - (FEE + SLIP) / rf
    cum = np.r_[0.0, np.cumsum(net)]
    early = (ts[s.astype(int)] < half).to_numpy()
    return s, net, {"trade": len(net), "hit%": round((net > 0).mean() * 100, 1), "jam_rata": round((q - s + 1).mean(), 1),
                    "gross_R": round(gross.sum(), 1), "net_R": round(net.sum(), 1), "net_R/trade": round(net.mean(), 3),
                    "maxDD_R": round((np.maximum.accumulate(cum) - cum).max(), 1),
                    "paruh1_R": round(net[early].sum(), 1), "paruh2_R": round(net[~early].sum(), 1)}


def run(ex):
    d = e2.load(ex, "1h")
    o, h, l, c = (d[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    atr = pd.Series(tr).ewm(alpha=1 / ATR_N, adjust=False).mean().to_numpy()
    ts = d["ts"]
    half = ts.iloc[0] + (ts.iloc[-1] - ts.iloc[0]) / 2
    rows, yrs = [], []
    for nm, kw in VARIANTS:
        sig, roc = signals(d["close"], atr, **kw)
        trades = sim(o, h, l, c, atr, sig, roc)
        if not trades:
            rows.append({"varian": nm, "trade": 0})
            continue
        s, net, st = stats(trades, ts, half)
        rows.append({"varian": nm, **st})
        if not kw:
            yr = ts.dt.year.to_numpy()[s.astype(int)]
            for y in np.unique(yr):
                yrs.append({"tahun": int(y), "trade": int((yr == y).sum()), "net_R": round(net[yr == y].sum(), 1)})
    f = "`" * 3
    bt._emit("\n### ROC+ATR long/short, BTC 1h. 1R = jarak SL awal (2 ATR). net_R = setelah fee 0,12% + slippage 0,10%. "
             "maxDD_R dalam R. paruh1/2 = pembagian waktu data 5 tahun\n" + f + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + f)
    bt._emit("\n### BASE per tahun\n" + f + "\n" + pd.DataFrame(yrs).to_string(index=False) + "\n" + f)


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
