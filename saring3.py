#!/usr/bin/env python3
"""
SARING3: tiga filter Riyanto untuk gerakan BTC 1h 2.000-4.000 USD, diuji sendiri-sendiri dan digabung. Entry di close candle pemicu, arah = arah candle.
  F1 akumulasi : selisih EMA 10/20/50/100 (maks-min)/harga <= 1% selama W candle 1h sebelum pemicu (W=6, varian 12)
  F2 pemicu    : badan candle >= B USD (B=700, varian 1000) dan badan >= 70% dari range
  F3 4h        : arah candle = posisi close 4h terakhir (sudah tutup) terhadap EMA50 4h, dan tidak ada swing high (long) / swing low (short) 4h
                 (pivot 5, hanya yang sudah terkonfirmasi, 120 candle 4h terakhir) dalam jarak < 1.500 USD searah entry
Keluar: SL = TP = 2,7%, maksimal 24 jam (kalau tidak kena, tutup di close). SL dicek dulu kalau SL dan TP kena di candle yang sama.
Satu posisi sekaligus. Fee 0,12% per trade. "Semua candle" = pembanding tanpa filter.
"""
import numpy as np
import pandas as pd

import backtest as bt
import eventtree2 as e2
import klasik as K

TP, MAXH, SPREAD, PIV, LOOK, DIST, WARM = 0.027, 24, 0.01, 5, 120, 1500.0, 200
VARIANTS = [(6, 700), (12, 700), (6, 1000), (12, 1000)]


def ema_spread(c):
    s = pd.concat([c.ewm(span=n, adjust=False).mean() for n in (10, 20, 50, 100)], axis=1)
    return ((s.max(axis=1) - s.min(axis=1)) / c).to_numpy()


def filter4h(d1, d4, dirc, c):
    """Per candle 1h: True kalau arah searah 4h dan tidak menabrak swing 4h dalam DIST."""
    c4 = d4["close"]
    dir4 = np.where(c4 > c4.ewm(span=50, adjust=False).mean(), 1, -1)
    h4, l4 = d4["high"].to_numpy(float), d4["low"].to_numpy(float)
    ph, pl = [], []
    for j in range(PIV, len(d4) - PIV):
        if h4[j] == h4[j - PIV:j + PIV + 1].max():
            ph.append((j + PIV, h4[j]))
        if l4[j] == l4[j - PIV:j + PIV + 1].min():
            pl.append((j + PIV, l4[j]))
    ph, pl = np.array(ph, dtype=float).reshape(-1, 2), np.array(pl, dtype=float).reshape(-1, 2)
    left = pd.DataFrame({"t": d1["ts"] + pd.Timedelta(hours=1)})
    right = pd.DataFrame({"t": d4["ts"] + pd.Timedelta(hours=4), "k": np.arange(len(d4))})
    k = pd.merge_asof(left, right, on="t")["k"].fillna(-1).astype(int).to_numpy()
    ok = np.zeros(len(c), dtype=bool)
    for i in range(len(c)):
        if k[i] < 0 or dirc[i] == 0 or dir4[k[i]] != dirc[i]:
            continue
        piv = ph if dirc[i] > 0 else pl
        a, b = np.searchsorted(piv[:, 0], k[i] - LOOK, side="right"), np.searchsorted(piv[:, 0], k[i], side="right")
        p = piv[a:b, 1]
        p = p[p > c[i]] - c[i] if dirc[i] > 0 else c[i] - p[p < c[i]]
        ok[i] = len(p) == 0 or p.min() >= DIST
    return ok


def sim(h, l, c, cand, d):
    out, nxt, n = [], 0, len(c)
    for i in np.flatnonzero(cand):
        if i < nxt or i >= n - 1:
            continue
        e, s, end = c[i], d[i], min(n - 1, i + MAXH)
        r, j, mfe, how = s * (c[end] / e - 1), end, 0.0, "t"
        for q in range(i + 1, end + 1):
            fav = (h[q] / e - 1) if s > 0 else (1 - l[q] / e)
            adv = (1 - l[q] / e) if s > 0 else (h[q] / e - 1)
            mfe = max(mfe, fav)
            if adv >= TP:
                r, j, how = -TP, q, "s"
                break
            if fav >= TP:
                r, j, how = TP, q, "p"
                break
        out.append((i, j, r, mfe, how))
        nxt = j + 1
    return out


def run(ex):
    d1, d4 = e2.load(ex, "1h"), e2.load(ex, "4h")
    o, h, l, c = (d1[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    dirc = np.sign(c - o).astype(int)
    base = (dirc != 0) & (np.arange(len(c)) >= WARM)
    f3 = filter4h(d1, d4, dirc, c)
    spr = pd.Series(ema_spread(d1["close"]))
    body = np.abs(c - o)
    f2 = {b: (body >= b) & (body >= 0.7 * (h - l)) for b in {v[1] for v in VARIANTS}}
    f1 = {w: (spr.rolling(w).max().shift(1) <= SPREAD).to_numpy() for w in {v[0] for v in VARIANTS}}
    rules = {"Semua candle (pembanding)": base, "F1 akumulasi (6c)": base & f1[6], "F2 pemicu (700)": base & f2[700], "F3 4h": base & f3,
             "F1+F2 (6c,700)": base & f1[6] & f2[700], "F2+F3 (700)": base & f2[700] & f3, "F1+F2+F3 (6c,700)": base & f1[6] & f2[700] & f3}
    for w, b in VARIANTS:
        if (w, b) != (6, 700):
            rules[f"F1+F2 ({w}c,{b})"] = base & f1[w] & f2[b]
            rules[f"F1+F2+F3 ({w}c,{b})"] = base & f1[w] & f2[b] & f3
    half = d1["ts"].iloc[0] + (d1["ts"].iloc[-1] - d1["ts"].iloc[0]) / 2
    early, year = (d1["ts"] < half).to_numpy(), d1["ts"].dt.year.to_numpy()
    months = (d1["ts"].iloc[-1] - d1["ts"].iloc[0]).days / 30
    rows, yrs = [], []
    for nm, cand in rules.items():
        tr = sim(h, l, c, cand, dirc)
        if not tr:
            rows.append({"aturan": nm, "trade": 0})
            continue
        i, j, r, mfe, how = (np.array(z) for z in zip(*tr))
        i, j, r, mfe = i.astype(int), j.astype(int), r.astype(float), mfe.astype(float)
        net = r - K.FEE
        rows.append({"aturan": nm, "trade": len(tr), "per_bulan": round(len(tr) / months, 1), "jam_rata": round((j - i).mean(), 1),
                     "hit%": round((net > 0).mean() * 100, 1), "TP%": round((how == "p").mean() * 100, 1), "SL%": round((how == "s").mean() * 100, 1),
                     "MFE_rata%": round(mfe.mean() * 100, 2), "kotor_total%": round(r.sum() * 100, 1), "fee_total%": round(len(tr) * K.FEE * 100, 1),
                     "bersih_total%": round(net.sum() * 100, 1), "kotor%/trade": round(r.mean() * 100, 3), "paruh1%": round(net[early[i]].sum() * 100, 1),
                     "paruh2%": round(net[~early[i]].sum() * 100, 1), "terburuk%": round(net.min() * 100, 1), "terbaik%": round(net.max() * 100, 1)})
        if nm.startswith(("F2+F3", "F1+F2+F3 (6c,700)")):
            for y in np.unique(year[i]):
                m = year[i] == y
                yrs.append({"aturan": nm, "tahun": int(y), "trade": int(m.sum()), "hit%": round((net[m] > 0).mean() * 100, 1), "bersih_total%": round(net[m].sum() * 100, 1)})
    fence = "`" * 3
    bt._emit("\n### Saring3, BTC 1h, entry close candle pemicu, SL=TP 2,7%, maks 24 jam, satu posisi. kotor = sebelum fee, fee 0,12% per trade. "
             "total% = jumlah hasil semua trade (tanpa leverage, tanpa bunga-berbunga). bersih_total = kotor_total - fee_total\n"
             + fence + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + fence)
    bt._emit("\n### Per tahun (bersih setelah fee)\n" + fence + "\n" + pd.DataFrame(yrs).to_string(index=False) + "\n" + fence)


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
