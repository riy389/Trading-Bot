#!/usr/bin/env python3
"""
DIRECTION_TEST: apakah arah 3 Supertrend (10-1, 11-2, 12-3 searah) lebih baik dari acak? Tanpa SL swing dan tanpa strategi lain.
- Entry sama seperti supertrend3.py (open candle setelah ketiga Supertrend baru searah). Dari entry itu dilihat, dalam 24 jam
  (candle 5m), mana yang tersentuh LEBIH DULU: +T poin searah posisi (menang) atau -T poin berlawanan (kalah). T = 500 dan 1000 poin.
- tie = kedua level ada di candle 5m yang sama, tidak_ada = tidak ada yang tersentuh dalam 24 jam.
- acc% = menang / (menang + kalah). ci = selang +-95% (dari akurasi 50%; sinyal 15m saling tumpang tindih, jadi selang ini terlalu optimis).
- butuh% = akurasi yang dibutuhkan supaya impas setelah biaya: (T + fee_poin) / (2T). fee_poin = biaya mesin x harga median.
- lulus = 1 kalau acc - ci > butuh% (arah terbukti cukup untuk menutup biaya).
- ACAK = pembanding: setiap jam dengan arah acak (seed 1), kelompok harga sama.
"""
import numpy as np
import pandas as pd

import backtest as bt
from measure_range import BINS, LABELS, WIN, first_touch
from supertrend3 import ENTRY_TFS, SYMBOL, load, prep, st_signals

TARGETS = [500.0, 1000.0]


def table(tf, name, o, h, l, js, sg, cost):
    keep = js <= len(o) - WIN
    js, sg = js[keep], sg[keep]
    price = o[js]
    bucket = np.asarray(pd.cut(price, BINS, labels=LABELS).astype(object))
    rows = []
    for T in TARGETS:
        code = first_touch(o, h, l, js, T)
        win, loss, tie = code == sg, code == -sg, code == 2
        for b in LABELS + ["SEMUA"]:
            m = np.ones(len(js), bool) if b == "SEMUA" else bucket == b
            n = int(m.sum())
            if n == 0:
                continue
            w, lo = int(win[m].sum()), int(loss[m].sum())
            acc = w / (w + lo) * 100 if w + lo else np.nan
            ci = 1.96 * np.sqrt(0.25 / (w + lo)) * 100 if w + lo else np.nan
            need = (T + cost * float(np.median(price[m]))) / (2 * T) * 100
            rows.append({"tf": tf, "set": name, "harga": b, "T": int(T), "n": n, "menang%": round(w / n * 100, 1), "kalah%": round(lo / n * 100, 1),
                         "tie%": round(float(tie[m].mean() * 100), 1), "tidak_ada%": round(float((code[m] == 0).mean() * 100), 1),
                         "acc%": round(acc, 1), "ci": round(ci, 1), "butuh%": round(need, 1), "lulus": int(acc - ci > need)})
    return rows


def run(d5, dfs):
    t5 = d5["ts"].dt.tz_convert(None).to_numpy("datetime64[ns]")
    o, h, l = (d5[k].to_numpy(dtype=float) for k in ("open", "high", "low"))
    base = {"o": o.tolist(), "h": h.tolist(), "l": l.tolist(), "c": d5["close"].to_numpy(float).tolist(),
            "H": h, "L": l, "ts": d5["ts"].tolist()}
    cost = bt.net_cost(SYMBOL)
    rng = np.random.default_rng(1)
    rows = []
    allstarts = np.arange(0, len(o) - WIN, 12)
    rows += table("-", "ACAK", o, h, l, allstarts, rng.choice([-1, 1], size=len(allstarts)), cost)
    for tf, df in dfs.items():
        S = prep(df, tf, t5, base)
        for name, vol in (("ST", False), ("ST_vol", True)):
            sigs = st_signals(S, vol)
            rows += table(tf, name, o, h, l, np.array([s[0] for s in sigs]), np.array([s[1] for s in sigs]), cost)
    fence = "`" * 3
    bt._emit("\n### DIRECTION_TEST: arah 3 Supertrend vs acak. Dari entry, mana yang tersentuh dulu dalam 24 jam: +T poin searah (menang) atau -T poin "
             "berlawanan (kalah). acc% = menang/(menang+kalah), ci = selang +-95% (terlalu optimis untuk sinyal yang tumpang tindih), "
             "butuh% = akurasi impas setelah biaya, lulus = acc - ci > butuh%. ACAK = tiap jam arah acak. ST_vol = hanya saat filter volatilitas menyala\n"
             + fence + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + fence)
    return rows


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    d5 = load(ex, "5m")
    run(d5, {tf: load(ex, tf) for tf in ENTRY_TFS})


if __name__ == "__main__":
    main()
