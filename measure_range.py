#!/usr/bin/env python3
"""
MEASURE_RANGE: ukur gerak BTC dalam POIN per kelompok harga, tanpa strategi apa pun (bukan asumsi, dari data candle 5m).
- range_harian = high tertinggi - low terendah per hari UTC (poin), dikelompokkan menurut harga open hari itu.
- naik_dulu / turun_dulu / tie / tidak_ada: dari open candle 5m tiap jam, mana yang tersentuh lebih dulu dalam 24 jam:
  +TARGET poin atau -TARGET poin. tie = kedua level ada di candle 5m yang sama. tidak_ada = tidak ada yang tersentuh.
- fee_poin = biaya round-trip mesin (bt.net_cost) x harga median kelompok.
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import backtest as bt
from supertrend3 import SYMBOL, load

BINS = [0, 20000, 30000, 50000, 75000, 1e9]
LABELS = ["<20k", "20-30k", "30-50k", "50-75k", ">75k"]
TARGET = 1000.0
WIN = 288            # 24 jam dalam candle 5m
STEP = 12            # titik awal tiap 1 jam
CHUNK = 2000


def first_touch(o, h, l, starts, target):
    """Return array kode per start: 1 = +target dulu, -1 = -target dulu, 2 = candle sama, 0 = tidak ada dalam WIN candle."""
    hw, lw = sliding_window_view(h, WIN), sliding_window_view(l, WIN)
    out = np.zeros(len(starts), dtype=int)
    for a in range(0, len(starts), CHUNK):
        s = starts[a:a + CHUNK]
        e = o[s][:, None]
        up, dn = hw[s] >= e + target, lw[s] <= e - target
        ui = np.where(up.any(1), up.argmax(1), WIN)
        di = np.where(dn.any(1), dn.argmax(1), WIN)
        out[a:a + CHUNK] = np.where(ui < di, 1, np.where(di < ui, -1, np.where(ui < WIN, 2, 0)))
    return out


def run(d5):
    o, h, l = (d5[k].to_numpy(dtype=float) for k in ("open", "high", "low"))
    starts = np.arange(0, len(o) - WIN, STEP)
    code = first_touch(o, h, l, starts, TARGET)
    day = d5["ts"].dt.tz_convert(None).dt.normalize()
    daily = pd.DataFrame({"day": day, "o": o, "h": h, "l": l}).groupby("day").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), n=("o", "size"))
    daily = daily[daily["n"] >= 280]                            # buang hari yang datanya belum penuh
    daily["rng"] = daily["h"] - daily["l"]
    daily["b"] = pd.cut(daily["o"], BINS, labels=LABELS)
    ent = pd.DataFrame({"b": pd.cut(o[starts], BINS, labels=LABELS), "code": code})
    cost = bt.net_cost(SYMBOL)
    rows = []
    for b in LABELS:
        d, e = daily[daily["b"] == b], ent[ent["b"] == b]
        if len(d) == 0:
            continue
        r = {"harga": b, "hari": len(d), "harga_med": round(float(d["o"].median())), "range_hari_med": round(float(d["rng"].median())),
             "%hari>=1000": round(float((d["rng"] >= 1000).mean() * 100), 1), "%hari>=2000": round(float((d["rng"] >= 2000).mean() * 100), 1),
             "fee_poin": round(cost * float(d["o"].median()))}
        for nm, c in (("naik_dulu%", 1), ("turun_dulu%", -1), ("tie%", 2), ("tidak_ada%", 0)):
            r[nm] = round(float((e["code"] == c).mean() * 100), 1)
        rows.append(r)
    fence = "`" * 3
    bt._emit(f"\n### Gerak BTC dalam poin per kelompok harga (candle 5m, {d5['ts'].iloc[0].date()} s/d {d5['ts'].iloc[-1].date()}). "
             f"range_hari = high - low per hari UTC. naik_dulu / turun_dulu = dari open tiap jam, +{TARGET:g} atau -{TARGET:g} poin tersentuh lebih dulu "
             "dalam 24 jam (tie = candle 5m sama, tidak_ada = tidak ada yang tersentuh). fee_poin = biaya round-trip dalam poin\n"
             + fence + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + fence)
    return rows


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(load(ex, "5m"))


if __name__ == "__main__":
    main()
