#!/usr/bin/env python3
"""
ZERO LAG, SEMUA TF + MTF. Aturan indikator asli (tunggu close menembus pita, entry/balik di open bar berikutnya, pita lawan = trailing stop, tanpa TP).
TF sendiri: 5m, 15m, 1h, 4h, 1D (1D dari resample 1h, UTC). MTF = tabel indikator: trend TF di atas (dari bar yang sudah close, tanpa lookahead).
MTF-A: masuk hanya saat TF entry flip DAN semua TF di atasnya searah; posisi dilepas saat TF entry flip balik (balik arah hanya kalau filter searah).
MTF-B: posisi = arah trend hanya selama TF entry dan semua TF di atasnya searah; kalau ada yang berbeda, flat.
Biaya fee 0,12% + slippage 0,10% per trade. R = hasil / jarak entry ke pita lawan TF entry saat sinyal.
"""
import numpy as np
import pandas as pd

import backtest as bt
import eventtree2 as e2
from zerolag import band, trend, stats

ORDER = ("5m", "15m", "1h", "4h", "1d")
FILT = ("5m", "15m", "1h", "4h")  # TF aturan MTF: 4H bias, 1H setup, 15M konfirmasi, 5M trigger


def load_all(ex):
    d = {tf: e2.load(ex, tf) for tf in ("5m", "15m", "1h", "4h")}
    g = d["1h"].set_index("ts")[["open", "high", "low", "close"]].resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index()
    d["1d"] = g.iloc[1:-1].reset_index(drop=True)
    out = {}
    for tf, x in d.items():
        t = pd.to_datetime(x["ts"]).astype("int64").to_numpy()
        up, dn = band(x)
        c = x["close"].to_numpy(float)
        out[tf] = {"t": t, "step": int(np.median(np.diff(t))), "o": x["open"].to_numpy(float), "c": c, "up": up, "dn": dn,
                   "tr": trend(c, up, dn), "yr": x["ts"].dt.year.to_numpy(), "months": (x["ts"].iloc[-1] - x["ts"].iloc[0]).days / 30}
    return out


def align(e, h):
    idx = np.searchsorted(h["t"] + h["step"], e["t"] + e["step"], side="right") - 1
    return np.where(idx >= 0, h["tr"][np.clip(idx, 0, None)], 0)


def target(tr, ok, mode):
    if mode == "none":
        return tr.copy()
    if mode == "B":
        return np.where(ok, tr, 0)
    tg, cur = np.zeros(len(tr), int), 0
    for i in range(1, len(tr)):
        if tr[i] != 0 and tr[i] != tr[i - 1]:
            cur = tr[i] if ok[i] else 0
        tg[i] = cur
    return tg


def trades(e, tg):
    o, c, up, dn = e["o"], e["c"], e["up"], e["dn"]
    out, pos, ep, rd, je = [], 0, 0.0, 0.0, 0
    for i in range(1, len(c) - 1):
        if tg[i] != tg[i - 1]:
            px = o[i + 1]
            if pos:
                out.append((je, pos, ep, px, rd, "flip" if tg[i] == -pos else "lepas"))
            pos = tg[i]
            if pos:
                ep, je = px, i + 1
                rd = max(abs((px - dn[i]) / px) if pos == 1 else abs((up[i] - px) / px), 1e-4)
    if pos:
        out.append((je, pos, ep, c[-1], rd, "end"))
    return out


def run(ex):
    D = load_all(ex)
    rows, yrs = [], []
    for tf in ORDER:
        e = D[tf]
        above = FILT[FILT.index(tf) + 1:] if tf in FILT else ()
        sets = [above[-k:] for k in range(1, len(above) + 1)]
        modes = [("sendiri", "none", ())] + [(f"MTF-{m} +{'/'.join(st)}", m, st) for st in sets for m in ("A", "B")]
        for nm, md, st in modes:
            ok = e["tr"] != 0
            for h in st:
                ok = ok & (align(e, D[h]) == e["tr"])
            r, y = stats(trades(e, target(e["tr"], ok, md)), e["yr"], tf, nm, e["months"])
            if r:
                rows.append(r)
                yrs += y
    fence = "`" * 3
    bt._emit("\n### Zero Lag semua TF + MTF (tanpa TP, pita lawan = trailing, balik arah di close). bersihR sudah termasuk fee 0,12% + slippage 0,10%. "
             "total% = jumlah hasil per trade tanpa leverage. keluar: flip = balik arah, lepas = keluar karena filter/sinyal tanpa balik\n"
             + fence + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + fence)
    bt._emit("\n### Per tahun (bersihR)\n" + fence + "\n" + pd.DataFrame(yrs).to_string(index=False) + "\n" + fence)


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
