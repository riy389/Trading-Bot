#!/usr/bin/env python3
"""
ZERO LAG TREND SIGNALS (AlgoAlpha), BTC. Port indikator (ZLEMA 70, pita = ATR(70) tertinggi 210 bar x 1,2) + aturan SL/TP.
V1 stop-and-reverse di pita: SL = pita lawan, 1R = jarak entry ke pita lawan saat entry. Level pita dari bar TF sebelumnya (tanpa lookahead).
V2 = V1 + time-stop 24 jam kalau belum pernah +1R. V3 = V1 + TP 1R (habis TP/time-stop, tunggu sinyal berlawanan).
V4 = indikator asli: tunggu close menembus pita, entry/balik di open bar berikutnya (pembanding).
Entry/keluar dicek di candle 5m: level tersentuh = fill di level, open melompati level = fill di open, SL dan TP di candle 5m yang sama = SL dulu.
Biaya fee 0,12% + slippage 0,10% per trade (dari nilai posisi). R = hasil / jarak ke pita lawan.
"""
import numpy as np
import pandas as pd

import backtest as bt
import eventtree2 as e2

FEE, SLIP, LEN, MULT = 0.0012, 0.0010, 70, 1.2
TFS = {"1h": 24, "4h": 6}  # TF -> time-stop (candle) = 24 jam
NAMES = ("V1 stop di pita", "V2 +time-stop 24j", "V3 +TP 1R", "V4 tunggu close")


def band(df):
    c, h, l = (df[k].astype(float) for k in ("close", "high", "low"))
    zl = (c + (c - c.shift((LEN - 1) // 2))).ewm(span=LEN, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    vol = tr.ewm(alpha=1 / LEN, adjust=False).mean().rolling(LEN * 3).max() * MULT
    return (zl + vol).to_numpy(), (zl - vol).to_numpy()


def trend(c, up, dn):
    tr, t = np.zeros(len(c), int), 0
    for i in range(1, len(c)):
        if c[i] > up[i] and c[i - 1] <= up[i - 1]:
            t = 1
        if c[i] < dn[i] and c[i - 1] >= dn[i - 1]:
            t = -1
        tr[i] = t
    return tr


def old(o, c, up, dn):
    tr, out, pos, ep, rd, je = trend(c, up, dn), [], 0, 0.0, 0.0, 0
    for i in range(1, len(c) - 1):
        if tr[i] != 0 and tr[i] != tr[i - 1]:
            px = o[i + 1]
            if pos:
                out.append((je, pos, ep, px, rd, "flip"))
            pos, ep, je = tr[i], px, i + 1
            rd = abs((px - dn[i]) / px) if pos == 1 else abs((up[i] - px) / px)
    if pos:
        out.append((je, pos, ep, c[-1], rd, "end"))
    return out


def sim(o5, h5, l5, c5, k5, U, D, tp, tsb):
    pos = last = je = ek = 0
    ep = rd = mfe = 0.0
    out = []
    for j in range(len(o5)):
        u, d = U[j], D[j]
        if u != u or d != d:
            continue
        o, h, l = o5[j], h5[j], l5[j]
        if pos:
            ex = None
            if tsb and k5[j] - ek >= tsb and mfe < 1:
                ex = (o, "ts")
            elif pos == 1:
                if l <= d:
                    ex = (min(o, d), "flip")
                elif tp and h >= ep * (1 + rd):
                    ex = (max(o, ep * (1 + rd)), "tp")
            else:
                if h >= u:
                    ex = (max(o, u), "flip")
                elif tp and l <= ep * (1 - rd):
                    ex = (min(o, ep * (1 - rd)), "tp")
            if ex:
                px, why = ex
                out.append((je, pos, ep, px, rd, why))
                last = pos
                if why == "flip":
                    pos, ep, je, ek, mfe = -pos, px, j, k5[j], 0.0
                    rd = (u - px) / px if pos == -1 else (px - d) / px
                else:
                    pos = 0
                continue
            mfe = max(mfe, ((h - ep) if pos == 1 else (ep - l)) / ep / rd)
            continue
        lg, sh = last <= 0 and h >= u, last >= 0 and l <= d
        if lg and sh:
            lg = (u - o) < (o - d)
            sh = not lg
        if lg:
            px = max(o, u)
            pos, ep, je, ek, mfe, rd = 1, px, j, k5[j], 0.0, (px - d) / px
        elif sh:
            px = min(o, d)
            pos, ep, je, ek, mfe, rd = -1, px, j, k5[j], 0.0, (u - px) / px
    if pos:
        out.append((je, pos, ep, c5[-1], rd, "end"))
    return out


def stats(tr, yr, tf, name, months):
    if not tr:
        return None, []
    je = np.array([t[0] for t in tr])
    d, ep, xp, rd = (np.array([t[i] for t in tr], float) for i in (1, 2, 3, 4))
    g = d * (xp - ep) / ep
    n = g - (FEE + SLIP)
    gR, nR, y = g / rd, (g - FEE - SLIP) / rd, yr[je]
    e = y <= 2024
    ex = " ".join(f"{k}:{v}" for k, v in pd.Series([t[5] for t in tr]).value_counts().items())
    row = {"TF": tf, "varian": name, "trade": len(tr), "per_bulan": round(len(tr) / months, 1), "hit%": round((n > 0).mean() * 100, 1),
           "kotorR/trade": round(gR.mean(), 3), "bersihR/trade": round(nR.mean(), 3), "bersihR_total": round(nR.sum(), 1),
           "kotor_total%": round(g.sum() * 100, 1), "bersih_total%": round(n.sum() * 100, 1), "jarak_SL%": round(rd.mean() * 100, 2),
           "R_21-24": round(nR[e].sum(), 1), "R_25-26": round(nR[~e].sum(), 1), "keluar": ex}
    yrs = [{"TF": tf, "varian": name, "tahun": int(a), "trade": int((y == a).sum()), "hit%": round((n[y == a] > 0).mean() * 100, 1),
            "bersihR": round(nR[y == a].sum(), 1)} for a in np.unique(y)]
    return row, yrs


def run(ex):
    d5 = e2.load(ex, "5m")
    t5, y5 = pd.to_datetime(d5["ts"]).astype("int64").to_numpy(), d5["ts"].dt.year.to_numpy()
    rows, yrs = [], []
    for tf, tsb in TFS.items():
        d = e2.load(ex, tf)
        t = pd.to_datetime(d["ts"]).astype("int64").to_numpy()
        yr, step = d["ts"].dt.year.to_numpy(), int(np.median(np.diff(t)))
        months = (d["ts"].iloc[-1] - d["ts"].iloc[0]).days / 30
        o, c = d["open"].to_numpy(float), d["close"].to_numpy(float)
        up, dn = band(d)
        m = t5 < t[-1] + step
        k5 = np.searchsorted(t, t5[m], side="right") - 1
        ok = k5 >= 0
        kk = np.clip(k5, 0, None)
        U = np.where(ok, np.r_[np.nan, up[:-1]][kk], np.nan)
        D = np.where(ok, np.r_[np.nan, dn[:-1]][kk], np.nan)
        a5 = [d5[k].to_numpy(float)[m] for k in ("open", "high", "low", "close")]
        runs = [(sim(*a5, k5, U, D, False, 0), y5[m]), (sim(*a5, k5, U, D, False, tsb), y5[m]),
                (sim(*a5, k5, U, D, True, 0), y5[m]), (old(o, c, up, dn), yr)]
        for nm, (tr, yy) in zip(NAMES, runs):
            r, y = stats(tr, yy, tf, nm, months)
            if r:
                rows.append(r)
                yrs += y
    fence = "`" * 3
    bt._emit("\n### Zero Lag BTC. R = hasil / jarak entry ke pita lawan (1R = 1% equity kalau ukuran posisi = 1%/jarak). bersihR sudah termasuk fee 0,12% + slippage 0,10%. "
             "total% = jumlah hasil per trade tanpa leverage. keluar: flip = kena pita lawan, tp, ts = time-stop\n"
             + fence + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + fence)
    bt._emit("\n### Per tahun (bersihR)\n" + fence + "\n" + pd.DataFrame(yrs).to_string(index=False) + "\n" + fence)


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
