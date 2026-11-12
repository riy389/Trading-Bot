#!/usr/bin/env python3
"""
EVENT TREE 2: ARAH (UP vs DOWN) di 15m. Fitur = 15m + konteks 1h (h1_) + konteks 4h (h4_), semua dari candle yang sudah close.
Pertanyaan: indikator di TF mana yang paling membedakan event naik vs turun, dan stabil di paruh 1 dan paruh 2?
Batas acak = label digeser melingkar (autokorelasi tetap, hubungan dengan fitur putus), skor terbaik dari semua fitur.
Tidak ada entry/SL/TP.
"""
import numpy as np
import pandas as pd

import backtest as bt
import eventtree as et

N, SHIFTS, TOP = 16, 30, 8


def load(ex, tf):
    d = bt.fetch_ohlcv_cached(ex, et.SYMBOL, tf, bt.DAYS_BACK_TF[tf])
    if d is None or len(d) < 500 or any(g[2] > bt.MAX_GAP_CANDLES for g in bt.find_ohlcv_gaps(d, tf)):
        raise SystemExit(f"Data {tf} tidak layak")
    return d.iloc[:-1].reset_index(drop=True)


def ctx(F, df, Fh, dfh, tfh, pre):
    c = et.add_ctx(pd.DataFrame(index=F.index), df, Fh, dfh, "15m", tfh)
    for k in c.columns:
        F[pre + k[2:]] = c[k]
    return F


def grp(n):
    return "1h" if n.startswith("h1_") else "4h" if n.startswith("h4_") else "15m"


def run(ex):
    df, d1, d4 = load(ex, "15m"), load(ex, "1h"), load(ex, "4h")
    F = ctx(et.features(df), df, et.features(d1), d1, "1h", "h1_")
    F = ctx(F, df, et.features(d4), d4, "4h", "h4_")
    X = et.build_X(df, F)
    names, Xv = list(X.columns), X.to_numpy()
    o, h, l = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low"))
    y, _ = et.label_events(o, h, l, N)
    fin = np.isfinite(Xv).all(axis=1)
    fin[:et.WARM] = False
    half = df["ts"].iloc[0] + (df["ts"].iloc[-1] - df["ts"].iloc[0]) / 2
    early = (df["ts"] < half).to_numpy()

    def score(yy):
        ev = (yy == 1) | (yy == -1)
        mi, mo = fin & ev & early, fin & ev & ~early
        a = np.array([et.auc(Xv[mi, j], yy[mi] == 1) for j in range(len(names))]) - 0.5
        b = np.array([et.auc(Xv[mo, j], yy[mo] == 1) for j in range(len(names))]) - 0.5
        return a, b, np.where(a * b > 0, np.minimum(np.abs(a), np.abs(b)), 0.0), mi, mo

    a, b, st, mi, mo = score(y)
    rng = np.random.default_rng(1)
    null = [score(np.roll(y, int(s)))[2].max() for s in rng.integers(3000, len(y) - 3000, SHIFTS)]
    thr = float(np.percentile(null, 95))
    allm = mi | mo
    fence = "`" * 3
    bt._emit(f"\n### Event tree 2 (arah, 15m, N={N}). UP: IS {int((y[mi] == 1).sum())} / OOS {int((y[mo] == 1).sum())}, "
             f"DOWN: IS {int((y[mi] == -1).sum())} / OOS {int((y[mo] == -1).sum())}. Skor stabil = min(|AUC-0,5|) di dua paruh, 0 kalau arahnya beda. "
             f"Batas acak (persentil 95 dari skor terbaik di data yang labelnya digeser) = {thr:.3f}")
    summ, rows = [], []
    for g in ("15m", "1h", "4h"):
        idx = [j for j, n in enumerate(names) if grp(n) == g]
        top = sorted(idx, key=lambda j: -st[j])
        summ.append({"TF": g, "n_fitur": len(idx), "di_atas_batas_acak": int(sum(st[j] > thr for j in idx)), "skor_terbaik": round(float(st[top[0]]), 3)})
        for j in top[:TOP]:
            rows.append({"TF": g, "fitur": names[j], "AUC_IS": round(0.5 + a[j], 3), "AUC_OOS": round(0.5 + b[j], 3), "skor": round(float(st[j]), 3),
                         "med_UP": round(float(np.median(Xv[allm & (y == 1), j])), 4), "med_DOWN": round(float(np.median(Xv[allm & (y == -1), j])), 4)})
    bt._emit("\n### Ringkasan per TF\n" + fence + "\n" + pd.DataFrame(summ).to_string(index=False) + "\n" + fence)
    bt._emit(f"\n### {TOP} fitur teratas per TF (AUC > 0,5 = nilai lebih tinggi sebelum event NAIK)\n" + fence + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + fence)


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
