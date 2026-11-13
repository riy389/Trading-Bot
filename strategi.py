#!/usr/bin/env python3
"""
STRATEGI BERTAHAP: tambah indikator satu per satu (yang paling menaikkan hit rate) di paruh data pertama, lalu aturan yang sama dibekukan di paruh kedua.
Hit rate: entry di open candle t, harga sentuh +-2,7% searah lebih dulu = menang, berlawanan lebih dulu = kalah, tidak sampai keduanya dalam N candle = habis waktu.
SL = TP = 2,7% simetris, tanpa fee, semua candle dihitung (tidak ada yang dibuang). Hit rate = menang / (menang + kalah).
Baris langkah 0 (tanpa syarat) = hit rate kalau asal entry.
"""
import numpy as np
import pandas as pd

import backtest as bt
import eventtree as et
import eventtree2 as e2

STEPS, MIN_DEC, QS = 8, 50, np.linspace(0.1, 0.9, 9)


def stat(m, win, loss):
    w, l = int((win & m).sum()), int((loss & m).sum())
    return int(m.sum()), w, l, (round(w / (w + l) * 100, 1) if w + l else None)


def apply(mask, conds, Xv):
    m = mask.copy()
    for j, op, thr in conds:
        m &= (Xv[:, j] > thr) if op == ">" else (Xv[:, j] <= thr)
    return m


def search(Xv, names, win, loss, is_m, oos_m, tf, nm):
    rows, conds, cur = [], [], is_m.copy()
    a, b = stat(is_m, win, loss), stat(oos_m, win, loss)
    rows.append({"TF": tf, "arah": nm, "langkah": 0, "indikator_ditambah": "(tanpa syarat)", "IS_n": a[0], "IS_menang": a[1], "IS_kalah": a[2], "IS_hit%": a[3],
                 "OOS_n": b[0], "OOS_menang": b[1], "OOS_kalah": b[2], "OOS_hit%": b[3]})
    for k in range(1, STEPS + 1):
        idx = np.where(cur)[0]
        wi, li = win[idx], loss[idx]
        best = None
        for j in range(len(names)):
            v = Xv[idx, j]
            for thr in np.unique(np.quantile(Xv[is_m, j], QS)):
                for op in (">", "<="):
                    s = v > thr if op == ">" else v <= thr
                    w, l = int(wi[s].sum()), int(li[s].sum())
                    if w + l >= MIN_DEC:
                        sc = (w + 1) / (w + l + 2)
                        if best is None or sc > best[0]:
                            best = (sc, j, op, float(thr))
        if best is None:
            break
        conds.append(best[1:])
        cur = apply(is_m, conds, Xv)
        if cur.sum() == idx.size:
            conds.pop()
            break
        a, b = stat(cur, win, loss), stat(apply(oos_m, conds, Xv), win, loss)
        rows.append({"TF": tf, "arah": nm, "langkah": k, "indikator_ditambah": f"{names[best[1]]} {best[2]} {best[3]:.4g}", "IS_n": a[0], "IS_menang": a[1],
                     "IS_kalah": a[2], "IS_hit%": a[3], "OOS_n": b[0], "OOS_menang": b[1], "OOS_kalah": b[2], "OOS_hit%": b[3]})
    return rows, conds


def run(ex):
    d1, d4 = e2.load(ex, "1h"), e2.load(ex, "4h")
    F1, F4 = et.features(d1), et.features(d4)
    rows, yrs = [], []
    for tf, N in (("1h", 12), ("15m", 16)):
        df = d1 if tf == "1h" else e2.load(ex, tf)
        F = et.features(df)
        if tf == "15m":
            F = e2.ctx(F, df, F1, d1, "1h", "h1_")
            F = e2.ctx(F, df, F4, d4, "4h", "h_")
        else:
            F = et.add_ctx(F, df, F4, d4, "1h", "4h")
        X = et.build_X(df, F)
        names, Xv = list(X.columns), X.to_numpy(dtype=float)
        o, h, l = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low"))
        y, _ = et.label_events(o, h, l, N)
        n = len(o)
        valid = np.isfinite(Xv).all(axis=1)
        valid[:et.WARM], valid[n - N + 1:] = False, False
        half = df["ts"].iloc[0] + (df["ts"].iloc[-1] - df["ts"].iloc[0]) / 2
        early = (df["ts"] < half).to_numpy()
        yr = df["ts"].dt.year.to_numpy()
        for lab, nm in ((-1, "TURUN"), (1, "NAIK")):
            win, loss = y == lab, y == -lab
            r, conds = search(Xv, names, win, loss, valid & early, valid & ~early, tf, nm)
            rows += r
            full = apply(valid, conds, Xv)
            for a in np.unique(yr[valid]):
                s = stat(full & (yr == a), win, loss)
                yrs.append({"TF": tf, "arah": nm, "tahun": int(a), "sinyal": s[0], "menang": s[1], "kalah": s[2], "hit%": s[3]})
        print(f"  {tf}: selesai")
    fence = "`" * 3
    bt._emit(f"\n### Strategi bertahap: tiap langkah menambah satu indikator (syarat berlaku bersamaan dengan langkah sebelumnya). Entry di open candle, SL = TP = {et.EV_PCT * 100:.1f}%, "
             "tanpa fee. IS = paruh data pertama (tempat aturan dicari), OOS = paruh kedua (aturan dibekukan, ujian sebenarnya). "
             "hit% = menang / (menang + kalah); sisanya habis waktu. Baris langkah 0 = hit rate kalau asal entry\n"
             + fence + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + fence)
    bt._emit("\n### Aturan akhir (langkah terakhir) per tahun, seluruh data\n" + fence + "\n" + pd.DataFrame(yrs).to_string(index=False) + "\n" + fence)


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
