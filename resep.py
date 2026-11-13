#!/usr/bin/env python3
"""
RESEP: angka indikator SEBELUM gerakan besar (>= 2,7% turun atau naik) dibanding SEBELUM tidak bergerak (kontrol = tidak sampai 1%).
Per indikator: nilai tengah dan rentang tengah (25% .. 75%), dan apakah pemisahannya searah di dua paruh data (stabil).
Lalu BERKAS KASUS (1h): penurunan terbesar 120 hari terakhir, semua angka indikator di candle sebelum penurunan mulai.
Nilai = close candle sebelum entry (tanpa lookahead). Jam kasus dalam WIB.
"""
import numpy as np
import pandas as pd

import backtest as bt
import eventtree as et
import eventtree2 as e2

TOP, NCASE, RECENT_DAYS = 15, 4, 120
SKIP = ("@", "hour_")
BASIC = ["rsi", "stoch_k", "mfi", "cci", "adx", "di_diff", "bb_pctb", "bbw_ratio", "atr_pct", "atr_ratio", "vol_rel", "macd_hist",
         "ema10_dist", "ema20_dist", "ema50_dist", "ema200_dist", "ribbon", "n_ema_above", "obv_slope", "cmf"]


def q(x):
    lo, md, hi = np.percentile(x, [25, 50, 75])
    return f"{md:.4g} ({lo:.4g} .. {hi:.4g})"


def build(df, F, N):
    X = et.build_X(df, F)
    o, h, l = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low"))
    y, mag = et.label_events(o, h, l, N)
    ok = (X.notna().all(axis=1).to_numpy() & ~np.isnan(y)).copy()
    ok[:et.WARM] = False
    return X, y, mag, ok


def table(df, X, y, ok, lab, tf, nm):
    half = df["ts"].iloc[0] + (df["ts"].iloc[-1] - df["ts"].iloc[0]) / 2
    early = (df["ts"] < half).to_numpy()
    sel = ok & ((y == lab) | (y == 0))
    pos = y == lab
    res = []
    for c in X.columns:
        if any(s in c for s in SKIP):
            continue
        v = X[c].to_numpy()
        a = et.auc(v[sel & early], pos[sel & early])
        b = et.auc(v[sel & ~early], pos[sel & ~early])
        st = min(abs(a - 0.5), abs(b - 0.5)) if (a - 0.5) * (b - 0.5) > 0 else 0.0
        res.append((c, a, b, st))
    res.sort(key=lambda r: -r[3])
    rows = []
    for c, a, b, st in res[:TOP]:
        v = X[c].to_numpy()
        rows.append({"TF": tf, "arah": nm, "indikator": c, "sebelum_gerakan": q(v[sel & pos]), "sebelum_diam": q(v[sel & ~pos]),
                     "nilai_gerakan": "lebih tinggi" if a > 0.5 else "lebih rendah", "AUC_p1": round(a, 2), "AUC_p2": round(b, 2), "stabil": round(st, 3)})
    return rows, [r[0] for r in res[:TOP]]


def cases(df, X, y, mag, ok, feats, N):
    recent = (df["ts"] >= df["ts"].iloc[-1] - pd.Timedelta(days=RECENT_DAYS)).to_numpy()
    start = ok & (y == -1) & recent & np.r_[True, y[:-1] != -1]
    picked = []
    for t in sorted(np.where(start)[0], key=lambda t: -mag[t]):
        if all(abs(t - p) > N for p in picked):
            picked.append(t)
        if len(picked) == NCASE:
            break
    sel = ok & ((y == -1) | (y == 0))
    out = pd.DataFrame({"indikator": feats})
    for t in picked:
        lbl = f"{df['ts'].iloc[t].tz_convert('Asia/Jakarta'):%d-%b %H:%M} (-{mag[t] * 100:.1f}%)"
        out[lbl] = [round(float(X[f].iloc[t]), 3) for f in feats]
    out["median_sebelum_turun"] = [round(float(np.median(X[f].to_numpy()[sel & (y == -1)])), 3) for f in feats]
    out["median_sebelum_diam"] = [round(float(np.median(X[f].to_numpy()[sel & (y == 0)])), 3) for f in feats]
    return out


def run(ex):
    d1, d4 = e2.load(ex, "1h"), e2.load(ex, "4h")
    F1, F4 = et.features(d1), et.features(d4)
    rows, case = [], None
    for tf, N in (("1h", 12), ("15m", 16)):
        df = d1 if tf == "1h" else e2.load(ex, tf)
        F = et.features(df)
        if tf == "15m":
            F = e2.ctx(F, df, F1, d1, "1h", "h1_")
            F = e2.ctx(F, df, F4, d4, "4h", "h_")
        else:
            F = et.add_ctx(F, df, F4, d4, "1h", "4h")
        X, y, mag, ok = build(df, F, N)
        for lab, nm in ((-1, "TURUN"), (1, "NAIK")):
            r, names = table(df, X, y, ok, lab, tf, nm)
            rows += r
            if tf == "1h" and lab == -1:
                case = cases(df, X, y, mag, ok, list(dict.fromkeys(BASIC + names)), N)
        print(f"  {tf}: selesai")
    fence = "`" * 3
    bt._emit(f"\n### Angka indikator sebelum gerakan besar (>= {et.EV_PCT * 100:.1f}%) vs sebelum diam (tidak sampai {et.CTL_PCT * 100:.0f}%). "
             "Format: nilai tengah (25% .. 75%). AUC > 0,5 = nilai lebih tinggi sebelum gerakan; stabil = pemisahan searah di dua paruh data (makin besar makin kuat, 0 = tidak konsisten). "
             "Kolom h_ = konteks 4h, h1_ = konteks 1h\n" + fence + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + fence)
    bt._emit(f"\n### Berkas kasus 1h: {NCASE} penurunan terbesar {RECENT_DAYS} hari terakhir. Angka = nilai indikator di close candle SEBELUM penurunan mulai (jam WIB), "
             "dibanding median semua kasus turun dan median sebelum diam\n" + fence + "\n" + case.to_string(index=False) + "\n" + fence)


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
