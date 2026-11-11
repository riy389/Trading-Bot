#!/usr/bin/env python3
"""
EVENT STUDY BTC: fitur apa SEBELUM sebuah gerakan >= 2% yang beda dari kondisi sebelum candle tenang (< 1%).
Spesifikasi: diskusi 11 Nov 2026 (Issue #1, checkpoint 11 Nov).

- Event A: satu candle |close - open| >= 2%. Kontrol A: |close - open| < 1%.
- Event B: dari OPEN candle t, harga menyentuh +2% atau -2% LEBIH DULU dalam N candle ke depan. Kontrol B: tidak menyentuh +-1%.
  Satu candle yang menyentuh keduanya sekaligus dibuang (arah tidak jelas). Event yang mulai < N candle setelah event sebelumnya dibuang (dedup).
- Fitur HANYA dari candle t-1 ... t-L (sudah close sebelum t mulai). Candle t dan t+1 hanya dilaporkan (tabel "belajar"), bukan fitur.
- Skor = AUC (0,5 = tidak ada informasi) di paruh pertama (IS) dan kedua (OOS). LULUS = AUC searah di kedua paruh,
  keduanya >= 0,55 atau <= 0,45, dan kelas kecil >= MIN_N di kedua paruh. Kontrol: label dikocok, dihitung lagi.
"""

import numpy as np
import pandas as pd

import backtest as bt
from utswing import swing_levels

SYMBOL = "BTC/USDT:USDT"
TFS = {"4h": (6, 2), "1h": (12, 2), "15m": (16, 5)}   # tf: (N candle untuk event B, jumlah lag fitur)
EV_PCT, CTL_PCT = 0.02, 0.01
AUC_MIN, MIN_N = 0.05, 100        # dikunci sebelum lihat hasil: |AUC - 0,5| >= 0,05 di kedua paruh, n kelas kecil >= 100 per paruh
SHUFFLES, TOP, WARM, SWING_N = 5, 4, 300, 5
COMPS = {"ANY_vs_CTRL": (1, -1, 0), "UP_vs_CTRL": (1, None, 0), "DOWN_vs_CTRL": (-1, None, 0), "UP_vs_DOWN": (1, None, -1)}


def features(df):
    """Fitur per candle, dihitung di CLOSE candle itu (semua kausal). Return DataFrame nama -> nilai."""
    o, h, l, c, v = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "volume"))
    cs = pd.Series(c)
    atr = bt._rma_atr(h, l, c, 14)
    atr_pct = pd.Series(atr / c)
    f = {"vol_rel": pd.Series(v) / pd.Series(v).rolling(20).mean(), "atr_pct": atr_pct,
         "atr_ratio": atr_pct / atr_pct.rolling(100).mean()}
    sma, sd = cs.rolling(20).mean(), cs.rolling(20).std()
    bbw = 4 * sd / sma
    f["bbw_ratio"] = bbw / bbw.rolling(100).mean()
    f["body"], f["range"], f["ret"] = pd.Series(np.abs(c - o) / o), pd.Series((h - l) / o), pd.Series((c - o) / o)
    ema = {n: cs.ewm(span=n, adjust=False).mean() for n in (20, 50, 200)}
    for n in (20, 50, 200):
        f[f"ema{n}_dist"] = (cs - ema[n]) / pd.Series(atr)
    f["ema20_slope"] = (ema[20] - ema[20].shift(3)) / pd.Series(atr)
    f["ema20_50"] = (ema[20] - ema[50]) / pd.Series(atr)
    f["rsi"] = pd.Series(bt._rsi(c, 14))
    up, dn = np.r_[np.nan, np.diff(h)], np.r_[np.nan, -np.diff(l)]
    with np.errstate(invalid="ignore", divide="ignore"):
        pdm = np.where((up > dn) & (up > 0), up, 0.0)
        mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
        pdm[0] = mdm[0] = np.nan
        tr = np.r_[np.nan, np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])])]
        a14 = bt._rma(tr, 14)
        pdi, mdi = 100 * bt._rma(pdm, 14) / a14, 100 * bt._rma(mdm, 14) / a14
        s = pdi + mdi
        f["adx"] = pd.Series(bt._rma(100 * np.abs(pdi - mdi) / np.where(s == 0, 1.0, s), 14))
    sh, sl = swing_levels(h, l, SWING_N)
    f["sw_hi_dist"], f["sw_lo_dist"] = pd.Series((sh - c) / atr), pd.Series((c - sl) / atr)
    return pd.DataFrame(f)


def lagged(F, L):
    """Fitur candle t = nilai di close candle t-k, k = 1..L."""
    return pd.DataFrame({f"{n}@{k}": F[n].shift(k) for n in F.columns for k in range(1, L + 1)})


def label_a(o, c):
    r = (c - o) / o
    y = np.full(len(c), np.nan)
    y[np.abs(r) < CTL_PCT] = 0
    y[r >= EV_PCT], y[r <= -EV_PCT] = 1, -1
    return y, 0


def label_b(o, h, l, N):
    n = len(o)
    y, amb = np.full(n, np.nan), 0
    for t in range(n - N + 1):
        up, dn = o[t] * (1 + EV_PCT), o[t] * (1 - EV_PCT)
        hh, ll = h[t:t + N], l[t:t + N]
        bu, bd = hh >= up, ll <= dn
        iu = bu.argmax() if bu.any() else N
        idn = bd.argmax() if bd.any() else N
        if iu == N and idn == N:
            if hh.max() < o[t] * (1 + CTL_PCT) and ll.min() > o[t] * (1 - CTL_PCT):
                y[t] = 0
        elif iu < idn:
            y[t] = 1
        elif idn < iu:
            y[t] = -1
        else:
            amb += 1
    last = -10 ** 9
    for t in np.flatnonzero(~np.isnan(y) & (y != 0)):
        if t - last < N:
            y[t] = np.nan
        else:
            last = t
    return y, amb


def auc(x, pos):
    m = np.isfinite(x)
    x, pos = x[m], pos[m]
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 < 2 or n0 < 2:
        return np.nan
    r = pd.Series(x).rank().to_numpy()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def score(X, y, is_half):
    """Semua fitur x semua perbandingan. Return list baris dengan AUC IS / OOS dan status lulus."""
    rows = []
    for comp, (a, b, ctl) in COMPS.items():
        pos_m = np.isin(y, [a] if b is None else [a, b])
        neg_m = y == ctl
        use = pos_m | neg_m
        n_small = {}
        for half, hm in (("IS", is_half), ("OOS", ~is_half)):
            n_small[half] = int(min((pos_m & hm).sum(), (neg_m & hm).sum()))
        for col in X.columns:
            x = X[col].to_numpy()
            r = {"comp": comp, "feature": col}
            for half, hm in (("IS", is_half), ("OOS", ~is_half)):
                k = use & hm
                r["AUC_" + half] = auc(x[k], pos_m[k])
            d1, d2 = r["AUC_IS"] - 0.5, r["AUC_OOS"] - 0.5
            ok = np.isfinite(d1) and np.isfinite(d2) and d1 * d2 > 0
            r["strength"] = min(abs(d1), abs(d2)) if ok else 0.0
            r["n_ok"] = min(n_small.values()) >= MIN_N
            r["lulus"] = int(r["strength"] >= AUC_MIN and r["n_ok"])
            rows.append(r)
    return rows


def analyse(df, tf, N, L, seed=1):
    """Return (baris_top, ringkasan, baris_belajar) untuk satu timeframe, dua definisi event."""
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    F = features(df)
    X = lagged(F, L)
    is_half = (df["ts"] < df["ts"].iloc[0] + (df["ts"].iloc[-1] - df["ts"].iloc[0]) / 2).to_numpy()
    rng = np.random.default_rng(seed)
    top, summ, learn = [], [], []
    for dname, (y, amb) in (("A_1candle", label_a(o, c)), ("B_N%d" % N, label_b(o, h, l, N))):
        y = y.copy()
        y[:WARM] = np.nan
        sc = pd.DataFrame(score(X, y, is_half))
        cnt = lambda v, hm: int(((y == v) & hm).sum())
        info = {"tf": tf, "def": dname, "up_IS": cnt(1, is_half), "up_OOS": cnt(1, ~is_half), "dn_IS": cnt(-1, is_half),
                "dn_OOS": cnt(-1, ~is_half), "ctl": int((y == 0).sum()), "ambigu": amb}
        sh_pass = []
        idx = np.flatnonzero(~np.isnan(y))
        for _ in range(SHUFFLES):
            ys = y.copy()
            ys[idx] = rng.permutation(y[idx])
            sh_pass.append(int(pd.DataFrame(score(X, ys, is_half))["lulus"].sum()))
        summ.append({**info, "tests": len(sc), "lulus": int(sc["lulus"].sum()), "lulus_acak": round(float(np.mean(sh_pass)), 1)})
        for comp in COMPS:
            t = sc[sc["comp"] == comp].sort_values("strength", ascending=False).head(TOP)
            for _, r in t.iterrows():
                top.append({"tf": tf, "def": dname, "comp": comp, "feature": r["feature"], "AUC_IS": round(r["AUC_IS"], 3),
                            "AUC_OOS": round(r["AUC_OOS"], 3), "n_ok": int(r["n_ok"]), "LULUS": int(r["lulus"])})
        # pelajaran: candle sebelum (2, 1), candle event (0), candle sesudah (+1). bukan fitur.
        for cls, nm in ((1, "UP"), (-1, "DOWN"), (0, "CTRL")):
            m = y == cls
            if m.sum() == 0:
                continue
            row = {"tf": tf, "def": dname, "kelas": nm, "n": int(m.sum())}
            for k in (2, 1, 0, -1):
                row[f"vol@{-k:+d}" if k else "vol@0"] = round(float(F["vol_rel"].shift(k)[m].median()), 2)
            for k in (1, 0, -1):
                row[f"rng%@{-k:+d}" if k else "rng%@0"] = round(float(F["range"].shift(k)[m].median() * 100), 2)
            learn.append(row)
    return top, summ, learn


def run(exchange):
    top, summ, learn = [], [], []
    for tf, (N, L) in TFS.items():
        df = bt.fetch_ohlcv_cached(exchange, SYMBOL, tf, bt.DAYS_BACK_TF[tf])
        if df is None or len(df) < 2000 or any(g[2] > bt.MAX_GAP_CANDLES for g in bt.find_ohlcv_gaps(df, tf)):
            print(f"  [!] Data {tf} tidak layak, skip.")
            continue
        df = df.iloc[:-1].reset_index(drop=True)
        a, b, c = analyse(df, tf, N, L)
        top, summ, learn = top + a, summ + b, learn + c
        print(f"  {tf}: selesai")
    fence = "`" * 3
    bt._emit("\n### Event study BTC. ringkasan: up/dn = jumlah event naik/turun per paruh (IS/OOS), ctl = kontrol, ambigu = candle yang menyentuh "
             f"+-{EV_PCT * 100:g}% sekaligus (dibuang), tests = jumlah uji, lulus = uji lulus (AUC searah di dua paruh, |AUC-0,5| >= {AUC_MIN}, "
             f"kelas kecil >= {MIN_N} per paruh), lulus_acak = rata-rata lulus kalau label dikocok ({SHUFFLES}x). Lulus yang tidak jauh di atas lulus_acak = kebetulan\n"
             + fence + "\n" + pd.DataFrame(summ).to_string(index=False) + "\n" + fence)
    bt._emit("\n### Fitur terbaik per perbandingan (@k = nilai di close candle t-k). AUC > 0,5: nilai fitur lebih tinggi di kelas pertama "
             "(UP/ANY), < 0,5: lebih rendah. n_ok = kelas kecil >= " + f"{MIN_N} di dua paruh\n"
             + fence + "\n" + pd.DataFrame(top).to_string(index=False) + "\n" + fence)
    bt._emit("\n### Belajar (bukan fitur entry): median volume relatif (vs rata-rata 20 candle) dan range% candle t-2, t-1, t (event), t+1\n"
             + fence + "\n" + pd.DataFrame(learn).to_string(index=False) + "\n" + fence)


def main():
    import ccxt
    exchange = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    exchange.load_markets()
    run(exchange)


if __name__ == "__main__":
    main()
