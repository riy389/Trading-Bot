#!/usr/bin/env python3
"""
EVENT TREE BTC: cari TANDA dari history, bukan dari teori. Bukan backtest strategi, tidak ada entry/exit/SL/TP.

- Event (look-ahead HANYA untuk memberi label): dari open candle t, harga menyentuh +EV_PCT atau -EV_PCT lebih dulu dalam N candle.
  EV_PCT 2,7% (sekitar 2.000 USD di harga 75rb), BIG_PCT 5,3% (sekitar 4.000 USD). Kontrol = tidak sampai 1% dalam N candle.
  Candle yang menyentuh dua sisi sekaligus dibuang. TIDAK ada dedup: dedup (membuang candle tepat setelah event) terbukti membuat tanda arah palsu di data acak.
  Akibatnya candle di tengah gerakan juga berlabel event; jumlah event efektif lebih kecil dari jumlah barisnya.
- Fitur = SEMUA indikator (pita EMA 10/20/50/100/200, RSI, MACD, stochastic, CCI, MFI, ADX/DI, BB, OBV, CMF, volume, struktur, divergence,
  sweep, MSS, FVG, konteks 4h) dihitung di close candle sebelum t (t-1, sebagian t-2 dan t-3). Tidak ada yang melihat ke depan.
- Pohon klasifikasi (UP / DOWN / kontrol) dilatih di paruh pertama, lalu aturan yang sama (beku) dihitung di paruh kedua sebagai pembanding.
  Tidak ada "lulus": kamu lihat sendiri seberapa sering kondisi itu diikuti event (P) dan seberapa besar kemungkinannya dibanding rata-rata (lift).
- Tabel kuantil: per indikator, berapa % kejadian UP / DOWN di tiap rentang nilainya (batas dari data, bukan dari buku).
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

import backtest as bt

SYMBOL = "BTC/USDT:USDT"
TFS = {"1h": (12, "4h"), "15m": (16, "4h")}   # tf: (N candle maksimal untuk event, tf konteks)
EV_PCT, BIG_PCT, CTL_PCT = 0.027, 0.053, 0.01
PIV_N, WARM = 5, 400
DEPTH, MIN_LEAF = 5, 40
LIFT_MIN, MIN_EV = 2.0, 20      # daun ditampilkan kalau di paruh 1 kejadiannya >= 2x rata-rata dan event >= 20
TOP_FEAT, NBINS = 6, 5


# ---------------- fitur ----------------

def pivot_state(h, l, rsi, n):
    """Per candle i (close): struktur dari 2 pivot high/low terkonfirmasi terakhir (pivot j baru dikenal di i = j+n) + divergence RSI."""
    N = len(h)
    o = {k: np.full(N, np.nan) for k in ("st_hi", "st_lo", "bear_div", "bull_div", "age_hi", "age_lo", "sh", "sl")}
    ph, pl = [], []
    for i in range(N):
        j = i - n
        if j >= n:
            if h[j] == h[j - n:j + n + 1].max():
                ph = (ph + [(j, h[j], rsi[j])])[-2:]
            if l[j] == l[j - n:j + n + 1].min():
                pl = (pl + [(j, l[j], rsi[j])])[-2:]
        if len(ph) == 2:
            o["st_hi"][i] = 1 if ph[1][1] > ph[0][1] else -1
            o["bear_div"][i] = float(ph[1][1] > ph[0][1] and ph[1][2] < ph[0][2])
            o["age_hi"][i], o["sh"][i] = i - ph[1][0], ph[1][1]
        if len(pl) == 2:
            o["st_lo"][i] = 1 if pl[1][1] > pl[0][1] else -1
            o["bull_div"][i] = float(pl[1][1] < pl[0][1] and pl[1][2] > pl[0][2])
            o["age_lo"][i], o["sl"][i] = i - pl[1][0], pl[1][1]
    return o


def features(df):
    """Fitur per candle, dihitung di CLOSE candle itu (kausal)."""
    o, h, l, c, v = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "volume"))
    cs, S = pd.Series(c), pd.Series
    atr = bt._rma_atr(h, l, c, 14)
    A = S(atr)
    atr_pct = S(atr / c)
    rsi = bt._rsi(c, 14)
    f = {"vol_rel": S(v) / S(v).rolling(20).mean(), "atr_pct": atr_pct, "atr_ratio": atr_pct / atr_pct.rolling(100).mean(),
         "body": S(np.abs(c - o) / o), "range": S((h - l) / o), "ret": S((c - o) / o), "rsi": S(rsi)}
    rng = np.where(h - l == 0, np.nan, h - l)
    f["up_wick"], f["lo_wick"] = S((h - np.maximum(o, c)) / rng), S((np.minimum(o, c) - l) / rng)
    sma, sd = cs.rolling(20).mean(), cs.rolling(20).std()
    f["bbw_ratio"] = (4 * sd / sma) / (4 * sd / sma).rolling(100).mean()
    f["bb_pctb"] = (cs - (sma - 2 * sd)) / (4 * sd)
    ema = {n: cs.ewm(span=n, adjust=False).mean() for n in (12, 20, 26, 50, 200)}
    for n in (20, 50, 200):
        f[f"ema{n}_dist"] = (cs - ema[n]) / A
    f["ema20_slope"] = (ema[20] - ema[20].shift(3)) / A
    f["ema20_50"] = (ema[20] - ema[50]) / A
    macd = ema[12] - ema[26]
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    f["macd_hist"], f["macd_hist_chg"] = hist / A, (hist - hist.shift(1)) / A
    up, dn = np.r_[np.nan, np.diff(h)], np.r_[np.nan, -np.diff(l)]
    with np.errstate(invalid="ignore", divide="ignore"):
        pdm, mdm = np.where((up > dn) & (up > 0), up, 0.0), np.where((dn > up) & (dn > 0), dn, 0.0)
        pdm[0] = mdm[0] = np.nan
        tr = np.r_[np.nan, np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])])]
        a14 = bt._rma(tr, 14)
        pdi, mdi = 100 * bt._rma(pdm, 14) / a14, 100 * bt._rma(mdm, 14) / a14
        f["adx"] = S(bt._rma(100 * np.abs(pdi - mdi) / np.where(pdi + mdi == 0, 1.0, pdi + mdi), 14))

    # --- indikator tambahan (pita EMA 10/20/50/100/200, osilator, volume) ---
    for n in (10, 100):
        ema[n] = cs.ewm(span=n, adjust=False).mean()
    f["ema10_dist"], f["ema100_dist"] = (cs - ema[10]) / A, (cs - ema[100]) / A
    f["ribbon"] = (np.sign(ema[10] - ema[20]) + np.sign(ema[20] - ema[50]) + np.sign(ema[50] - ema[100]) + np.sign(ema[100] - ema[200]))
    f["n_ema_above"] = sum((cs > ema[n]).astype(float) for n in (10, 20, 50, 100, 200))
    for n in (10, 50, 200):
        f[f"ema{n}_slope"] = (ema[n] - ema[n].shift(3)) / A
    f["ema10_20"], f["ema50_200"] = (ema[10] - ema[20]) / A, (ema[50] - ema[200]) / A
    f["rsi7"] = S(bt._rsi(c, 7))
    hh, ll = S(h).rolling(14).max(), S(l).rolling(14).min()
    k = 100 * (cs - ll) / (hh - ll).replace(0, np.nan)
    f["stoch_k"], f["stoch_d"], f["willr"] = k, k.rolling(3).mean(), k - 100
    tp = (S(h) + S(l) + cs) / 3
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    f["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * mad.replace(0, np.nan))
    mf = tp * S(v)
    pos = mf.where(tp > tp.shift(1), 0.0).rolling(14).sum()
    neg = mf.where(tp < tp.shift(1), 0.0).rolling(14).sum()
    f["mfi"] = 100 * pos / (pos + neg).replace(0, np.nan)
    obv = (np.sign(cs.diff()).fillna(0) * S(v)).cumsum()
    f["obv_slope"] = (obv - obv.shift(10)) / S(v).rolling(10).sum()
    clv = ((cs - S(l)) - (S(h) - cs)) / (S(h) - S(l)).replace(0, np.nan)
    f["cmf"] = (clv * S(v)).rolling(20).sum() / S(v).rolling(20).sum()
    f["roc5"], f["roc10"] = cs / cs.shift(5) - 1, cs / cs.shift(10) - 1
    f["di_diff"] = S(pdi - mdi)
    f["di_pos"], f["di_neg"] = S(pdi), S(mdi)
    ps = pivot_state(h, l, rsi, PIV_N)
    for k in ("st_hi", "st_lo", "bear_div", "bull_div", "age_hi", "age_lo"):
        f[k] = S(ps[k])
    f["struct"] = f["st_hi"] + f["st_lo"]
    sh, sl = S(ps["sh"]), S(ps["sl"])
    f["sw_hi_dist"], f["sw_lo_dist"] = (sh - cs) / A, (cs - sl) / A
    shp, slp, cp = sh.shift(1), sl.shift(1), cs.shift(1)
    sweep_up, sweep_dn = ((S(h) > shp) & (cs < shp)).astype(float), ((S(l) < slp) & (cs > slp)).astype(float)
    bos_up, bos_dn = ((cs > shp) & (cp <= shp)).astype(float), ((cs < slp) & (cp >= slp)).astype(float)
    sprev = f["struct"].shift(1)
    mss_up, mss_dn = bos_up * (sprev < 0), bos_dn * (sprev > 0)
    big = S(np.abs(c - o)) > 1.5 * A
    disp = (big * np.sign(c - o)).astype(float)
    for nm, x in (("sweep_up", sweep_up), ("sweep_dn", sweep_dn), ("bos_up", bos_up), ("bos_dn", bos_dn),
                  ("mss_up", mss_up), ("mss_dn", mss_dn)):
        f[nm + "_3"] = x.rolling(3).max()
    f["disp_3"] = disp.rolling(3).apply(lambda x: x[np.argmax(np.abs(x))], raw=True)
    f["fvg_up_5"] = (((S(l) - S(h).shift(2)) / A).clip(lower=0)).rolling(5).max()
    f["fvg_dn_5"] = (((S(l).shift(2) - S(h)) / A).clip(lower=0)).rolling(5).max()
    f["dc_hi_dist"] = (cs - S(h).rolling(20).max().shift(1)) / A
    f["dc_lo_dist"] = (cs - S(l).rolling(20).min().shift(1)) / A
    return pd.DataFrame(f)


CTX = ["struct", "adx", "rsi", "ema10_dist", "ema20_dist", "ema50_dist", "ema100_dist", "ema200_dist", "ribbon", "n_ema_above",
       "ema20_slope", "atr_ratio", "bb_pctb", "macd_hist", "stoch_k", "obv_slope", "di_diff"]


def add_ctx(F, df, Fh, dfh, tf, tfh):
    """Fitur konteks dari TF lebih besar: nilai candle HTF terakhir yang SUDAH close saat candle base close (tanpa lookahead)."""
    th = (dfh["ts"] + pd.Timedelta(tfh)).dt.tz_convert(None).to_numpy("datetime64[ns]")
    tb = (df["ts"] + pd.Timedelta(tf)).dt.tz_convert(None).to_numpy("datetime64[ns]")
    idx = np.searchsorted(th, tb, side="right") - 1
    for k in CTX:
        val = Fh[k].to_numpy()
        F["h_" + k] = np.where(idx >= 0, val[np.clip(idx, 0, None)], np.nan)
    return F


def build_X(df, F):
    """Keputusan di close candle t-1 untuk entry di open t: semua fitur digeser 1, sebagian juga lag 2, plus jam entry."""
    X = F.shift(1)
    for k in ("atr_pct", "range", "ret", "vol_rel", "rsi", "macd_hist", "stoch_k", "ribbon", "ema10_dist", "obv_slope"):
        X[k + "@2"], X[k + "@3"] = F[k].shift(2), F[k].shift(3)
    hr = df["ts"].dt.hour + df["ts"].dt.minute / 60
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    return X


# ---------------- label (look-ahead, hanya untuk memberi nama pada kejadian) ----------------

def label_events(o, h, l, N):
    """y: 1 = naik >= EV_PCT lebih dulu, -1 = turun, 0 = kontrol (tidak sampai CTL_PCT), nan = dibuang. mag = gerak terjauh searah dalam N candle."""
    n = len(o)
    y, mag = np.full(n, np.nan), np.full(n, np.nan)
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
            y[t], mag[t] = 1, hh.max() / o[t] - 1
        elif idn < iu:
            y[t], mag[t] = -1, 1 - ll.min() / o[t]
    return y, mag


def auc(x, pos):
    m = np.isfinite(x)
    x, pos = x[m], pos[m]
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 < 2 or n0 < 2:
        return np.nan
    r = pd.Series(x).rank().to_numpy()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def leaf_rules(tr, names):
    t, out = tr.tree_, {}

    def rec(node, conds):
        if t.children_left[node] == -1:
            out[node] = " & ".join(conds) or "(semua)"
            return
        f, th = names[t.feature[node]], t.threshold[node]
        rec(t.children_left[node], conds + [f"{f}<={th:.3g}"])
        rec(t.children_right[node], conds + [f"{f}>{th:.3g}"])
    rec(0, [])
    return out


def analyse(tf, df, N, Fh=None, dfh=None, tfh=None):
    o, h, l = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low"))
    F = features(df)
    if Fh is not None:
        F = add_ctx(F, df, Fh, dfh, tf, tfh)
    X = build_X(df, F)
    y, mag = label_events(o, h, l, N)
    ok = np.isfinite(X.to_numpy()).all(axis=1) & ~np.isnan(y)
    ok[:WARM] = False
    half = df["ts"].iloc[0] + (df["ts"].iloc[-1] - df["ts"].iloc[0]) / 2
    is_m, oos_m = ok & (df["ts"] < half).to_numpy(), ok & (df["ts"] >= half).to_numpy()
    cls = np.where(y == 1, 1, np.where(y == -1, 2, 0))
    Xi, ci, Xo, co = X[is_m].to_numpy(), cls[is_m], X[oos_m].to_numpy(), cls[oos_m]
    names = list(X.columns)
    allm = is_m | oos_m

    # 1) ringkasan kejadian
    info = {"tf": tf, "N": N, "n_IS": int(is_m.sum()), "n_OOS": int(oos_m.sum()),
            "UP_IS": int((ci == 1).sum()), "UP_OOS": int((co == 1).sum()), "DN_IS": int((ci == 2).sum()), "DN_OOS": int((co == 2).sum()),
            "UP_base%": round(float((cls[allm] == 1).mean() * 100), 2), "DN_base%": round(float((cls[allm] == 2).mean() * 100), 2),
            "UP_capai_big%": round(float((mag[allm & (y == 1)] >= BIG_PCT).mean() * 100), 1),
            "DN_capai_big%": round(float((mag[allm & (y == -1)] >= BIG_PCT).mean() * 100), 1)}

    # 2) profil indikator: median per kelas + AUC dari paruh 1, diurutkan menurut pemisahan
    prof = []
    for j, nm in enumerate(names):
        a_up = auc(Xi[:, j][(ci == 1) | (ci == 0)], (ci[(ci == 1) | (ci == 0)] == 1))
        a_dn = auc(Xi[:, j][(ci == 2) | (ci == 0)], (ci[(ci == 2) | (ci == 0)] == 2))
        prof.append((nm, a_up, a_dn))
    prof = pd.DataFrame(prof, columns=["fitur", "AUC_UP", "AUC_DOWN"])
    prof["kuat"] = np.maximum((prof["AUC_UP"] - 0.5).abs(), (prof["AUC_DOWN"] - 0.5).abs())
    prof = prof.sort_values("kuat", ascending=False)
    Xa, ca = X[allm].reset_index(drop=True), cls[allm]
    med = []
    for nm in prof["fitur"].head(15):
        med.append({"tf": tf, "fitur": nm, "AUC_UP": round(float(prof.loc[prof.fitur == nm, "AUC_UP"].iloc[0]), 3),
                    "AUC_DOWN": round(float(prof.loc[prof.fitur == nm, "AUC_DOWN"].iloc[0]), 3),
                    "med_UP": round(float(Xa.loc[ca == 1, nm].median()), 4), "med_DOWN": round(float(Xa.loc[ca == 2, nm].median()), 4),
                    "med_KONTROL": round(float(Xa.loc[ca == 0, nm].median()), 4)})

    # 3) kuantil: peluang UP / DOWN per rentang nilai indikator (batas dari data)
    qrows = []
    for nm in prof["fitur"].head(TOP_FEAT):
        v = Xa[nm].to_numpy()
        edges = np.unique(np.quantile(v, np.linspace(0, 1, NBINS + 1)))
        b = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2)
        for k in range(len(edges) - 1):
            m = b == k
            qrows.append({"tf": tf, "fitur": nm, "rentang": f"{edges[k]:.4g} s/d {edges[k + 1]:.4g}", "n": int(m.sum()),
                          "P_UP%": round(float((ca[m] == 1).mean() * 100), 2), "P_DOWN%": round(float((ca[m] == 2).mean() * 100), 2)})

    # 4) pohon: kombinasi semua indikator
    tr = DecisionTreeClassifier(max_depth=DEPTH, min_samples_leaf=MIN_LEAF, class_weight="balanced", random_state=1).fit(Xi, ci)
    rules = leaf_rules(tr, names)
    li, lo = tr.apply(Xi), tr.apply(Xo)
    bi, bo = {c: (ci == c).mean() for c in (1, 2)}, {c: (co == c).mean() for c in (1, 2)}
    leaves = []
    for lf in np.unique(li):
        mi, mo = li == lf, lo == lf
        for c, nm in ((1, "UP"), (2, "DOWN")):
            n_ev = int((ci[mi] == c).sum())
            p_i = (ci[mi] == c).mean()
            if n_ev >= MIN_EV and p_i >= LIFT_MIN * bi[c]:
                p_o = (co[mo] == c).mean() if mo.sum() else np.nan
                leaves.append({"tf": tf, "arah": nm, "n_IS": int(mi.sum()), "P_IS%": round(p_i * 100, 1), "lift_IS": round(p_i / bi[c], 1),
                               "cover_IS%": round(n_ev / (ci == c).sum() * 100, 1), "n_OOS": int(mo.sum()),
                               "P_OOS%": round(p_o * 100, 1) if mo.sum() else None,
                               "lift_OOS": round(p_o / bo[c], 1) if mo.sum() and bo[c] > 0 else None,
                               "cover_OOS%": round(int((co[mo] == c).sum()) / max((co == c).sum(), 1) * 100, 1), "aturan": rules[lf]})
    return info, med, qrows, leaves


def run(exchange):
    infos, meds, qrows, leaves, cache = [], [], [], [], {}
    for tf, (N, tfh) in TFS.items():
        df = bt.fetch_ohlcv_cached(exchange, SYMBOL, tf, bt.DAYS_BACK_TF[tf])
        if df is None or len(df) < 2000 or any(g[2] > bt.MAX_GAP_CANDLES for g in bt.find_ohlcv_gaps(df, tf)):
            print(f"  [!] Data {tf} tidak layak, skip.")
            continue
        df = df.iloc[:-1].reset_index(drop=True)
        Fh = dfh = None
        if tfh:
            if tfh not in cache:
                d = bt.fetch_ohlcv_cached(exchange, SYMBOL, tfh, bt.DAYS_BACK_TF[tfh]).iloc[:-1].reset_index(drop=True)
                cache[tfh] = (d, features(d))
            dfh, Fh = cache[tfh]
        i, m, q, lv = analyse(tf, df, N, Fh, dfh, tfh)
        infos.append(i)
        meds += m
        qrows += q
        leaves += lv
        print(f"  {tf}: selesai, {len(lv)} daun tampil")
    fence = "`" * 3
    bt._emit(f"\n### Event tree BTC. Event = harga sentuh +-{EV_PCT * 100:.1f}% (sekitar 2.000 USD) lebih dulu dalam N candle; kontrol = tidak sampai {CTL_PCT * 100:.0f}%. "
             f"*_base% = berapa % candle yang jadi event (rata-rata), *_capai_big% = % event yang sampai {BIG_PCT * 100:.1f}% (sekitar 4.000 USD)\n"
             + fence + "\n" + pd.DataFrame(infos).to_string(index=False) + "\n" + fence)
    bt._emit("\n### Indikator yang paling membedakan sebelum event vs kontrol (nilai di close candle sebelum event, @k = k candle sebelumnya). "
             "AUC > 0,5 = nilai lebih tinggi sebelum event, < 0,5 = lebih rendah; med_* = nilai tengah per kelompok\n"
             + fence + "\n" + pd.DataFrame(meds).to_string(index=False) + "\n" + fence)
    bt._emit("\n### Peluang per rentang nilai indikator (batas rentang dari data: kuantil). P_UP% / P_DOWN% = % candle di rentang itu yang jadi awal event naik / turun\n"
             + fence + "\n" + pd.DataFrame(qrows).to_string(index=False) + "\n" + fence)
    lv = pd.DataFrame(leaves)
    if len(lv):
        bt._emit("\n### Kombinasi semua indikator (pohon) yang kejadiannya >= " + f"{LIFT_MIN:g}" + "x rata-rata di paruh 1; paruh 2 = aturan yang sama dihitung ulang tanpa dilatih. "
                 "P = % candle yang memenuhi aturan lalu jadi awal event, lift = P dibanding rata-rata, cover = % semua event yang tertangkap aturan itu\n"
                 + fence + "\n" + lv.drop(columns=["aturan"]).to_string(index=False) + "\n" + fence
                 + "\n" + "\n".join(f"- {r.tf} {r.arah}: {r.aturan}" for r in lv.itertuples()))


def main():
    import ccxt
    exchange = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    exchange.load_markets()
    run(exchange)


if __name__ == "__main__":
    main()
