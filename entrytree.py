#!/usr/bin/env python3
"""
ENTRY TREE BTC: pohon keputusan dangkal dari semua fitur (struktur, divergence, sweep, MSS, FVG, BB, MACD, RSI, regime, konteks 4h)
untuk cari kondisi ENTRY lengkap dengan arah. SL = TP = 2% (simetris), keluar paksa setelah N candle.

- Sinyal di close candle t-1, entry di OPEN candle t. Semua fitur kausal (hanya data sampai close t-1).
- Hasil per candle t (long): +1R kalau +2% tersentuh dulu, -1R kalau -2% dulu, timeout = (close akhir / open - 1) / 2%. Short = kebalikannya.
  Candle yang menyentuh +-2% sekaligus dihitung kalah (-1R) untuk dua arah.
- Pohon (regresi, kedalaman 4) dilatih di PARUH PERTAMA data. Daun dipilih kalau |rata-rata R| >= SEL_R dan n >= SEL_N di paruh pertama:
  rata-rata positif = LONG, negatif = SHORT. Paruh KEDUA hanya dipakai sekali untuk menguji daun terpilih.
- Kontrol: target paruh pertama dikocok, pipeline sama, hasilnya diuji di paruh kedua yang asli.
- Penilaian = hit rate TP-duluan dan R gross. Net (setelah biaya) hanya kolom info.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

import backtest as bt

SYMBOL = "BTC/USDT:USDT"
TFS = {"4h": (6, None), "1h": (12, "4h"), "15m": (16, "4h")}   # tf: (N candle maksimal tahan, tf konteks)
P = 0.02                       # SL = TP = 2% dari harga entry
PIV_N, WARM = 5, 400
DEPTH, MIN_LEAF = 4, 150
SEL_R, SEL_N, PASS_R, PASS_N = 0.10, 100, 0.05, 100   # dikunci sebelum lihat hasil (0,10R = hit rate sekitar 55%)
SHUFFLES = 5


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


CTX = ["struct", "adx", "rsi", "ema50_dist", "ema200_dist", "atr_ratio", "bb_pctb", "macd_hist"]


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
    for k in ("atr_pct", "range", "ret", "vol_rel", "rsi"):
        X[k + "@2"] = F[k].shift(2)
    hr = df["ts"].dt.hour + df["ts"].dt.minute / 60
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    return X


# ---------------- label ----------------

def outcomes(o, h, l, c, N):
    """R long per candle t (entry open t, SL = TP = P, maks N candle). kind: 1 TP dulu, -1 SL dulu, 0 timeout, 2 menyentuh keduanya sekaligus."""
    n = len(o)
    R, kind = np.full(n, np.nan), np.zeros(n, int)
    for t in range(n - N + 1):
        hh, ll = h[t:t + N], l[t:t + N]
        bu, bd = hh >= o[t] * (1 + P), ll <= o[t] * (1 - P)
        iu = bu.argmax() if bu.any() else N
        idn = bd.argmax() if bd.any() else N
        if iu == N and idn == N:
            R[t], kind[t] = (c[t + N - 1] / o[t] - 1) / P, 0
        elif iu < idn:
            R[t], kind[t] = 1.0, 1
        elif idn < iu:
            R[t], kind[t] = -1.0, -1
        else:
            R[t], kind[t] = -1.0, 2
    return R, kind


def r_dir(R, kind, d):
    return np.where(kind == 2, -1.0, d * R)


def stats(Rd, kind, cost_r, days):
    n = len(Rd)
    if n == 0:
        return {"n": 0}
    win, loss = (Rd >= 1 - 1e-9), (Rd <= -1 + 1e-9)
    return {"n": n, "hit%": round(float(win.sum() / max(win.sum() + loss.sum(), 1) * 100), 1),
            "timeout%": round(float((kind == 0).mean() * 100), 1), "R_gross": round(float(Rd.mean()), 3),
            "R_net": round(float(Rd.mean() - cost_r), 3), "per_day": round(n / days, 2)}


# ---------------- pohon ----------------

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


def fit_select(Xi, Ri, ki):
    tr = DecisionTreeRegressor(max_depth=DEPTH, min_samples_leaf=MIN_LEAF, random_state=1).fit(Xi, Ri)
    leaf = tr.apply(Xi)
    sel = {}
    for lf in np.unique(leaf):
        m = leaf == lf
        mr = Ri[m].mean()
        if m.sum() >= SEL_N and abs(mr) >= SEL_R:
            sel[lf] = 1 if mr > 0 else -1
    return tr, sel


def apply_sel(tr, sel, X, R, kind):
    """Pilih baris yang jatuh di daun terpilih. Return (mask, arah per baris, R searah)."""
    leaf = tr.apply(X)
    d = np.zeros(len(X))
    for lf, s in sel.items():
        d[leaf == lf] = s
    m = d != 0
    return m, d, np.where(kind == 2, -1.0, d * R)


def analyse(df, tf, N, Fh=None, dfh=None, tfh=None, seed=1):
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    F = features(df)
    if Fh is not None:
        F = add_ctx(F, df, Fh, dfh, tf, tfh)
    X = build_X(df, F)
    R, kind = outcomes(o, h, l, c, N)
    ok = np.isfinite(X.to_numpy()).all(axis=1) & np.isfinite(R)
    ok[:WARM] = False
    half = df["ts"].iloc[0] + (df["ts"].iloc[-1] - df["ts"].iloc[0]) / 2
    is_m, oos_m = ok & (df["ts"] < half).to_numpy(), ok & (df["ts"] >= half).to_numpy()
    cost_r = bt.net_cost(SYMBOL) / P
    days_oos = max((df["ts"][oos_m].iloc[-1] - df["ts"][oos_m].iloc[0]).days, 1)
    Xi, Ri, ki = X[is_m].to_numpy(), R[is_m], kind[is_m]
    Xo, Ro, ko = X[oos_m].to_numpy(), R[oos_m], kind[oos_m]
    names = list(X.columns)

    tr, sel = fit_select(Xi, Ri, ki)
    rules = leaf_rules(tr, names)
    li, lo = tr.apply(Xi), tr.apply(Xo)
    leaves, oos_R = [], []
    for lf, s in sorted(sel.items()):
        mi, mo = li == lf, lo == lf
        si = stats(r_dir(Ri[mi], ki[mi], s), ki[mi], cost_r, max((df["ts"][is_m].iloc[-1] - df["ts"][is_m].iloc[0]).days, 1))
        so = stats(r_dir(Ro[mo], ko[mo], s), ko[mo], cost_r, days_oos)
        passed = int(so.get("n", 0) >= PASS_N and so.get("R_gross", -9) >= PASS_R)   # R_gross sudah searah arah daun
        leaves.append({"tf": tf, "arah": "LONG" if s == 1 else "SHORT", "n_IS": si["n"], "hit_IS": si["hit%"], "R_IS": si["R_gross"],
                       "n_OOS": so.get("n", 0), "hit_OOS": so.get("hit%"), "R_OOS": so.get("R_gross"), "LULUS": passed,
                       "aturan": rules[lf]})
    m, d, Rd = apply_sel(tr, sel, Xo, Ro, ko)
    agg = stats(Rd[m], ko[m], cost_r, days_oos)
    ev_tp = ((ko == 1) | (ko == -1))
    cover = float((m & (Rd >= 1 - 1e-9)).sum() / max(ev_tp.sum(), 1) * 100)   # P(sinyal | event): % event TP-duluan yang tertangkap sinyal searah
    base = stats(Ro, ko, cost_r, days_oos)                                     # entry long di semua candle (pembanding)
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(SHUFFLES):
        perm = rng.permutation(len(Ri))
        trs, ss = fit_select(Xi, Ri[perm], ki[perm])
        ms, ds, Rds = apply_sel(trs, ss, Xo, Ro, ko)
        nulls.append(Rds[ms].mean() if ms.sum() else np.nan)
    nulls = np.array(nulls, dtype=float)
    summ = {"tf": tf, "N": N, "n_IS": int(is_m.sum()), "n_OOS": int(oos_m.sum()), "daun": len(sel), "lulus": sum(x["LULUS"] for x in leaves),
            "OOS_n": agg.get("n", 0), "OOS_hit%": agg.get("hit%"), "OOS_R": agg.get("R_gross"), "OOS_R_net(info)": agg.get("R_net"),
            "per_day": agg.get("per_day"), "timeout%": agg.get("timeout%"), "cover%": round(cover, 1),
            "base_long_hit%": base["hit%"], "z_vs_null": None, "null_R_mean": round(float(np.nanmean(nulls)), 3) if np.isfinite(nulls).any() else None,
            "null_R_sd": round(float(np.nanstd(nulls)), 3) if np.isfinite(nulls).any() else None}
    if summ["null_R_mean"] is not None and agg.get("n", 0):
        summ["z_vs_null"] = round((agg["R_gross"] - summ["null_R_mean"]) / max(summ["null_R_sd"], 0.01), 1)
    return summ, leaves


def run(exchange):
    summ, leaves, cache = [], [], {}
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
        s, lv = analyse(df, tf, N, Fh, dfh, tfh)
        summ.append(s)
        leaves += lv
        print(f"  {tf}: {s['daun']} daun terpilih, {s['lulus']} lulus")
    fence = "`" * 3
    bt._emit("\n### Entry tree BTC, SL = TP = 2%, keluar paksa setelah N candle. Pohon dilatih di paruh 1 (IS), diuji sekali di paruh 2 (OOS). "
             f"daun terpilih = |rata-rata R| >= {SEL_R} dan n >= {SEL_N} di IS (R+ = LONG, R- = SHORT). lulus = di OOS searah, n >= {PASS_N}, R >= {PASS_R}. "
             "OOS_hit% = TP-duluan / (TP + SL) semua sinyal terpilih (impas gross = 50%), OOS_R = rata-rata R gross, OOS_R_net(info) = setelah biaya. "
             "cover% = % event TP-duluan (naik atau turun) yang tertangkap sinyal searah. base_long_hit% = entry long di semua candle. "
             "null = label IS dikocok, pipeline sama, diuji di OOS asli (R rata-rata kalau tidak ada informasi)\n"
             + fence + "\n" + pd.DataFrame(summ).to_string(index=False) + "\n" + fence)
    lv = pd.DataFrame(leaves)
    if len(lv):
        bt._emit("\n### Daun terpilih (aturan entry) dan hasilnya di IS vs OOS. Kolom aturan = kondisi di close candle sebelum entry\n"
                 + fence + "\n" + lv.drop(columns=["aturan"]).to_string(index=False) + "\n" + fence
                 + "\n" + "\n".join(f"- {r.tf} {r.arah} (lulus={r.LULUS}): {r.aturan}" for r in lv.itertuples()))


def main():
    import ccxt
    exchange = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    exchange.load_markets()
    run(exchange)


if __name__ == "__main__":
    main()
