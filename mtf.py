#!/usr/bin/env python3
"""
MTF God Setup, BTC, 5 tahun. Perbandingan D (baseline) vs E, F, G1, G2.
Tiap konfigurasi hanya beda satu hal dari pendahulunya:
E = D + zona FVG aktif; F = E + Sweep OR FVG; G1 = F tanpa SMA50 15M; G2 = F + BEP.
"""
import numpy as np
import pandas as pd

import backtest as bt
import klasik as K

MAX_PCTB = 0.75
PIV4, PIV1, PIV15 = 5, 5, 3
RR, MAXH5 = 3.0, 576
POI_AGE, ARM15, FILL5, OB_LOOK, WARM = 72, 48, 36, 12, 200
BASE = dict(zone=False, or_sw=False, tr15=True, bep=False, w1=0.80, hlo=0.004, hhi=0.015, rlo=0.007, rhi=0.015, sw=12)
CFG = {
    "D baseline": dict(BASE),
    "E FVG zona aktif": dict(BASE, zone=True),
    "F E + Sweep OR FVG": dict(BASE, zone=True, or_sw=True),
    "G1 F tanpa SMA15": dict(BASE, zone=True, or_sw=True, tr15=False),
    "G2 F + BEP": dict(BASE, zone=True, or_sw=True, bep=True),
}
H1 = 3600 * 10 ** 9


def ns(d):
    i = pd.DatetimeIndex(d["ts"])
    try:
        i = i.as_unit("ns")
    except AttributeError:
        pass
    return i.asi8


def ind(d):
    c = d["close"]
    sma, sd = c.rolling(20).mean(), c.rolling(20).std()
    sma50 = c.rolling(50).mean()
    bw = 4 * sd / sma
    return dict(pctb=((c - (sma - 2 * sd)) / (4 * sd)).to_numpy(), ratio=(bw / bw.rolling(100).mean()).to_numpy(),
                half=(2 * sd / sma).to_numpy(), up=((c > sma50) & (sma50 > sma50.shift(3))).to_numpy(),
                dn=((c < sma50) & (sma50 < sma50.shift(3))).to_numpy())


def lastpiv(x, p, hi):
    n, s = len(x), pd.Series(x)
    w = (s.rolling(2 * p + 1, center=True).max() if hi else s.rolling(2 * p + 1, center=True).min()).to_numpy()
    js = np.flatnonzero(x == w)
    js = js[js + p < n]
    pr, ix = np.full(n, np.nan), np.full(n, np.nan)
    pr[js + p], ix[js + p] = x[js], js
    return pd.Series(pr).ffill().to_numpy(), pd.Series(ix).ffill().fillna(-1).to_numpy().astype(int)


def bias4(d4):
    h, l, c = (d4[k].to_numpy(float) for k in ("high", "low", "close"))
    ph, phi = lastpiv(h, PIV4, True)
    pl, pli = lastpiv(l, PIV4, False)
    b, cur, uh, ul = np.zeros(len(c), int), 0, -2, -2
    for t in range(len(c)):
        if phi[t] >= 0 and phi[t] != uh and c[t] > ph[t]:
            cur, uh = 1, phi[t]
        elif pli[t] >= 0 and pli[t] != ul and c[t] < pl[t]:
            cur, ul = -1, pli[t]
        b[t] = cur
    return b


def view(d, I, s):
    o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    if s > 0:
        return dict(o=o, h=h, l=l, c=c, pb=I["pctb"], tr=I["up"])
    return dict(o=-o, h=-l, l=-h, c=-c, pb=1 - I["pctb"], tr=I["dn"])


def prep1(v, d1, s):
    o, h, l, c = v["o"], v["h"], v["l"], v["c"]
    n, day = len(c), d1["ts"].dt.floor("D")
    if s > 0:
        pdl = day.map(d1.groupby(day)["low"].min().shift(1)).to_numpy(float)
    else:
        pdl = -day.map(d1.groupby(day)["high"].max().shift(1)).to_numpy(float)
    pl, _ = lastpiv(l, PIV1, False)
    plp = np.r_[np.nan, pl[:-1]]
    sweep = ((l < pdl) & (c > pdl)) | ((l < plp) & (c > plp))
    v["sweep"] = sweep
    v["ls"] = pd.Series(np.where(sweep, np.arange(n), np.nan)).ffill().fillna(-10 ** 9).to_numpy().astype(int)

    tap_single = np.zeros(n, bool)
    tap_zone = np.zeros(n, bool)
    for k in range(2, n):
        a, b = k - 2, k - 1
        if l[k] > h[a] and c[b] > o[b] and (c[b] - o[b]) >= 0.5 * (h[b] - l[b]) and c[a] < o[a]:
            end_w = min(k + 1 + POI_AGE, n)
            hit = np.flatnonzero(l[k + 1:end_w] <= l[k])
            if len(hit):
                first_idx = k + 1 + hit[0]
                if c[first_idx] >= l[a]:
                    tap_single[first_idx] = True
                # zona aktif: selama belum ada close menembus dasar OB (l[a])
                for m in range(first_idx, end_w):
                    if c[m] < l[a]:
                        break
                    if l[m] <= l[k]:
                        tap_zone[m] = True
    v["tap_single"] = tap_single
    v["tap_zone"] = tap_zone


def trade5(V5, ts5, L, t15, T, sweep_t, cf):
    k0, kk = np.searchsorted(ts5, t15), np.searchsorted(ts5, T) - 1
    ks = [k for k in range(k0, kk + 1) if V5["c"][k] > L]
    if not ks:
        return "noob", None, 0
    k = ks[0]
    ob = next((q for q in range(k - 1, max(k - 1 - OB_LOOK, 0) - 1, -1) if V5["c"][q] < V5["o"][q]), None)
    if ob is None or not V5["pb"][ob] <= MAX_PCTB:
        return "noob", None, 0
    entry, sl = V5["o"][ob], V5["l"][np.searchsorted(ts5, sweep_t):kk + 1].min()
    risk = (entry - sl) / abs(entry)
    if risk < cf["rlo"] or (cf["rhi"] and risk > cf["rhi"]):
        return "noob", None, 0
    f = next((q for q in range(kk + 1, min(kk + 1 + FILL5, len(ts5))) if V5["l"][q] <= entry), None)
    if f is None:
        return "nofill", None, ts5[min(kk + FILL5, len(ts5) - 1)]

    end = min(f + MAXH5, len(ts5) - 1)
    tp = entry + RR * (entry - sl)
    be_trigger = entry + 1.5 * (entry - sl)
    be_sl = entry + 0.0015 * abs(entry)  # BEP + 0.15%, benar untuk long dan short (ruang harga short dinegatifkan)

    cur_sl = sl
    r, j, how = (V5["c"][end] - entry) / abs(entry), end, "t"
    for q in range(f, end + 1):
        if V5["l"][q] <= cur_sl:
            r = (cur_sl - entry) / abs(entry)
            j = q
            how = "b" if cur_sl > sl else "s"
            break
        if q > f and V5["h"][q] >= tp:
            r = RR * risk
            j = q
            how = "p"
            break
        if cf["bep"] and V5["h"][q] >= be_trigger:
            cur_sl = max(cur_sl, be_sl)

    mfe = (V5["h"][f:j + 1].max() - entry) / abs(entry)
    return "ok", (f, j, r, mfe, how, risk), ts5[j] + 5 * 60 * 10 ** 9


def scan(cf, X):
    ts, V1, V15, V5, I = X["ts"], X["V1"], X["V15"], X["V5"], X["I"]
    b4 = np.where((X["b"] == 1) & I["4h"]["up"], 1, np.where((X["b"] == -1) & I["4h"]["dn"], -1, 0))
    k = np.searchsorted(ts["4h"] + 4 * H1, ts["1h"] + H1, side="right") - 1
    b1 = np.where(k >= 0, b4[np.maximum(k, 0)], 0)
    ok1 = (I["1h"]["ratio"] >= cf["w1"]) & (I["1h"]["half"] >= cf["hlo"]) & ((I["1h"]["half"] <= cf["hhi"]) if cf["hhi"] else True)
    n1, n15 = len(ts["1h"]), len(ts["15m"])
    F = dict(bias=0, setup=0, lebar=0, choch=0, ob5=0, terisi=0)
    tr, t, busy = [], WARM, 0
    while t < n1:
        if ts["1h"][t] < busy or b1[t] == 0:
            t += 1
            continue
        s, v = int(b1[t]), V1[int(b1[t])]
        F["bias"] += 1

        tap_arr = v["tap_zone"] if cf["zone"] else v["tap_single"]
        has_fvg_sweep = tap_arr[t] and (0 <= t - v["ls"][t] <= cf["sw"])
        has_pure_sweep = cf["or_sw"] and v["sweep"][t]
        if not (has_fvg_sweep or has_pure_sweep):
            t += 1
            continue
        F["setup"] += 1

        if not (ok1[t] and v["pb"][t] <= MAX_PCTB):
            t += 1
            continue
        F["lebar"] += 1

        sweep_idx = t if (has_pure_sweep and not has_fvg_sweep) else v["ls"][t]
        sweep_t, w = ts["1h"][sweep_idx], V15[s]
        u0 = np.searchsorted(ts["15m"], ts["1h"][t] + H1)
        hit = None
        for u in range(max(u0, 1), min(u0 + ARM15, n15)):
            L, pi = w["lph"][u - 1], w["lpi"][u - 1]
            tr_ok = w["tr"][u] if cf["tr15"] else True
            if pi >= 0 and ts["15m"][pi] >= sweep_t and w["c"][u] > L and w["c"][u - 1] <= L and tr_ok:
                hit = (u, L)
                break
        if hit is None:
            t += 1
            continue
        F["choch"] += 1

        u, L = hit
        why, res, busy_new = trade5(V5[s], ts["5m"], L, ts["15m"][u], ts["15m"][u] + 15 * 60 * 10 ** 9, sweep_t, cf)
        if why == "noob":
            t += 1
            continue
        F["ob5"] += 1
        busy = busy_new
        if res is not None:
            F["terisi"] += 1
            tr.append(res)
        t += 1
    return F, tr


def stat(nm, per, tr, ts5, mask_year):
    if not tr:
        return {"konfig": nm, "periode": per, "trade": 0}
    f, j, r, mfe, how, risk = (np.array(z) for z in zip(*tr))
    f, r, mfe, risk = f.astype(int), r.astype(float), mfe.astype(float), risk.astype(float)
    m = mask_year(ts5.dt.year.to_numpy()[f])
    if not m.any():
        return {"konfig": nm, "periode": per, "trade": 0}
    r, mfe, risk, how = r[m], mfe[m], risk[m], how[m]
    net = r - K.FEE
    return {"konfig": nm, "periode": per, "trade": len(r), "hit%": round((net > 0).mean() * 100, 1), "TP%": round((how == "p").mean() * 100, 1),
            "SL%": round((how == "s").mean() * 100, 1), "BE%": round((how == "b").mean() * 100, 1), "SL_rata%": round(risk.mean() * 100, 2),
            "MFE_rata%": round(mfe.mean() * 100, 2), "kotor_total%": round(r.sum() * 100, 1), "fee_total%": round(len(r) * K.FEE * 100, 1),
            "bersih_total%": round(net.sum() * 100, 1), "terburuk%": round(net.min() * 100, 1), "terbaik%": round(net.max() * 100, 1)}


def run(ex):
    D = {}
    for tf in ("4h", "1h", "15m", "5m"):
        D[tf] = bt.fetch_ohlcv_cached(ex, bt.PAIRS[0], tf, bt.DAYS_BACK_TF[tf])
        if D[tf] is None:
            raise SystemExit(f"data {tf} kosong/ditolak (jalankan Run Cek Data)")
    I = {tf: ind(d) for tf, d in D.items()}
    X = dict(I=I, ts={tf: ns(d) for tf, d in D.items()}, b=bias4(D["4h"]), V1={}, V15={}, V5={})
    for s in (1, -1):
        X["V1"][s], X["V15"][s], X["V5"][s] = view(D["1h"], I["1h"], s), view(D["15m"], I["15m"], s), view(D["5m"], I["5m"], s)
        prep1(X["V1"][s], D["1h"], s)
        X["V15"][s]["lph"], X["V15"][s]["lpi"] = lastpiv(X["V15"][s]["h"], PIV15, True)
    names = ["1H bar dgn bias 4H", "setup: sapuan / POI 1H", "lolos BB 1H", "CHoCH 15M", "OB 5M + SL batas", "limit terisi"]
    fun, rows = {"lapisan": names}, []
    for nm, cf in CFG.items():
        F, tr = scan(cf, X)
        fun[nm.split()[0]] = list(F.values())
        for per, mk in (("semua", lambda y: y > 0), ("2021-2024", lambda y: y <= 2024), ("2025-2026", lambda y: y >= 2025)):
            rows.append(stat(nm, per, tr, D["5m"]["ts"], mk))
    fence = "`" * 3
    bt._emit("\n### MTF Ekspansi Frekuensi (D, E, F, G1, G2). Gugur per lapisan:\n"
             + fence + "\n" + pd.DataFrame(fun).to_string(index=False) + "\n" + fence)
    bt._emit("\n### Hasil (kotor = sebelum fee; bersih_total% = setelah fee 0,12%)\n" + fence + "\n"
             + pd.DataFrame(rows).to_string(index=False) + "\n" + fence)


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
