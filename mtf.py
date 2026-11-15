#!/usr/bin/env python3
"""
MTF God Setup, BTC, 5 tahun, 4 konfigurasi (A-D) dibandingkan dalam satu run. Semua aturan dihitung dengan data yang sudah tutup (tanpa lihat ke depan).
  4H bias   : BOS terakhir (close menembus swing pivot-5 terkonfirmasi) + searah SMA50 4H (harga di sisi benar, SMA50 miring 3 candle) + WIDE>=1,15 (hanya konfigurasi A)
  1H setup  : FVG 1H (impuls badan>=50% range, candle sebelumnya berlawanan = OB) searah bias, disentuh pertama kali <=72 candle (tidak close menembus
              dasar OB) + sapuan (PDL/PDH atau swing 1H pivot-5, wick tembus lalu close kembali) + batas lebar pita/WIDE/%B sesuai konfigurasi
  15M       : CHoCH = close pertama menembus swing 15M (pivot-3) yang terbentuk setelah sapuan, dalam 12 jam, searah SMA50 15M
  5M trigger: OB 5M = candle berlawanan terakhir sebelum close 5M pertama yang menembus level CHoCH; limit di open OB, berlaku 3 jam; %B 5M di candle OB tidak telat
Keluar: SL = titik terjauh sejak sapuan, TP = 3R, maks 48 jam (tutup di close). SL dicek dulu kalau SL dan TP satu candle. Batas SL sesuai konfigurasi.
Satu posisi sekaligus. Fee K.FEE per trade. Sell = cermin dari buy (harga dinegatifkan).
"""
import numpy as np
import pandas as pd

import backtest as bt
import klasik as K

WIDE4, MAX_PCTB = 1.15, 0.70
PIV4, PIV1, PIV15 = 5, 5, 3
RR, MAXH5 = 3.0, 576
POI_AGE, ARM15, FILL5, OB_LOOK, WARM = 72, 48, 36, 12, 200
CFG = {  # w1 = WIDE 1H, hlo/hhi = batas setengah pita 1H, rlo/rhi = batas SL, sw = umur sapuan (candle 1H), bw = bias 4H wajib WIDE
    "A sekarang": dict(bw=True, w1=1.15, hlo=0.012, hhi=None, rlo=0.004, rhi=None, sw=24),
    "B tanpa wide di bias 4H": dict(bw=False, w1=1.15, hlo=0.012, hhi=None, rlo=0.004, rhi=None, sw=24),
    "C B + BB 1H longgar": dict(bw=False, w1=0.80, hlo=0.004, hhi=0.015, rlo=0.004, rhi=None, sw=24),
    "D C + SL 0,7-1,5% + sapuan 12": dict(bw=False, w1=0.80, hlo=0.004, hhi=0.015, rlo=0.007, rhi=0.015, sw=12),
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
    """Pivot terakhir yang SUDAH terkonfirmasi di tiap indeks: (harga, indeks candle pivot)."""
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
        v = dict(o=o, h=h, l=l, c=c, pb=I["pctb"], tr=I["up"])
    else:
        v = dict(o=-o, h=-l, l=-h, c=-c, pb=1 - I["pctb"], tr=I["dn"])
    return v


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
    v["ls"] = pd.Series(np.where(sweep, np.arange(n), np.nan)).ffill().fillna(-10 ** 9).to_numpy().astype(int)
    tap = np.zeros(n, bool)
    for k in range(2, n):
        a, b = k - 2, k - 1
        if l[k] > h[a] and c[b] > o[b] and (c[b] - o[b]) >= 0.5 * (h[b] - l[b]) and c[a] < o[a]:
            hit = np.flatnonzero(l[k + 1:k + 1 + POI_AGE] <= l[k])
            if len(hit) and c[k + 1 + hit[0]] >= l[a]:
                tap[k + 1 + hit[0]] = True
    v["tap"] = tap


def trade5(V5, ts5, L, t15, T, sweep_t, cf):
    """Dari CHoCH 15M sampai keluar. Hasil: (alasan_gugur, trade, busy_ns)."""
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
    end, tp = min(f + MAXH5, len(ts5) - 1), entry + RR * (entry - sl)
    sl_h, tp_h = np.flatnonzero(V5["l"][f:end + 1] <= sl), np.flatnonzero(V5["h"][f + 1:end + 1] >= tp) + 1
    i_sl, i_tp = (sl_h[0] if len(sl_h) else 10 ** 9), (tp_h[0] if len(tp_h) else 10 ** 9)
    if i_sl <= i_tp and i_sl < 10 ** 9:
        r, j, how = -risk, f + i_sl, "s"
    elif i_tp < 10 ** 9:
        r, j, how = RR * risk, f + i_tp, "p"
    else:
        r, j, how = (V5["c"][end] - entry) / abs(entry), end, "t"
    mfe = (V5["h"][f:j + 1].max() - entry) / abs(entry)
    return "ok", (f, j, r, mfe, how, risk), ts5[j] + 5 * 60 * 10 ** 9


def scan(cf, X):
    ts, V1, V15, V5, I = X["ts"], X["V1"], X["V15"], X["V5"], X["I"]
    b4 = np.where((X["b"] == 1) & I["4h"]["up"] & (X["w4"] | (not cf["bw"])), 1,
                  np.where((X["b"] == -1) & I["4h"]["dn"] & (X["w4"] | (not cf["bw"])), -1, 0))
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
        if not (v["tap"][t] and 0 <= t - v["ls"][t] <= cf["sw"]):
            t += 1
            continue
        F["setup"] += 1
        if not (ok1[t] and v["pb"][t] <= MAX_PCTB):
            t += 1
            continue
        F["lebar"] += 1
        sweep_t, w = ts["1h"][v["ls"][t]], V15[s]
        u0 = np.searchsorted(ts["15m"], ts["1h"][t] + H1)
        hit = None
        for u in range(max(u0, 1), min(u0 + ARM15, n15)):
            L, pi = w["lph"][u - 1], w["lpi"][u - 1]
            if pi >= 0 and ts["15m"][pi] >= sweep_t and w["c"][u] > L and w["c"][u - 1] <= L and w["tr"][u]:
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
            "SL%": round((how == "s").mean() * 100, 1), "SL_rata%": round(risk.mean() * 100, 2), "MFE_rata%": round(mfe.mean() * 100, 2),
            "kotor_total%": round(r.sum() * 100, 1), "fee_total%": round(len(r) * K.FEE * 100, 1), "bersih_total%": round(net.sum() * 100, 1),
            "terburuk%": round(net.min() * 100, 1), "terbaik%": round(net.max() * 100, 1)}


def run(ex):
    D = {}
    for tf in ("4h", "1h", "15m", "5m"):
        D[tf] = bt.fetch_ohlcv_cached(ex, bt.PAIRS[0], tf, bt.DAYS_BACK_TF[tf])
        if D[tf] is None:
            raise SystemExit(f"data {tf} kosong/ditolak (jalankan Run Cek Data)")
    I = {tf: ind(d) for tf, d in D.items()}
    X = dict(I=I, ts={tf: ns(d) for tf, d in D.items()}, b=bias4(D["4h"]), w4=I["4h"]["ratio"] >= WIDE4, V1={}, V15={}, V5={})
    for s in (1, -1):
        X["V1"][s], X["V15"][s], X["V5"][s] = view(D["1h"], I["1h"], s), view(D["15m"], I["15m"], s), view(D["5m"], I["5m"], s)
        prep1(X["V1"][s], D["1h"], s)
        X["V15"][s]["lph"], X["V15"][s]["lpi"] = lastpiv(X["V15"][s]["h"], PIV15, True)
    names = ["1H bar dgn bias 4H (tak sedang posisi)", "setup: sapuan + sentuh POI 1H", "lolos lebar pita/WIDE/%B 1H", "CHoCH 15M + SMA50 15M",
             "OB 5M + %B + SL (batas)", "limit terisi"]
    fun, rows = {"lapisan": names}, []
    for nm, cf in CFG.items():
        F, tr = scan(cf, X)
        fun[nm[0]] = list(F.values())
        for per, mk in (("semua", lambda y: y > 0), ("2021-2024", lambda y: y <= 2024), ("2025-2026", lambda y: y >= 2025)):
            rows.append(stat(nm, per, tr, D["5m"]["ts"], mk))
    fence = "`" * 3
    bt._emit("\n### MTF, 4 konfigurasi (" + "; ".join(CFG) + "). SL=sapuan, TP=3R, maks 48 jam, satu posisi, fee 0,12% per trade. Gugur per lapisan:\n"
             + fence + "\n" + pd.DataFrame(fun).to_string(index=False) + "\n" + fence)
    bt._emit("\n### Hasil (kotor = sebelum fee; total% = jumlah hasil semua trade, tanpa leverage, tanpa bunga-berbunga)\n" + fence + "\n"
             + pd.DataFrame(rows).to_string(index=False) + "\n" + fence)


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
