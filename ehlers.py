#!/usr/bin/env python3
"""
UJI EHLERS COMBO (simwai, Pine v5) + GATE. Aturan beku, tidak di-tune:
Parameter default Pine: Length 20, Rms 50, SNR thr 0,1, Exit 100, adaptif MATI. Candle 1h dan 4h dari cache 1h Binance.
Varian: EXACT = port persis Pine (termasuk bug rms=math.avg(x,50) dan rumus SNR); FIX = rms rolling SMA, SNR = 10*log10(.)+6.
Entry 6 syarat Pine, eksekusi open candle berikutnya. Exit Pine = crossunder/over(src[100], iT) -> open berikutnya.
Sinyal berlawanan = balik arah (strategy.entry). TAMBAHAN: SL = 2 x ATR14 candle sinyal (R = jarak SL), cek intrabar, gap fill di open.
Biaya 0,22% per trade (dalam R) + funding nyata. 13 koin = satu portofolio, BTC referensi. Data setelah 2026-04-30 tidak dipakai.
N_TRIED = 25 + 4 varian = 29 untuk semua gate.
LULUS (dikunci sebelum lihat hasil): gate LOLOS, PF net >= 1,2 dan net/trade > 0 di IS dan OOS, >= 30 trade per paruh.
"""
import math

import numpy as np
import pandas as pd

N_TRIED, HOLDOUT, WARM = 29, "2026-05-01", 200
L, RMSL, THR, EXL = 20, 50, 0.1, 100


def snr_norm(h, l, p, fix):
    n, ln10 = len(p), math.log(10)
    Rg, Sm, De, Q1, I1, I2, Q2, Re, Im, Pe = ([0.0] * n for _ in range(10))
    SN, ns, hmin, hmax = [math.nan] * n, np.full(n, np.nan), 1.0, -1.0

    def z(a, i, k):
        return a[i - k] if i >= k else 0.0

    for i in range(n):
        Rg[i] = .1 * (h[i] - l[i]) + .9 * z(Rg, i, 1)
        pr = z(Pe, i, 1)
        pp = .075 * pr + .54
        if i > 5:
            Sm[i] = (4 * p[i] + 3 * z(p, i, 1) + 2 * z(p, i, 2) + z(p, i, 3)) / 10
            De[i] = (.0962 * Sm[i] + .5769 * z(Sm, i, 2) - .5769 * z(Sm, i, 4) - .0962 * z(Sm, i, 6)) * pp
            Q1[i] = (.0962 * De[i] + .5769 * z(De, i, 2) - .5769 * z(De, i, 4) - .0962 * z(De, i, 6)) * pp
            I1[i] = z(De, i, 3)
        jI = (.0962 * I1[i] + .5769 * z(I1, i, 2) - .5769 * z(I1, i, 4) - .0962 * z(I1, i, 6)) * pp
        jQ = (.0962 * Q1[i] + .5769 * z(Q1, i, 2) - .5769 * z(Q1, i, 4) - .0962 * z(Q1, i, 6)) * pp
        I2[i] = .2 * (I1[i] - jQ) + .8 * z(I2, i, 1)
        Q2[i] = .2 * (Q1[i] + jI) + .8 * z(Q2, i, 1)
        Re[i] = .2 * (I2[i] * z(I2, i, 1) + Q2[i] * z(Q2, i, 1)) + .8 * z(Re, i, 1)
        Im[i] = .2 * (I2[i] * z(Q2, i, 1) - Q2[i] * z(I2, i, 1)) + .8 * z(Im, i, 1)
        per = 0.0
        if Im[i] != 0 and Re[i] != 0:
            at = math.atan(Im[i] / Re[i])
            per = 2 * math.pi / at if at != 0 else 0.0
        per = min(per, 1.5 * pr) if per > 1.5 * pr else per
        per = .67 * pr if per < .67 * pr else per
        per = min(max(per, 6.0), 100.0)
        Pe[i] = .2 * per + .8 * pr
        x = (I1[i] ** 2 + Q1[i] ** 2) / (Rg[i] ** 2) if Rg[i] != 0 else math.nan
        prev = SN[i - 1] if i > 0 and not math.isnan(SN[i - 1]) else 0.0
        if fix:
            SN[i] = .25 * (10 * math.log10(x) + 6) + .75 * prev if x == x and x > 0 else math.nan
        else:
            SN[i] = .25 * (10 * math.log(x / ln10 + 6) + .75 * prev) if x == x else math.nan
        if SN[i] == SN[i]:
            hmin, hmax = min(SN[i], hmin), max(SN[i], hmax)
            ns[i] = 2 * (SN[i] - hmin) / max(hmax - hmin, 1)
    return ns


def elegant(c, fix):
    n = len(c)
    a1 = math.exp(-1.414 * math.pi / L)
    c2, c3 = 2 * a1 * math.cos(1.414 * math.pi / L), -a1 * a1
    c1 = 1 - c2 - c3
    d = c - np.r_[0.0, 0.0, c[:-2]]
    if fix:
        rms = np.sqrt(pd.Series(d ** 2).rolling(RMSL).mean().to_numpy())
    else:
        rms = np.sqrt((d ** 2 + RMSL) / 2)
    nd = np.where(np.nan_to_num(rms) != 0, d / np.where(rms == 0, 1, rms), 0.0)
    fi = np.tanh(np.nan_to_num(nd))
    ss = np.zeros(n)
    for i in range(3, n):
        ss[i] = c1 * (fi[i] + fi[i - 1]) / 2 + c2 * ss[i - 1] + c3 * ss[i - 2]
    w = np.arange(L, 0, -1)
    sg = np.convolve(ss, w)[:n] / w.sum()
    sg[:L - 1] = np.nan
    slo = ss - sg
    pv = np.nan_to_num(np.r_[0.0, slo[:-1]])
    return np.where(slo > 0, np.where(slo > pv, 2, 1), np.where(slo < 0, np.where(slo < pv, -2, -1), 0))


def decycler(c):
    n, t = len(c), 2 * math.pi / L
    al = (math.cos(t) + math.sin(t) - 1) / math.cos(t)
    dec = np.zeros(n)
    for i in range(n):
        dec[i] = al / 2 * (c[i] + (c[i - 1] if i else 0.0)) + (1 - al) * (dec[i - 1] if i else 0.0)
    return dec


def itrend(c):
    n, al = len(c), 2 / (L + 1) / 2
    it = np.full(n, np.nan)
    z = lambda i: 0.0 if i < 0 or it[i] != it[i] else it[i]
    for i in range(n):
        if i < 7:
            it[i] = (c[i] + 2 * c[i - 1] + c[i - 2]) / 4 if i >= 2 else np.nan
        else:
            it[i] = ((al - al * al / 4) * c[i] + .5 * al * al * c[i - 1] - (al - .75 * al * al) * c[i - 2]
                     + 2 * (1 - al) * z(i - 1) - (1 - al) ** 2 * z(i - 2))
    return it


def spearman(c):
    n, sig = len(c), np.zeros(len(c))
    for i in range(n):
        a, rk = [0.0] * (L + 2), [0.0] * (L + 2)
        for k in range(1, L + 1):
            a[k] = c[i - k + 1] if i - k + 1 >= 0 else 0.0
            rk[k] = k
        for ii in range(1, L + 1):
            hi = ii - 1  # Pine: "for j = 1 to 0" menghitung mundur 1, 0
            for j in (range(1, hi + 1) if hi >= 1 else (1, 0)):
                if a[j + 1] < a[j]:
                    a[j], a[j + 1], rk[j], rk[j + 1] = a[j + 1], a[j], rk[j + 1], rk[j]
        s = sum((k - rk[k]) ** 2 for k in range(1, L + 1))
        sig[i] = 2 * (0.5 - (1 - 6 * s / (L * (L * L - 1))))
    slo = sig - np.r_[0.0, sig[:-1]]
    pv = np.r_[0.0, slo[:-1]]
    up, dn = (slo > 0) | (sig > 0), (slo < 0) | (sig < 0)
    return np.where(up, np.where(slo > pv, 2, 1), np.where(dn, np.where(slo < pv, -2, -1), 0))


def shift(a, k, f=np.nan):
    return np.r_[np.full(k, f), a[:-k]] if k else a


def atr14(h, l, c):
    n, pc = len(c), np.r_[np.nan, c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    tr[0] = h[0] - l[0]
    a = np.full(n, np.nan)
    a[13] = tr[:14].mean()
    for i in range(14, n):
        a[i] = (a[i - 1] * 13 + tr[i]) / 14
    return a


def signals(o, h, l, c, fix, sp):
    ns, eo, dec, it = snr_norm(h, l, c, fix), elegant(c, fix), decycler(c), itrend(c)
    ds = np.where(c > dec, 1, np.where(c < dec, -1, 0))
    cu = lambda a, b: (a > b) & (shift(a, 1) <= shift(b, 1))
    cd = lambda a, b: (a < b) & (shift(a, 1) >= shift(b, 1))
    nz = np.nan_to_num(ns, nan=-9)
    zero = np.zeros(len(c))
    S = {"long": [cu(eo, zero), cu(ds, zero), cu(c, dec), c > it, shift(it, 1) < it, sp > 0, nz > THR],
         "short": [cd(eo, zero), cd(ds, zero), cd(c, dec), c < it, shift(it, 1) > it, sp < 0, nz > THR]}
    ok = {}
    w = np.arange(len(c)) >= WARM
    for side, cs in S.items():
        m = w.copy()
        for k, x in enumerate(cs):
            m = m & x
            ok.setdefault(k, np.zeros(len(c), bool))
            ok[k] = ok[k] | m
        S[side] = m
    fun = [int(ok[k].sum()) for k in range(7)]
    ent = np.where(S["long"], 1, np.where(S["short"], -1, 0))
    a = shift(c, EXL)
    exl, exs = cd(a, it), cu(a, it)
    return ent, exl, exs, fun


def sim(o, h, l, c, atr, ent, exl, exs, ts, H):
    n, out, pos, e, r, s, pend = len(c), [], 0, 0.0, 0.0, 0, None
    for q in range(WARM, n):
        if pend:
            act, d, rr = pend
            pend = None
            if pos and act != "enter":
                out.append((ts[s], ts[q], pos, e, o[q], r))
                pos = 0
            if act != "exit":
                pos, e, r, s = d, o[q], rr, q
        if pos:
            sl = e - pos * r
            if (pos > 0 and o[q] <= sl) or (pos < 0 and o[q] >= sl):
                out.append((ts[s], ts[q], pos, e, o[q], r))
                pos = 0
            elif (pos > 0 and l[q] <= sl) or (pos < 0 and h[q] >= sl):
                out.append((ts[s], ts[q] + H, pos, e, sl, r))
                pos = 0
        if q == n - 1:
            break
        if pos and ((pos > 0 and exl[q]) or (pos < 0 and exs[q])):
            pend = ("exit", 0, 0.0)
        d = int(ent[q])
        if d and d != pos and atr[q] == atr[q]:
            pend = ("rev" if pos else "enter", d, 2 * atr[q])
    if pos:
        out.append((ts[s], ts[n - 1] + H, pos, e, c[n - 1], r))
    return out


def trades(name, k, f, fix, sp, hrs, cost):
    o, h, l, c = (k[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    ts = k.index.to_numpy().astype("datetime64[ns]")
    ent, exl, exs, fun = signals(o, h, l, c, fix, sp)
    out = sim(o, h, l, c, atr14(h, l, c), ent, exl, exs, ts, np.timedelta64(hrs, "h"))
    if not out:
        return pd.DataFrame(columns=["coin", "t", "g", "fee", "fund"]), fun
    t0, tx, d, e, x, r = (np.array(v) for v in zip(*out))
    ft, cs = f.index.to_numpy().astype("datetime64[ns]"), np.r_[0.0, np.cumsum(f["rate"].fillna(0).to_numpy())]
    fs = cs[np.searchsorted(ft, tx.astype("datetime64[ns]"))] - cs[np.searchsorted(ft, t0.astype("datetime64[ns]"))]
    spd = r.astype(float) / e.astype(float)
    return pd.DataFrame({"coin": name, "t": t0, "g": d * (x - e) / r, "fee": cost / spd, "fund": d * fs / spd}), fun


def main():
    from datetime import datetime, timedelta, timezone

    import backtest as bt
    import derivtree as dt
    import gates
    from datacoins import SYMS
    from derivrule1h import COST
    from utswing4h import AGG, load, pf, summ

    def to_tf(k, hrs):
        k = k[k.index < HOLDOUT]
        if hrs == 1:
            return k[["open", "high", "low", "close"]]
        r = k.resample(f"{hrs}h")
        return r.agg(AGG)[r["close"].count() == hrs].dropna()

    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    months = sorted({(dt.START + timedelta(days=i)).isoformat()[:7] for i in range((end - dt.START).days + 1)})[:-1]
    K = {s: load(s, months) for s in SYMS}
    core, f3, rows, funn, pk = [s for s in SYMS if s != "BTC"], "`" * 3, [], [], {}
    for hrs in (1, 4):
        D = {s: to_tf(K[s][0], hrs) for s in SYMS}
        SP = {s: spearman(D[s]["close"].to_numpy(dtype=float)) for s in SYMS}
        for fix in (False, True):
            tag = f"{hrs}h {'FIX' if fix else 'EXACT'}"
            R = {}
            for s in SYMS:
                R[s], fun = trades(s, D[s], K[s][1], fix, SP[s], hrs, COST)
                if s != "BTC":
                    funn.append((tag, fun))
            T = pd.concat([R[s] for s in core]).sort_values("t", kind="stable").reset_index(drop=True)
            pk[tag] = {s: round(float((R[s]["g"] - R[s]["fee"] - R[s]["fund"]).mean()), 3) if len(R[s]) else 0.0 for s in SYMS}
            if len(T) == 0:
                rows.append({"set": tag, "trade": 0})
                continue
            half = T["t"].iloc[0] + (T["t"].iloc[-1] - T["t"].iloc[0]) / 2
            m = summ(T, half)
            if len(T) >= 60:
                gt = gates.run_gates((T["g"] - T["fund"]).to_numpy(), T["fee"].to_numpy(), N_TRIED)
                m.update(p_IS=gt["p_IS"], p_OOS=gt["p_OOS"], fold=gt["fold_pos"], gate="LOLOS" if gt["LOLOS"] else "GUGUR")
                ok = (gt["LOLOS"] and m["PF_IS"] >= 1.2 and m["PF_OOS"] >= 1.2 and m["net/tr IS"] > 0 and m["net/tr OOS"] > 0
                      and min(m["n_IS"], m["n_OOS"]) >= 30)
            else:
                m["gate"], ok = "trade<60", False
            rows.append({"set": tag, **m, "VONIS": "LULUS" if ok else "GUGUR"})
    fn = pd.DataFrame([{"set": t, **{f"s{k}": v for k, v in enumerate(fu)}} for t, fu in funn]).groupby("set", sort=False).sum().reset_index()
    fn.columns = ["set", "1 EO", "2 +decSig", "3 +src>dec", "4 +src>iT", "5 +iT naik", "6 +Spearman", "7 +SNR"]
    bt._emit(f"\n### Ehlers Combo, N_TRIED={N_TRIED}, biaya 0,22% + funding, SL 2xATR14 (tambahan). Corong sinyal (kumulatif, 13 koin, long+short)\n"
             + f3 + "\n" + fn.to_string(index=False) + "\n" + f3)
    bt._emit("\n### Hasil portofolio 13 koin + vonis kriteria terkunci\n" + f3 + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + f3)
    bt._emit("\n### net/trade per koin (BTC = referensi)\n" + f3 + "\n" + pd.DataFrame(pk).to_string() + "\n" + f3)


if __name__ == "__main__":
    main()
