#!/usr/bin/env python3
"""MARKET READER TEST: apakah label pasar memprediksi gerak harga ke depan? Tanpa entry/SL/TP.
BTC 1h + konteks 4h (hanya candle sudah close). Label: TREND_UP, TREND_DOWN, RANGE, NONE.
Return ke depan: open candle i+1 -> close candle i+H, sampel tidak tumpang tindih.
Pembanding acak: label digeser melingkar (30x). Parameter dikunci sebelum lihat hasil."""
import numpy as np
import pandas as pd

import backtest as bt
import eventtree2 as e2

HS = (4, 12, 24)
WARM, N_SHIFT, FEE = 1000, 30, 0.16
ER_T, ER_R, SW = 0.30, 0.20, 5
NAMES = {1: "TREND_UP", -1: "TREND_DOWN", 2: "RANGE", 0: "NONE"}


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy(float)


def er(c, n=30):
    s = pd.Series(c)
    d = (s - s.shift(n)).abs()
    v = s.diff().abs().rolling(n).sum()
    return (d / v.replace(0, np.nan)).fillna(0).to_numpy(float)


def swings(h, l, n):
    N = len(h)
    out = np.full((4, N), np.nan)
    a1 = a2 = b1 = b2 = np.nan
    for i in range(N):
        j = i - n
        if j >= n:
            if h[j] == h[j - n:i + 1].max():
                a2, a1 = a1, h[j]
            if l[j] == l[j - n:i + 1].min():
                b2, b1 = b1, l[j]
        out[:, i] = (a1, a2, b1, b2)
    return out


def build(d1, d4):
    o, h, l, c = (d1[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(c)
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
    e50 = ema(c, 50)
    slope = np.full(n, np.nan)
    slope[24:] = (e50[24:] - e50[:-24]) / atr[24:]
    hi1, hi2, lo1, lo2 = swings(h, l, SW)
    su = (hi1 > hi2) & (lo1 > lo2)
    sd = (hi1 < hi2) & (lo1 < lo2)
    t1 = pd.to_datetime(d1["ts"], utc=True).dt.tz_localize(None).to_numpy()
    t4 = pd.to_datetime(d4["ts"], utc=True).dt.tz_localize(None).to_numpy()
    c4 = d4["close"].to_numpy(float)
    e4 = ema(c4, 200)
    j = np.searchsorted(t4 + np.timedelta64(4, "h"), t1 + np.timedelta64(1, "h"), side="right") - 1
    jj = np.clip(j, 0, None)
    bias = np.where(j >= 0, c4[jj] - e4[jj], np.nan)
    r = er(c)
    up = (r >= ER_T) & (slope > 0) & su & (bias > 0)
    dn = (r >= ER_T) & (slope < 0) & sd & (bias < 0)
    lab = np.where(up, 1, np.where(dn, -1, np.where(r < ER_R, 2, 0)))
    return lab, o, c, d1["ts"].dt.year.to_numpy()


def pick(idx, H):
    out, last = [], -H
    for i in idx:
        if i - last >= H:
            out.append(i)
            last = i
    return np.array(out, dtype=int)


def get(lab, L, a, valid, H):
    sel = pick(np.flatnonzero((lab == L) & valid), H)
    x = a[sel]
    return sel, (np.abs(x) if L == 2 else (x if L == 1 else -x))


def run(ex):
    d1, d4 = e2.load(ex, "1h"), e2.load(ex, "4h")
    lab, o, c, yr = build(d1, d4)
    n = len(c)
    half = n // 2
    rng = np.random.default_rng(1)
    shifts = rng.integers(3000, n - 3000, N_SHIFT)
    fence = "`" * 3
    share = pd.Series(lab[WARM:]).map(NAMES).value_counts(normalize=True).mul(100).round(1)
    rows = []
    for H in HS:
        a = np.full(n, np.nan)
        a[:n - H] = c[H:] / o[1:n - H + 1] - 1
        valid = np.isfinite(a)
        valid[:WARM] = False
        sb = pick(np.flatnonzero(valid), H)
        xb = np.abs(a[sb])
        yb = yr[sb]
        b_is, b_oos = xb[sb < half].mean(), xb[sb >= half].mean()
        for L in (1, -1, 2):
            sel, x = get(lab, L, a, valid, H)
            if len(x) < 30:
                continue
            m = x.mean()
            null = [get(np.roll(lab, int(s)), L, a, valid, H)[1].mean() for s in shifts]
            z = (m - np.mean(null)) / (np.std(null) + 1e-12)
            m_is, m_oos = x[sel < half].mean(), x[sel >= half].mean()
            ys = yr[sel]
            yn = yp = 0
            for y in np.unique(ys):
                k = ys == y
                if k.sum() < 10 or not (yb == y).any():
                    continue
                yn += 1
                yv = x[k].mean()
                yp += int(yv < xb[yb == y].mean()) if L == 2 else int(yv > 0)
            if L == 2:
                ok = m_is < b_is and m_oos < b_oos and z <= -2 and yp >= yn - 1
            else:
                ok = m_is > 0 and m_oos > 0 and z >= 2 and yp >= yn - 1
            rows.append({"label": NAMES[L], "H_jam": H, "n": len(x),
                         "mean%": round(m * 100, 3),
                         "pembanding%": round(xb.mean() * 100, 3) if L == 2 else "",
                         "hit%": round((x > 0).mean() * 100, 1) if L != 2 else "",
                         "IS%": round(m_is * 100, 3), "OOS%": round(m_oos * 100, 3),
                         "tahun_ok": f"{yp}/{yn}", "z": round(float(z), 2),
                         "info_net%": round(m * 100 - FEE, 3) if L != 2 else "",
                         "lulus": "YA" if ok else "tidak"})
    bt._emit("\n### Market reader: porsi label (% candle 1h)\n" + fence + "\n" + share.to_string() + "\n" + fence)
    bt._emit("\n### Market reader: return ke depan per label (TREND = searah label, RANGE = gerak absolut)\n"
             + fence + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + fence)


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
