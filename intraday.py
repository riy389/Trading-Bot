#!/usr/bin/env python3
"""
Bitcoin intraday time-series momentum (Shen, Urquhart, Wang 2022, Financial Review), BTC 5m.
Sesi New York (DST ikut): buka 09:00, tutup 17:00. r_ONFH = open+30m / close 17:00 kemarin - 1. r_SLH = 16:00 -> 16:30.
Trade: masuk 16:30, keluar 17:00, tanpa SL/TP. A: r_ONFH saja. B: r_SLH saja (long kalau r_SLH<0).
C: long kalau ONFH>0 dan SLH<0, short kalau ONFH<=0 dan SLH>=0, selain itu diam. D: C hanya di hari volatilitas buka tinggi
(realized vol 30 menit pertama di atas tercile atas 365 hari sebelumnya). E: C dengan buka 09:30.
Utama = gross %. Net = gross - 0,22% (fee 0,12% + slippage 0,10%) per trade, info saja.
"""
import numpy as np
import pandas as pd

import backtest as bt
import eventtree2 as e2

TZ, COST = "America/New_York", 0.22


def build(d, open_min):
    ts = pd.to_datetime(d["ts"])
    ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
    px = dict(zip(ts, d["open"].astype(float)))
    days = sorted(set(ts.dt.tz_convert(TZ).dt.date))

    def P(day, mm):
        return px.get(pd.Timestamp(f"{day} {mm // 60:02d}:{mm % 60:02d}", tz=TZ).tz_convert("UTC"), np.nan)

    rows = []
    for day in days[1:]:
        prev = day - pd.Timedelta(days=1)
        c_prev, o30 = P(prev, 17 * 60), P(day, open_min + 30)
        p16, p1630, p17 = P(day, 16 * 60), P(day, 16 * 60 + 30), P(day, 17 * 60)
        op = [P(day, open_min + 5 * k) for k in range(7)]
        if not np.isfinite([c_prev, o30, p16, p1630, p17, *op]).all():
            continue
        rows.append({"date": pd.Timestamp(day), "onfh": o30 / c_prev - 1, "slh": p1630 / p16 - 1, "lh": p17 / p1630 - 1,
                     "vol": float((np.diff(np.log(op)) ** 2).sum())})
    r = pd.DataFrame(rows)
    thr = r["vol"].shift(1).rolling(365, min_periods=100).quantile(2 / 3)
    r["hv"] = (r["vol"] > thr).fillna(False)
    return r


def sig(r, kind):
    on, sl = r["onfh"].to_numpy(), r["slh"].to_numpy()
    if kind == "A":
        return np.where(on > 0, 1, -1)
    if kind == "B":
        return np.where(sl < 0, 1, -1)
    s = np.where((on > 0) & (sl < 0), 1, np.where((on <= 0) & (sl >= 0), -1, 0))
    return np.where(r["hv"].to_numpy(), s, 0) if kind == "D" else s


def stat(r, s, half):
    m = s != 0
    g = (s[m] * r["lh"].to_numpy()[m]) * 100
    early = (r["date"][m] < half).to_numpy()
    sd = g.std(ddof=1) if len(g) > 2 else np.nan
    return g, early, {"trade": len(g), "hit%": round((g > 0).mean() * 100, 1), "gross%/trade": round(g.mean(), 4),
                      "t": round(g.mean() / (sd / np.sqrt(len(g))), 2), "net%/trade": round(g.mean() - COST, 4),
                      "gross_IS": round(g[early].mean(), 4), "gross_OOS": round(g[~early].mean(), 4),
                      "gross_total%": round(g.sum(), 1)}


def run(ex):
    d = e2.load(ex, "5m")
    r9, r93 = build(d, 9 * 60), build(d, 9 * 60 + 30)
    half = r9["date"].iloc[0] + (r9["date"].iloc[-1] - r9["date"].iloc[0]) / 2
    rows, yr = [], []
    base = r9["lh"].to_numpy() * 100
    rows.append({"varian": "pembanding: selalu long 16:30-17:00", "trade": len(base), "hit%": round((base > 0).mean() * 100, 1),
                 "gross%/trade": round(base.mean(), 4)})
    for nm, r, k in [("A r_ONFH saja", r9, "A"), ("B r_SLH saja", r9, "B"), ("C kombinasi", r9, "C"),
                     ("D C + vol tinggi", r9, "D"), ("E C buka 09:30", r93, "C")]:
        s = sig(r, k)
        if (s != 0).sum() < 3:
            rows.append({"varian": nm, "trade": int((s != 0).sum())})
            continue
        g, early, st = stat(r, s, half)
        rows.append({"varian": nm, **st})
        if k in ("C", "D") and nm[0] != "E":
            y = r["date"].dt.year.to_numpy()[s != 0]
            for v in np.unique(y):
                yr.append({"varian": k, "tahun": int(v), "trade": int((y == v).sum()), "gross%/trade": round(g[y == v].mean(), 4)})
    f = "`" * 3
    bt._emit(f"\n### Intraday momentum BTC 5m, hari valid: {len(r9)}. Masuk 16:30 keluar 17:00 (New York). gross% = return per trade, net = gross - {COST}%. "
             "IS/OOS = paruh 1/2 data\n" + f + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + f)
    bt._emit("\n### C dan D per tahun\n" + f + "\n" + pd.DataFrame(yr).to_string(index=False) + "\n" + f)


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
