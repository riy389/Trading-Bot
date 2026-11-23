#!/usr/bin/env python3
"""
UJI KOIN BARU. Aturan A LT 1h dibekukan persis derivrule1h.py (mode A, varian LT), tidak ada yang di-tune:
filter atr_pct > persentil 67 dari 30 hari sebelumnya, L = ls_acct relatif rata-rata 30 hari, T = taker_delta_24h,
entry hanya kalau L dan T searah, entry di open candle berikutnya, SL = TP = 3 x ATR(14) 1h, time-stop 24 candle.
Data: Binance perp 1h (cache datacoins.py), harga dan derivatif dari bursa yang sama.
Gate: 13 koin selain BTC = satu portofolio, trade diurutkan menurut waktu entry, N_TRIED = 23.
BTC hanya baris referensi (bukan di gate). Data setelah HOLDOUT tidak dipakai (disisihkan untuk uji akhir).
Net = gross - 0,22% per trade (dalam R), ikut dihitung di gate seperti derivrule1h.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import backtest as bt
import derivtree as dt
import gates
from datacoins import SYMS, parse_k1h, parse_m1h
from derivrule1h import COST, WIN, sim

N_TRIED, HOLDOUT, MAXB = 23, "2026-05-01", 24


def load(s, days, months):
    p = s + "USDT"
    k = dt.load_set(f"{s}_klines1h", f"{dt.B}/monthly/klines/{p}/1h/{p}-1h-{{d}}.zip", parse_k1h, months, lambda t: t.strftime("%Y-%m"))
    m = dt.load_set(f"{s}_metrics1h", f"{dt.B}/daily/metrics/{p}/{p}-metrics-{{d}}.zip", parse_m1h, days, lambda t: t.date().isoformat())
    return k, m


def build(k, m):
    k = k[k.index < HOLDOUT].dropna(subset=["open", "high", "low", "close", "volume"])
    cs, pc = k["close"].astype(float), k["close"].astype(float).shift(1)
    trp = pd.concat([k["high"] - k["low"], (k["high"] - pc).abs(), (k["low"] - pc).abs()], axis=1).max(axis=1, skipna=False) / cs
    atr = trp.rolling(14).mean()
    v = k["volume"].where(k["volume"] > 0)
    dl = 2 * k["taker_buy_volume"].where(v.notna()) - v
    tk = dl.rolling(24).sum() / v.rolling(24).sum()
    ls = np.log(m["count_long_short_ratio"]).replace([np.inf, -np.inf], np.nan).reindex(k.index)
    thr = atr.shift(1).rolling(WIN, min_periods=240).quantile(2 / 3)
    lsr = ls - ls.rolling(WIN, min_periods=240).mean()
    ok = (atr.notna() & thr.notna() & lsr.notna() & tk.notna()).to_numpy(copy=True)
    volok = (atr > thr).to_numpy() & ok
    dL, dT = np.where(lsr.to_numpy() < 0, 1, -1), np.where(tk.to_numpy() > 0, 1, -1)
    d = {x[0]: k[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close")}
    return {"o": d["o"], "h": d["h"], "l": d["l"], "c": d["c"], "ts": k.index, "ok": ok, "volok": volok,
            "sig": np.where(dL == dT, dL, 0) * volok, "rf": 3 * atr.to_numpy()}


def trades(name, d, sig):
    tr = sim(d["o"], d["h"], d["l"], d["c"], sig, d["rf"], MAXB)
    if not tr:
        return pd.DataFrame({"coin": [], "t": pd.to_datetime([]), "g": [], "r": []})
    a = np.array(tr, dtype=float)
    return pd.DataFrame({"coin": name, "t": d["ts"][a[:, 0].astype(int)], "g": a[:, 2] * (a[:, 4] / a[:, 3] - 1) / a[:, 5], "r": a[:, 5]})


def summ(d, half):
    d = d.sort_values("t", kind="stable")
    g, c = d["g"].to_numpy(), (COST / d["r"]).to_numpy()
    cum = np.r_[0.0, np.cumsum(g - c)]
    early = (d["t"] < half).to_numpy()
    return {"trade": len(g), "hit%": round((g > 0).mean() * 100, 1), "gross_R": round(g.sum(), 1), "gross_R/trade": round(g.mean(), 3),
            "net_R": round((g - c).sum(), 1), "maxDD_R": round((np.maximum.accumulate(cum) - cum).max(), 1),
            "IS_R": round(g[early].sum(), 1), "OOS_R": round(g[~early].sum(), 1)}


def main():
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    days = [(dt.START + timedelta(days=i)).isoformat() for i in range((end - dt.START).days + 1)]
    months = sorted({d[:7] for d in days})[:-1]
    D = {}
    for s in SYMS:
        D[s] = build(*load(s, days, months))
        D[s]["tr"] = trades(s, D[s], D[s]["sig"])
    core = [s for s in SYMS if s != "BTC"]
    P = pd.concat([D[s]["tr"] for s in core]).sort_values("t", kind="stable").reset_index(drop=True)
    first, last = min(D[s]["ts"][D[s]["ok"]][0] for s in core), max(D[s]["ts"][D[s]["ok"]][-1] for s in core)
    half = first + (last - first) / 2
    g, c = P["g"].to_numpy(), (COST / P["r"]).to_numpy()
    gt = gates.run_gates(g, c, N_TRIED)
    rng, R = np.random.default_rng(1), []
    for _ in range(10):
        R.append(summ(pd.concat([trades(s, D[s], rng.choice([-1, 1], size=len(D[s]["c"])) * D[s]["volok"]) for s in core]), half))
    rows = [{"varian": f"A LT {len(core)} koin", **summ(P, half), "p_IS": gt["p_IS"], "p_OOS": gt["p_OOS"], "fold": gt["fold_pos"],
             "gate": "LOLOS" if gt["LOLOS"] else "GUGUR"},
            {"varian": "ARAH ACAK (rata2 10 seed)", **{k: round(float(np.mean([r[k] for r in R])), 1) for k in R[0]}}]
    pk = []
    for s in SYMS:
        t = D[s]["tr"]
        x = t["g"].to_numpy()
        pk.append({"koin": s + (" (ref)" if s == "BTC" else ""), "trade": len(x), "hit%": round((x > 0).mean() * 100, 1), "gross_R": round(x.sum(), 1),
                   "gross_R/trade": round(x.mean(), 3), "net_R": round((x - COST / t["r"].to_numpy()).sum(), 1)})
    yr = P.assign(thn=P["t"].dt.year, net=g - c).groupby("thn").agg(trade=("g", "size"), gross_R=("g", "sum"), net_R=("net", "sum")).round(1).reset_index()
    pos = sum(r["gross_R"] > 0 for r in pk if "ref" not in r["koin"])
    f = "`" * 3
    bt._emit(f"\n### Uji koin baru, aturan A LT 1h dibekukan (data Binance, sebelum {HOLDOUT}, N_TRIED={N_TRIED}, alpha_adj={gt['alpha_adj']:.5f}). "
             f"Net = gross - 0,22% per trade. IS/OOS = paruh 1/2 waktu. Koin dengan gross > 0: {pos}/{len(core)}\n"
             + f + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + f)
    bt._emit("\n### Per koin (BTC = referensi, tidak masuk gate)\n" + f + "\n" + pd.DataFrame(pk).to_string(index=False) + "\n" + f)
    bt._emit("\n### Portofolio per tahun\n" + f + "\n" + yr.to_string(index=False) + "\n" + f)


if __name__ == "__main__":
    main()
