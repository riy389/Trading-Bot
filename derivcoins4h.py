#!/usr/bin/env python3
"""
UJI 4h. Aturan A LT PERSIS derivcoins.py (semua jumlah candle sama: ATR 14, taker 24, WIN 720, time-stop 24), hanya candle 4h
hasil resample dari cache 1h datacoins.py (tanpa unduh baru). Candle 4h hanya dipakai kalau lengkap 4 jam. Percobaan ke-24.
Gate: 13 koin selain BTC = satu portofolio, N_TRIED = 24. BTC hanya referensi. Holdout sama (setelah 2026-04-30 tidak dipakai).
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import backtest as bt
import derivtree as dt
import gates
from datacoins import SYMS
from derivcoins import HOLDOUT, build, load, summ, trades
from derivrule1h import COST

N_TRIED = 24
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "taker_buy_volume": "sum"}


def to4h(k, m):
    r = k.resample("4h")
    k4 = r.agg(AGG)[r["close"].count() == 4]
    return k4, m.resample("4h").last()


def main():
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    days = [(dt.START + timedelta(days=i)).isoformat() for i in range((end - dt.START).days + 1)]
    months = sorted({d[:7] for d in days})[:-1]
    D = {}
    for s in SYMS:
        D[s] = build(*to4h(*load(s, days, months)))
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
    rows = [{"varian": f"A LT 4h {len(core)} koin", **summ(P, half), "p_IS": gt["p_IS"], "p_OOS": gt["p_OOS"], "fold": gt["fold_pos"],
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
    bt._emit(f"\n### Uji 4h, aturan A LT sama persis (data Binance, sebelum {HOLDOUT}, N_TRIED={N_TRIED}, alpha_adj={gt['alpha_adj']:.5f}). "
             f"Net = gross - 0,22% per trade. IS/OOS = paruh 1/2 waktu. Koin dengan gross > 0: {pos}/{len(core)}\n"
             + f + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + f)
    bt._emit("\n### Per koin 4h (BTC = referensi, tidak masuk gate)\n" + f + "\n" + pd.DataFrame(pk).to_string(index=False) + "\n" + f)
    bt._emit("\n### Portofolio 4h per tahun\n" + f + "\n" + yr.to_string(index=False) + "\n" + f)


if __name__ == "__main__":
    main()
