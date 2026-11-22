#!/usr/bin/env python3
"""
TES ATURAN DERIVATIF, BTC 1h (belum strategi final). Fitur dari derivtree.feats (kausal), sinyal di close candle i, entry di open i+1.
Filter volatilitas: atr_pct > persentil 67 dari 30 hari sebelumnya. Arah:
  L  = ls_acct relatif 30 hari (di bawah rata-rata = long, di atas = short)
  T  = taker_delta_24h (>0 long, <0 short)
  LT = hanya kalau L dan T searah
Exit: SL dan TP simetris (1R = jarak SL), time-stop 24 jam (96 candle) atau 4 jam (16). Satu posisi. Kalau SL dan TP tersentuh di candle yang sama, SL dihitung dulu.
Mode P = jarak 2,7% tetap. Mode A = jarak 3 x ATR(14) 15m. Pembanding = arah acak dengan filter volatilitas yang sama (rata-rata 10 seed).
Utama gross R. Net R = gross - 0,22% per trade (dalam R), info saja.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import backtest as bt
import eventtree as et
import eventtree2 as e2
import derivtree as dt
import gates

COST, TF, WIN = 0.0022, "1h", 720
N_TRIED = 22  # naikkan tiap ada varian/ide baru, jangan reset


def sim(o, h, l, c, sig, rf, maxb):
    n, idx, out, i = len(c), np.flatnonzero(sig != 0), [], 0
    while True:
        k = np.searchsorted(idx, i)
        if k >= len(idx):
            break
        i = idx[k]
        if i >= n - 2:
            break
        d, s = sig[i], i + 1
        e, r = o[s], rf[i]
        sl, tp = e * (1 - d * r), e * (1 + d * r)
        x, last = None, min(s + maxb - 1, n - 1)
        q = last
        for q in range(s, last + 1):
            if d > 0:
                if l[q] <= sl:
                    x = sl
                    break
                if h[q] >= tp:
                    x = tp
                    break
            else:
                if h[q] >= sl:
                    x = sl
                    break
                if l[q] <= tp:
                    x = tp
                    break
        if x is None:
            x, q = c[last], last
        out.append((s, q, d, e, x, r))
        i = max(q, i + 1)
    return out


def stats(tr, ts, half):
    s, q, d, e, x, r = (np.array(z, dtype=float) for z in zip(*tr))
    g = d * (x / e - 1) / r
    net = g - COST / r
    cum = np.r_[0.0, np.cumsum(net)]
    early = (ts[s.astype(int)] < half).to_numpy()
    return s, g, net, {"trade": len(g), "hit%": round((g > 0).mean() * 100, 1), "gross_R": round(g.sum(), 1), "gross_R/trade": round(g.mean(), 3),
                       "net_R": round(net.sum(), 1), "maxDD_R": round((np.maximum.accumulate(cum) - cum).max(), 1),
                       "IS_R": round(g[early].sum(), 1), "OOS_R": round(g[~early].sum(), 1)}


def run(ex):
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    days = [(dt.START + timedelta(days=i)).isoformat() for i in range((end - dt.START).days + 1)]
    months = sorted({d[:7] for d in days})[:-1]
    kl = dt.load_set("klines5m", dt.B + "/daily/klines/BTCUSDT/5m/BTCUSDT-5m-{d}.zip", dt.parse_k, days, lambda t: t.date().isoformat())
    met = dt.load_set("metrics", dt.B + "/daily/metrics/BTCUSDT/BTCUSDT-metrics-{d}.zip", dt.parse_m, days, lambda t: t.date().isoformat())
    fund = dt.load_set("funding", dt.B + "/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{d}.zip", dt.parse_f, months, lambda t: t.strftime("%Y-%m"))
    df = e2.load(ex, TF)
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    F = dt.feats(df, TF, kl, met, fund)
    atr = F["atr_pct"]
    thr = atr.shift(1).rolling(WIN, min_periods=240).quantile(2 / 3)
    lsr = F["ls_acct"] - F["ls_acct"].rolling(WIN, min_periods=240).mean()
    tk = F["taker_delta_24h"]
    ok = (atr.notna() & thr.notna() & lsr.notna() & tk.notna()).to_numpy(copy=True)
    ok[:et.WARM] = False
    volok = (atr > thr).to_numpy() & ok
    dL, dT = np.where(lsr.to_numpy() < 0, 1, -1), np.where(tk.to_numpy() > 0, 1, -1)
    dirs = {"L": dL, "T": dT, "LT": np.where(dL == dT, dL, 0)}
    ts = df["ts"]
    half = ts[ok].iloc[0] + (ts[ok].iloc[-1] - ts[ok].iloc[0]) / 2
    rfs = {"P": np.full(len(c), 0.027), "A": 3 * atr.to_numpy()}
    rows, yrs, rng = [], [], np.random.default_rng(1)
    for mode in ("P", "A"):
        cfgs = [(f"{mode} {d}", d, True, 24) for d in ("L", "T", "LT")]
        cfgs += [(f"{mode} LT tanpa filter vol", "LT", False, 24), (f"{mode} LT time-stop 4h", "LT", True, 4)]
        for nm, d, vf, mb in cfgs:
            sig = dirs[d] * ((volok if vf else ok).astype(int))
            tr = sim(o, h, l, c, sig, rfs[mode], mb)
            s, g, net, st = stats(tr, ts, half)
            gt = gates.run_gates(g, g - net, N_TRIED)
            rows.append({"varian": nm, **st, "p_IS": gt["p_IS"], "p_OOS": gt["p_OOS"], "gate": "LOLOS" if gt["LOLOS"] else "GUGUR"})
            if nm in (f"{mode} LT",):
                y = ts.dt.year.to_numpy()[s.astype(int)]
                for v in np.unique(y):
                    yrs.append({"varian": nm, "tahun": int(v), "trade": int((y == v).sum()), "gross_R": round(g[y == v].sum(), 1), "net_R": round(net[y == v].sum(), 1)})
        res = []
        for _ in range(10):
            sig = rng.choice([-1, 1], size=len(c)) * volok.astype(int)
            res.append(stats(sim(o, h, l, c, sig, rfs[mode], 24), ts, half)[3])
        rows.append({"varian": f"{mode} ARAH ACAK (rata2 10 seed)", **{k: round(float(np.mean([r[k] for r in res])), 1) for k in res[0]}})
    f = "`" * 3
    bt._emit(f"\n### Tes aturan derivatif BTC 1h. Entry open candle berikutnya, SL=TP simetris, 1R = jarak SL. Mode P = 2,7% tetap, A = 3 x ATR. "
             f"Time-stop 24 candle (24 jam) kecuali dicatat. IS/OOS = paruh 1/2 waktu. Net = gross - 0,22% per trade. Baris valid: {int(ok.sum())}\n"
             + f + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + f)
    bt._emit("\n### LT per tahun\n" + f + "\n" + pd.DataFrame(yrs).to_string(index=False) + "\n" + f)


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
