#!/usr/bin/env python3
"""
TRADE SIM 15m: aturan event tree + filter arah, entry di open candle t (sinyal di close t-1), satu posisi sekali.
SL = di balik swing terakhir (+0,3 ATR), maks 1,8%; swing tidak valid atau lebih jauh = trade dilewati.
A: TP tetap 2,7%.  B: TP = 2R.  Batas tahan 16 candle (keluar di close). Kena SL dan TP di candle yang sama dihitung SL.
Pembanding: entry acak (candle dan arah acak, jumlah sinyal sama, SL/TP sama). Fee hanya info (R net).
"""
import numpy as np
import pandas as pd

import backtest as bt
import eventtree as et
import eventtree2 as e2

N_HOLD, SL_MAX, SL_BUF, TP_PCT, BIG, FEE, RR_B, NRAND = 16, 0.018, 0.3, 0.027, 0.053, 0.0012, 2.0, 300


def rules(X):
    a, hc, di = X["atr_pct"], X["hour_cos"], X["h_di_diff"]
    sess = (a > 0.00184) & (a <= 0.00336) & (hc <= -0.773)
    ext = (a > 0.00745) & (di <= -13.4)
    up = (sess & (X["h_macd_hist"] > -0.0514) & (X["struct"] > -1)) | (ext & (X["atr_ratio"] <= 1.43))
    dn = (((a > 0.00336) & (a <= 0.00745) & (di <= -13.4) & (X["h_ema20_slope"] > -0.796))
          | (sess & (X["h_macd_hist"] <= -0.0514) & (X["ema20_50"] > -0.244)) | (ext & (X["atr_ratio"] > 1.43)))
    return np.where(up & ~dn, 1, np.where(dn & ~up, -1, 0))


def run(ex):
    df, d1, d4 = e2.load(ex, "15m"), e2.load(ex, "1h"), e2.load(ex, "4h")
    F = e2.ctx(et.features(df), df, et.features(d1), d1, "1h", "h1_")
    F = e2.ctx(F, df, et.features(d4), d4, "4h", "h_")
    X = et.build_X(df, F)
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    n = len(o)
    atr = bt._rma_atr(h, l, c, 14)
    ps = et.pivot_state(h, l, bt._rsi(c, 14), et.PIV_N)
    prev = lambda a: np.r_[np.nan, a[:-1]]
    stop_b, stop_s = prev(ps["sl"]) - SL_BUF * prev(atr), prev(ps["sh"]) + SL_BUF * prev(atr)
    fin = X.notna().all(axis=1).to_numpy()
    fin[:et.WARM], fin[n - N_HOLD:] = False, False
    side = rules(X)
    side[~fin] = 0
    h1, di = X["h1_ema20_dist"].to_numpy(), X["di_diff"].to_numpy()
    okd = np.where((h1 > 0) & (di > 0), 1, np.where((h1 < 0) & (di < 0), -1, 0))
    modes = {"aturan": side, "aturan+arah": np.where(side == okd, side, 0)}

    def one(t, s, var):
        e = o[t]
        stop = stop_b[t] if s > 0 else stop_s[t]
        d = s * (e - stop) / e
        if not (0 < d <= SL_MAX):
            return None
        tpd = TP_PCT if var == "A" else RR_B * d
        tp = e * (1 + s * tpd)
        mfe = 0.0
        for k in range(t, t + N_HOLD):
            mfe = max(mfe, s * ((h[k] if s > 0 else l[k]) / e - 1))
            if (l[k] <= stop) if s > 0 else (h[k] >= stop):
                return k, -1.0, d, mfe, "SL"
            if (h[k] >= tp) if s > 0 else (l[k] <= tp):
                return k, tpd / d, d, mfe, "TP"
        return k, s * (c[k] / e - 1) / d, d, mfe, "T"

    def trades(ts, ss, var):
        last, out = -1, []
        for t, s in zip(ts, ss):
            if t <= last:
                continue
            r = one(t, s, var)
            if r:
                last = r[0]
                out.append((t,) + r[1:])
        return out

    rng = np.random.default_rng(1)
    pool = np.where(fin)[0]

    def rand(nsig, var):
        m = []
        for _ in range(NRAND):
            tr = trades(np.sort(rng.choice(pool, nsig, replace=False)), rng.choice([-1, 1], nsig), var)
            if tr:
                m.append(np.mean([x[1] for x in tr]))
        return float(np.mean(m)), float(np.std(m))

    days = (df["ts"].iloc[-1] - df["ts"].iloc[0]).days
    rows, yrs = [], []
    for mn, sd in modes.items():
        ts = np.where(sd != 0)[0]
        for var in ("A", "B"):
            tr = pd.DataFrame(trades(ts, sd[ts], var), columns=["t", "R", "d", "mfe", "why"])
            if tr.empty:
                continue
            rm, rs = rand(len(ts), var)
            rows.append({"aturan": mn, "TP": var, "sinyal": len(ts), "trade": len(tr), "per_hari": round(len(tr) / days, 2),
                         "TP%": round((tr.why == "TP").mean() * 100, 1), "SL%": round((tr.why == "SL").mean() * 100, 1),
                         "waktu%": round((tr.why == "T").mean() * 100, 1), "R_gross": round(tr.R.mean(), 3),
                         "R_net": round((tr.R - FEE / tr.d).mean(), 3), "R_acak": round(rm, 3), "z": round((tr.R.mean() - rm) / rs, 1),
                         "SL_med%": round(tr.d.median() * 100, 2), "mfe_med%": round(tr.mfe.median() * 100, 2),
                         "mfe>=5,3%": round((tr.mfe >= BIG).mean() * 100, 1)})
            tr["tahun"] = df["ts"].iloc[tr.t.to_numpy()].dt.year.to_numpy()
            for y, g in tr.groupby("tahun"):
                yrs.append({"aturan": mn, "TP": var, "tahun": int(y), "trade": len(g), "TP%": round((g.why == "TP").mean() * 100, 1),
                            "R_gross": round(g.R.mean(), 3), "R_net": round((g.R - FEE / g.d).mean(), 3)})
    fence = "`" * 3
    bt._emit(f"\n### Trade sim 15m ({days} hari). SL struktur maks {SL_MAX * 100:.1f}%, A = TP {TP_PCT * 100:.1f}%, B = TP {RR_B:g}R, tahan {N_HOLD} candle. "
             f"R = hasil dibagi jarak SL. R_acak = rata-rata entry acak (jumlah sinyal sama), z = selisih dalam simpangan baku acak. "
             f"R_net memakai fee {FEE * 100:.2f}% pulang-pergi (info). mfe = gerak terjauh searah selama ditahan\n"
             + fence + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + fence)
    bt._emit("\n### Per tahun\n" + fence + "\n" + pd.DataFrame(yrs).to_string(index=False) + "\n" + fence)


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
