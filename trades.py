#!/usr/bin/env python3
"""
TRADE TERPISAH: aturan (maks 3 indikator, dicari di paruh 1 seperti strategi.py, target pencarian 2,7%) diuji sebagai trade sungguhan:
entry di open candle sinyal, satu posisi sekali (satu gerakan = satu trade), SL = TP = M% simetris, habis waktu keluar di close candle terakhir.
M diuji dalam rentang 1,5% sampai 5,4% (2.000 USD di BTC 74.000 = 2,7%; 4.000 USD = 5,4%). Fee 0,12% pulang-pergi dikurangkan dari tiap trade.
Kena SL dan TP di candle yang sama dihitung kalah. IS = paruh 1 (tempat aturan dicari), OOS = paruh 2 (dibekukan).
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

import backtest as bt
import eventtree as et
import eventtree2 as e2
import strategi as st

MS, FEE, DEPTH, M_SEARCH = (0.015, 0.02, 0.027, 0.04, 0.054), 0.0012, 3, 0.027


def outcome(o, h, l, c, N, s, M):
    n = len(o)
    r = n - N + 1
    up = swv(h, N) >= o[:r, None] * (1 + M)
    dn = swv(l, N) <= o[:r, None] * (1 - M)
    tp, sl = (up, dn) if s > 0 else (dn, up)
    kt = np.where(tp.any(1), tp.argmax(1), N)
    ks = np.where(sl.any(1), sl.argmax(1), N)
    win, loss = kt < ks, (ks <= kt) & (ks < N)
    off = np.where(win | loss, np.minimum(kt, ks), N - 1)
    ret = np.where(win, M, np.where(loss, -M, s * (c[N - 1:] / o[:r] - 1)))
    pad = lambda a, v: np.r_[a, np.full(n - r, v)]
    return pad(win, False), pad(loss, False), pad(off, 0), pad(ret, 0.0)


def trades(mask, win, loss, off, ret):
    last, out = -1, []
    for t in np.where(mask)[0]:
        if t <= last:
            continue
        last = t + off[t]
        out.append((win[t], loss[t], ret[t]))
    return out


def summ(tr):
    if not tr:
        return 0, np.nan, np.nan, np.nan, np.nan
    w, l, r = (np.array(z) for z in zip(*tr))
    net = r - FEE
    return len(tr), (w.sum() / (w.sum() + l.sum()) * 100 if w.sum() + l.sum() else np.nan), (1 - (w.sum() + l.sum()) / len(tr)) * 100, net.mean() * 100, net.sum() * 100


def run(ex):
    d1, d4 = e2.load(ex, "1h"), e2.load(ex, "4h")
    F1, F4 = et.features(d1), et.features(d4)
    rows, rules = [], []
    st.STEPS = DEPTH
    for tf, N in (("1h", 12), ("15m", 16)):
        df = d1 if tf == "1h" else e2.load(ex, tf)
        F = et.features(df)
        if tf == "15m":
            F = e2.ctx(F, df, F1, d1, "1h", "h1_")
            F = e2.ctx(F, df, F4, d4, "4h", "h_")
        else:
            F = et.add_ctx(F, df, F4, d4, "1h", "4h")
        X = et.build_X(df, F)
        names, Xv = list(X.columns), X.to_numpy(dtype=float)
        o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        n = len(o)
        valid = np.isfinite(Xv).all(axis=1)
        valid[:et.WARM], valid[n - N + 1:] = False, False
        half = df["ts"].iloc[0] + (df["ts"].iloc[-1] - df["ts"].iloc[0]) / 2
        early = (df["ts"] < half).to_numpy()
        months = max((df["ts"].iloc[-1] - half).days / 30, 1)
        for lab, nm in ((-1, "TURUN"), (1, "NAIK")):
            w0, l0, _, _ = outcome(o, h, l, c, N, lab, M_SEARCH)
            _, conds = st.search(Xv, names, w0, l0, valid & early, valid & ~early, tf, nm)
            rules.append(f"- {tf} {nm}: " + " DAN ".join(f"{names[j]} {op} {thr:.4g}" for j, op, thr in conds))
            m_is, m_oos = st.apply(valid & early, conds, Xv), st.apply(valid & ~early, conds, Xv)
            for M in MS:
                win, loss, off, ret = outcome(o, h, l, c, N, lab, M)
                a, b = summ(trades(m_is, win, loss, off, ret)), summ(trades(m_oos, win, loss, off, ret))
                rows.append({"TF": tf, "arah": nm, "M%": round(M * 100, 1), "fee_dari_M%": round(FEE / M * 100, 1), "impas_hit%": round((0.5 + FEE / (2 * M)) * 100, 1),
                             "IS_trade": a[0], "IS_hit%": round(a[1], 1), "IS_bersih%/trade": round(a[3], 3),
                             "OOS_trade": b[0], "OOS_per_bulan": round(b[0] / months, 1), "OOS_hit%": round(b[1], 1), "OOS_habis%": round(b[2], 1),
                             "OOS_bersih%/trade": round(b[3], 3), "OOS_total_bersih%": round(b[4], 1)})
        print(f"  {tf}: selesai")
    fence = "`" * 3
    bt._emit(f"\n### Trade terpisah (satu posisi sekali). M = jarak SL = TP. Referensi: 2.000 USD = 2,5% (BTC 80.000), 2,7% (74.000), 2,9% (70.000); 4.000 USD = 5,0% sampai 5,7%. "
             f"Fee {FEE * 100:.2f}% pulang-pergi sudah dikurangkan di kolom bersih. impas_hit% = hit rate minimal agar tidak rugi setelah fee. "
             "hit% = menang / (menang + kalah), habis% = trade yang tidak sampai TP atau SL dalam batas waktu. IS = paruh 1, OOS = paruh 2 (dibekukan)\n"
             + fence + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + fence + "\nAturan (maks 3 indikator, dicari di paruh 1):\n" + "\n".join(rules))


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
