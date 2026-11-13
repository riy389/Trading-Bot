#!/usr/bin/env python3
"""
IKUT SINYAL: posisi selalu mengikuti strategi. Dibuka di sinyal, ditutup dan dibalik HANYA saat sinyal berlawanan muncul (bukan karena candle, tanpa SL/TP).
Aturan: BB ikut (high menyentuh BB(20,2) atas = BELI, low menyentuh bawah = JUAL) dan EMA10/20 cross.
Diuji di 1h dan 4h. TF lebih lambat = trade lebih jarang dan lebih besar, supaya hasil kotor per trade bisa melewati fee 0,12%.
Sinyal dari candle yang sudah close, entry/balik di open candle berikutnya. Fee 0,12% per trade.
"""
import numpy as np
import pandas as pd

import backtest as bt
import eventtree2 as e2
import klasik as K

TFS, HOURS, RULES = ("1h", "4h"), {"1h": 1.0, "4h": 4.0}, ("BB ikut", "EMA10/20")


def run(ex):
    rows, yrs = [], []
    for tf in TFS:
        df = e2.load(ex, tf)
        o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        half = df["ts"].iloc[0] + (df["ts"].iloc[-1] - df["ts"].iloc[0]) / 2
        early = (df["ts"] < half).to_numpy()
        year = df["ts"].dt.year.to_numpy()
        months = (df["ts"].iloc[-1] - df["ts"].iloc[0]).days / 30
        sigs = K.signals(df)
        for nm in RULES:
            tr = K.sim(o, h, l, c, sigs[nm], None, True, 0)
            t, r, _, hold = (np.array(z) for z in zip(*tr))
            r = r.astype(float)
            net = r - K.FEE
            rows.append({"TF": tf, "aturan": nm, "trade": len(tr), "per_bulan": round(len(tr) / months, 1), "jam_rata": round(hold.mean() * HOURS[tf], 1),
                         "hit%": round((net > 0).mean() * 100, 1), "kotor_total%": round(r.sum() * 100, 1), "fee_total%": round(len(tr) * K.FEE * 100, 1),
                         "bersih_total%": round(net.sum() * 100, 1), "kotor%/trade": round(r.mean() * 100, 3), "paruh1%": round(net[early[t]].sum() * 100, 1),
                         "paruh2%": round(net[~early[t]].sum() * 100, 1), "terburuk%": round(net.min() * 100, 1), "terbaik%": round(net.max() * 100, 1)})
            for y in np.unique(year[t]):
                m = year[t] == y
                yrs.append({"TF": tf, "aturan": nm, "tahun": int(y), "trade": int(m.sum()), "hit%": round((net[m] > 0).mean() * 100, 1), "bersih_total%": round(net[m].sum() * 100, 1)})
    fence = "`" * 3
    bt._emit("\n### Ikut sinyal (selalu ada posisi, dibalik hanya saat sinyal strategi berlawanan, tanpa SL/TP). kotor = sebelum fee, fee 0,12% per trade. "
             "total% = jumlah hasil semua trade (tanpa leverage, tanpa bunga-berbunga). bersih_total = kotor_total - fee_total\n"
             + fence + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + fence)
    bt._emit("\n### Per tahun (bersih setelah fee)\n" + fence + "\n" + pd.DataFrame(yrs).to_string(index=False) + "\n" + fence)


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
