#!/usr/bin/env python3
"""
BB SESUAI DATA ANALISA: bukan sentuh pita, tapi posisi harga di dalam pita (%B) dan lebar pita, dengan rumus yang sama persis seperti di eventtree.py
(close, SMA20, std 20, %B = (close - bawah) / (4 x std), bbw_ratio = lebar pita / rata-rata lebar 100 candle terakhir).
Data: %B lebih tinggi sebelum gerakan naik (1h: median 0,55 vs 0,32 sebelum turun), lebar pita sudah lebih besar sebelum gerakan besar (0,95 vs 0,66 saat diam).
  BB tengah        : %B memotong 0,5 ke atas = BELI, ke bawah = JUAL (harga melewati garis tengah pita)
  BB tengah+lebar  : sama, tapi hanya saat bbw_ratio >= 0,95 (pita sedang melebar)
  BB sentuh (lama) : pembanding dari tes klasik (high/low menyentuh pita luar, searah)
Ikut sinyal: dibalik hanya saat sinyal berlawanan, tanpa SL/TP, entry/balik di open candle berikutnya, fee 0,12% per trade.
"""
import numpy as np
import pandas as pd

import backtest as bt
import eventtree2 as e2
import klasik as K

TFS, HOURS, WIDE = ("1h", "4h"), {"1h": 1.0, "4h": 4.0}, 0.95


def signals(df):
    cs = df["close"]
    sma, sd = cs.rolling(20).mean(), cs.rolling(20).std()
    pctb = (cs - (sma - 2 * sd)) / (4 * sd)
    bw = 4 * sd / sma
    wide = (bw / bw.rolling(100).mean() >= WIDE).to_numpy()
    base = np.where((pctb > 0.5) & (pctb.shift(1) <= 0.5), 1, np.where((pctb < 0.5) & (pctb.shift(1) >= 0.5), -1, 0))
    out = {"BB tengah": base, "BB tengah+lebar": np.where(wide, base, 0), "BB sentuh (lama)": K.signals(df)["BB ikut"]}
    for v in out.values():
        v[:K.WARM] = 0
    return out


def run(ex):
    rows, yrs = [], []
    for tf in TFS:
        df = e2.load(ex, tf)
        o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        half = df["ts"].iloc[0] + (df["ts"].iloc[-1] - df["ts"].iloc[0]) / 2
        early = (df["ts"] < half).to_numpy()
        year = df["ts"].dt.year.to_numpy()
        months = (df["ts"].iloc[-1] - df["ts"].iloc[0]).days / 30
        for nm, sig in signals(df).items():
            tr = K.sim(o, h, l, c, sig, None, True, 0)
            t, r, _, hold = (np.array(z) for z in zip(*tr))
            r = r.astype(float)
            net = r - K.FEE
            t0 = (t - hold).astype(int)
            mfe = np.array([(h[a:b].max() / o[a] - 1) if sig[a - 1] > 0 else (1 - l[a:b].min() / o[a]) for a, b in zip(t0, t)])
            rows.append({"TF": tf, "aturan": nm, "trade": len(tr), "per_bulan": round(len(tr) / months, 1), "jam_rata": round(hold.mean() * HOURS[tf], 1),
                         "hit%": round((net > 0).mean() * 100, 1), "MFE_rata%": round(mfe.mean() * 100, 2), "MFE>=2,7%": round((mfe >= 0.027).mean() * 100, 1), "kotor_total%": round(r.sum() * 100, 1), "fee_total%": round(len(tr) * K.FEE * 100, 1),
                         "bersih_total%": round(net.sum() * 100, 1), "kotor%/trade": round(r.mean() * 100, 3), "paruh1%": round(net[early[t]].sum() * 100, 1),
                         "paruh2%": round(net[~early[t]].sum() * 100, 1), "terburuk%": round(net.min() * 100, 1), "terbaik%": round(net.max() * 100, 1)})
            for y in np.unique(year[t]):
                m = year[t] == y
                yrs.append({"TF": tf, "aturan": nm, "tahun": int(y), "trade": int(m.sum()), "hit%": round((net[m] > 0).mean() * 100, 1), "bersih_total%": round(net[m].sum() * 100, 1)})
    fence = "`" * 3
    bt._emit("\n### BB sesuai data analisa, ikut sinyal (selalu ada posisi, dibalik hanya saat sinyal berlawanan, tanpa SL/TP). kotor = sebelum fee, fee 0,12% per trade. "
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
