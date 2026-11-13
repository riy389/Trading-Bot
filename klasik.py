#!/usr/bin/env python3
"""
KLASIK + BALIK ARAH: aturan sederhana (tanpa dicari/disetel) diuji sebagai trade sungguhan.
  BB lawan : high menyentuh Bollinger(20,2) atas = JUAL, low menyentuh bawah = BELI (sinyal hanya di candle pertama sentuhan)
  BB ikut  : kebalikannya (sentuh atas = BELI, sentuh bawah = JUAL)
  EMA      : EMA10 memotong ke atas EMA20 = BELI, ke bawah = JUAL
Sinyal dari candle yang sudah close, entry di open candle berikutnya.
mode "balik": ada sinyal berlawanan saat posisi terbuka -> posisi langsung ditutup di open (untung atau rugi berapa pun) dan langsung buka posisi arah sebaliknya.
mode "tanpa balik": sinyal berlawanan diabaikan, keluar di SL/TP atau habis waktu (12 candle di 1h, 16 di 15m).
SL = TP = M% (kena keduanya di satu candle dihitung SL). M kosong = hanya keluar karena balik arah. Fee 0,12% per trade (pulang-pergi), tiap posisi baru = satu trade.
"""
import numpy as np
import pandas as pd

import backtest as bt
import eventtree2 as e2

FEE, WARM = 0.0012, 200
CASES = (("balik", None), ("balik", 0.015), ("balik", 0.027), ("balik", 0.04), ("tanpa balik", 0.027))
HOURS = {"1h": 1.0, "15m": 0.25}
HOLD = {"1h": 12, "15m": 16}


def signals(df):
    c, hi, lo = df["close"], df["high"], df["low"]
    mid, sd = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
    t_lo, t_up = lo <= mid - 2 * sd, hi >= mid + 2 * sd
    e_lo, e_up = t_lo & ~t_lo.shift(1, fill_value=False), t_up & ~t_up.shift(1, fill_value=False)
    bb = np.where(e_lo & ~e_up, 1, np.where(e_up & ~e_lo, -1, 0))
    d = c.ewm(span=10, adjust=False).mean() - c.ewm(span=20, adjust=False).mean()
    ema = np.where((d > 0) & (d.shift(1) <= 0), 1, np.where((d < 0) & (d.shift(1) >= 0), -1, 0))
    out = {"BB lawan": bb, "BB ikut": -bb, "EMA10/20": ema}
    for v in out.values():
        v[:WARM] = 0
    return out


def sim(o, h, l, c, sig, M, rev, N):
    pos, ent, t0, out = 0, 0.0, 0, []
    for t in range(1, len(o)):
        s = sig[t - 1]
        if pos and rev and s == -pos:
            out.append((t, pos * (o[t] / ent - 1), "balik", t - t0))
            pos = 0
        if not pos and s:
            pos, ent, t0 = int(s), o[t], t
        if pos:
            if M:
                sl, tp = ent * (1 - pos * M), ent * (1 + pos * M)
                if (l[t] <= sl) if pos > 0 else (h[t] >= sl):
                    out.append((t, -M, "SL", t - t0 + 1))
                    pos = 0
                    continue
                if (h[t] >= tp) if pos > 0 else (l[t] <= tp):
                    out.append((t, M, "TP", t - t0 + 1))
                    pos = 0
                    continue
            if not rev and t - t0 + 1 >= N:
                out.append((t, pos * (c[t] / ent - 1), "waktu", N))
                pos = 0
    return out


def run(ex):
    rows = []
    for tf in ("1h", "15m"):
        df = e2.load(ex, tf)
        o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        half = df["ts"].iloc[0] + (df["ts"].iloc[-1] - df["ts"].iloc[0]) / 2
        early = (df["ts"] < half).to_numpy()
        months = (df["ts"].iloc[-1] - df["ts"].iloc[0]).days / 30
        for nm, sig in signals(df).items():
            for mode, M in CASES:
                tr = sim(o, h, l, c, sig, M, mode == "balik", HOLD[tf])
                if not tr:
                    continue
                t, r, why, hold = (np.array(z) for z in zip(*tr))
                net = r.astype(float) - FEE
                rows.append({"TF": tf, "aturan": nm, "mode": mode, "M%": round(M * 100, 1) if M else "-", "trade": len(tr), "per_bulan": round(len(tr) / months, 1),
                             "hit%": round((net > 0).mean() * 100, 1), "bersih%/trade": round(net.mean() * 100, 3), "total%": round(net.sum() * 100, 1),
                             "paruh1%": round(net[early[t]].sum() * 100, 1), "paruh2%": round(net[~early[t]].sum() * 100, 1),
                             "TP%": round((why == "TP").mean() * 100), "SL%": round((why == "SL").mean() * 100), "balik%": round((why == "balik").mean() * 100),
                             "waktu%": round((why == "waktu").mean() * 100), "jam_rata": round(hold.mean() * HOURS[tf], 1), "terburuk%": round(net.min() * 100, 1)})
        print(f"  {tf}: selesai")
    fence = "`" * 3
    bt._emit("\n### Aturan klasik + balik arah. bersih = sudah dikurangi fee 0,12% per trade. total% = jumlah hasil semua trade (tanpa leverage, tanpa bunga-berbunga), "
             "paruh1/paruh2 = total di separuh pertama dan kedua data. hit% = trade yang untung bersih. TP/SL/balik/waktu = cara keluar. "
             "Aturan tidak disetel di data (parameter klasik), jadi paruh1 dan paruh2 sama-sama ujian\n"
             + fence + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + fence)


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
