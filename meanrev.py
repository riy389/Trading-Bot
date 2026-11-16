#!/usr/bin/env python3
"""
MEAN REVERSION (3 Candle Searah + Big Move > 70% + SMA 200 untuk Short), SEMUA TF + MTF.
Aturan asli Pine Script (100% Non-Repaint, evaluasi saat candle close, eksekusi di open bar berikutnya):
  - Long Entry : 3 candle merah berturut-turut + candle ke-3 memiliki body ratio > 70%.
  - Long Exit  : close > high[1] (atau terlepas oleh filter MTF pada mode MTF-B).
  - Short Entry: close < SMA(200) + 3 candle hijau berturut-turut + candle ke-3 memiliki body ratio > 70%.
  - Short Exit : close < low[1] (atau terlepas oleh filter MTF pada mode MTF-B).
Biaya fee 0,12% + slippage 0,10% per trade (ditangani oleh modul stats).
"""
import numpy as np
import pandas as pd

import backtest as bt
import eventtree2 as e2
from zerolag import stats

ORDER = ("5m", "15m", "1h", "4h", "1d")
FILT = ("5m", "15m", "1h", "4h")

MOVE_LIMIT = 0.70
MA_LENGTH = 200


def calc_meanrev_signals(df: pd.DataFrame):
    """Menghitung sinyal entry/exit Mean Reversion sesuai logika Pine Script."""
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(c)

    # SMA 200
    ma = pd.Series(c).rolling(MA_LENGTH).mean().to_numpy(float)

    # ATR(14) untuk normalisasi satuan risiko R pada tabel statistik
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr14 = pd.Series(tr).rolling(14).mean().bfill().to_numpy(float)

    down_bar = o > c
    up_bar = o < c

    is_three_down = np.zeros(n, dtype=bool)
    is_three_up = np.zeros(n, dtype=bool)
    is_three_down[2:] = down_bar[2:] & down_bar[1:-1] & down_bar[:-2]
    is_three_up[2:] = up_bar[2:] & up_bar[1:-1] & up_bar[:-2]

    rng = 0.001 + (h - l)
    big_move_down = ((o - c) / rng) > MOVE_LIMIT
    big_move_up = ((c - o) / rng) > MOVE_LIMIT

    long_buy = is_three_down & big_move_down
    short_buy = (c < ma) & is_three_up & big_move_up

    long_exit = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)
    long_exit[1:] = c[1:] > h[:-1]
    short_exit[1:] = c[1:] < l[:-1]

    # Bangun status posisi mandiri (sendiri) dan bias tren makro (close vs SMA 200) untuk filter MTF
    hold = np.zeros(n, dtype=int)
    cur = 0
    for i in range(MA_LENGTH, n):
        if cur == 1 and long_exit[i]:
            cur = 0
        elif cur == -1 and short_exit[i]:
            cur = 0

        if cur == 0:
            if long_buy[i]:
                cur = 1
            elif short_buy[i]:
                cur = -1
        hold[i] = cur

    # Bias MTF dari timeframe lebih besar: +1 jika close > SMA 200, -1 jika close < SMA 200
    bias = np.where(np.isnan(ma), 0, np.where(c > ma, 1, -1))

    return {
        "hold": hold,
        "bias": bias,
        "long_buy": long_buy,
        "short_buy": short_buy,
        "long_exit": long_exit,
        "short_exit": short_exit,
        "atr": atr14,
    }


def load_all(ex):
    d = {tf: e2.load(ex, tf) for tf in ("5m", "15m", "1h", "4h")}
    g = d["1h"].set_index("ts")[["open", "high", "low", "close"]].resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna().reset_index()
    d["1d"] = g.iloc[1:-1].reset_index(drop=True)

    out = {}
    for tf, x in d.items():
        t = pd.to_datetime(x["ts"]).astype("int64").to_numpy()
        sig = calc_meanrev_signals(x)
        out[tf] = {
            "t": t,
            "step": int(np.median(np.diff(t))),
            "o": x["open"].to_numpy(float),
            "c": x["close"].to_numpy(float),
            "sig": sig,
            "yr": x["ts"].dt.year.to_numpy(),
            "months": (x["ts"].iloc[-1] - x["ts"].iloc[0]).days / 30,
        }
    return out


def align(e, h):
    """Menyelaraskan bias SMA 200 TF besar ke TF entry tanpa lookahead bias."""
    idx = np.searchsorted(h["t"] + h["step"], e["t"] + e["step"], side="right") - 1
    h_bias = h["sig"]["bias"]
    return np.where(idx >= 0, h_bias[np.clip(idx, 0, None)], 0)


def build_target(sig, ok_mtf, mtf_mode):
    """Menghitung target posisi (-1, 0, 1) pada setiap bar."""
    if mtf_mode == "none":
        return sig["hold"].copy()

    n = len(sig["hold"])
    long_buy = sig["long_buy"]
    short_buy = sig["short_buy"]
    long_exit = sig["long_exit"]
    short_exit = sig["short_exit"]

    tg = np.zeros(n, dtype=int)
    cur = 0
    for i in range(MA_LENGTH, n):
        if cur == 1:
            if long_exit[i] or (mtf_mode == "B" and not ok_mtf[i]):
                cur = 0
        elif cur == -1:
            if short_exit[i] or (mtf_mode == "B" and not ok_mtf[i]):
                cur = 0

        if cur == 0:
            if long_buy[i] and ok_mtf[i]:
                cur = 1
            elif short_buy[i] and ok_mtf[i]:
                cur = -1
        tg[i] = cur
    return tg


def trades(e, tg):
    o, c, atr = e["o"], e["c"], e["sig"]["atr"]
    out, pos, ep, rd, je = [], 0, 0.0, 0.0, 0
    for i in range(1, len(c) - 1):
        if tg[i] != tg[i - 1]:
            px = o[i + 1]
            if pos:
                out.append((je, pos, ep, px, rd, "flip" if tg[i] == -pos else "lepas"))
            pos = tg[i]
            if pos:
                ep, je = px, i + 1
                rd = max(atr[i] / px, 1e-4)
    if pos:
        out.append((je, pos, ep, c[-1], rd, "end"))
    return out


def run(ex):
    D = load_all(ex)
    rows, yrs = [], []

    for tf in ORDER:
        e = D[tf]
        sig = e["sig"]
        above = FILT[FILT.index(tf) + 1:] if tf in FILT else ()
        sets = [above[-k:] for k in range(1, len(above) + 1)]
        modes = [("sendiri", "none", ())] + [
            (f"MTF-{m} +{'/'.join(st)}", m, st) for st in sets for m in ("A", "B")
        ]

        for nm, md, st in modes:
            # Untuk MTF, posisi Long/Short hanya diambil jika selaras dengan bias SMA 200 TF lebih besar
            want_dir = np.where(sig["long_buy"], 1, np.where(sig["short_buy"], -1, sig["hold"]))
            ok = np.ones(len(e["c"]), dtype=bool)
            for h in st:
                ok = ok & (align(e, D[h]) == want_dir)

            tg = build_target(sig, ok, md)
            r, y = stats(trades(e, tg), e["yr"], tf, nm, e["months"])
            if r:
                rows.append(r)
                yrs += y

    fence = "`" * 3
    bt._emit(
        "\n### Mean Reversion (3-Bar + Body >70% + SMA200) semua TF + MTF. bersihR termasuk fee 0,12% + slippage 0,10%.\n"
        + fence + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + fence
    )
    bt._emit(
        "\n### Per tahun (MeanRev - bersihR)\n"
        + fence + "\n" + pd.DataFrame(yrs).to_string(index=False) + "\n" + fence
    )


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
