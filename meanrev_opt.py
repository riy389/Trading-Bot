#!/usr/bin/env python3
"""
MEAN REVERSION OPTIMIZED (3-Bar + Body >70% + Filter Ukuran Gerakan + Hard Stop Loss).
Menguji apakah memperbesar minimum move 3 candle + memasang Hard Stop Loss mampu
mengubah kotorR positif di TF 15m (dan 5m/1h) menjadi bersihR positif setelah fee 0,12% + slippage 0,10%.
Eksekusi entry di open bar berikutnya setelah konfirmasi close; Hard SL dicek secara intrabar (low/high).
"""
import numpy as np
import pandas as pd

import backtest as bt
import eventtree2 as e2
from zerolag import stats

ORDER = ("5m", "15m", "1h", "4h")
FILT = ("5m", "15m", "1h", "4h")

MOVE_LIMIT = 0.70
MA_LENGTH = 200

VARIANTS = {
    "Opt-1 (Move>=1.0% + SL_Bar3)": {"min_move": 0.010, "sl_mode": "bar3"},
    "Opt-2 (Move>=1.5% + SL_Bar3)": {"min_move": 0.015, "sl_mode": "bar3"},
    "Opt-3 (Move>=1.2% + SL_1.5ATR)": {"min_move": 0.012, "sl_mode": "atr"},
}


def prepare_base(df: pd.DataFrame):
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(c)

    ma = pd.Series(c).rolling(MA_LENGTH).mean().to_numpy(float)

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

    # Total persentase gerakan dari open bar ke-1 hingga close bar ke-3
    move_pct_down = np.zeros(n, dtype=float)
    move_pct_up = np.zeros(n, dtype=float)
    move_pct_down[2:] = (o[:-2] - c[2:]) / np.maximum(o[:-2], 1e-8)
    move_pct_up[2:] = (c[2:] - o[:-2]) / np.maximum(o[:-2], 1e-8)

    # Titik ekstrem terendah / tertinggi selama 3 candle pemicu
    low3 = np.copy(l)
    high3 = np.copy(h)
    low3[2:] = np.minimum(l[2:], np.minimum(l[1:-1], l[:-2]))
    high3[2:] = np.maximum(h[2:], np.maximum(h[1:-1], h[:-2]))

    long_exit = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)
    long_exit[1:] = c[1:] > h[:-1]
    short_exit[1:] = c[1:] < l[:-1]

    bias = np.where(np.isnan(ma), 0, np.where(c > ma, 1, -1))

    return {
        "o": o,
        "h": h,
        "l": l,
        "c": c,
        "ma": ma,
        "atr": atr14,
        "bias": bias,
        "base_long": is_three_down & big_move_down,
        "base_short": (c < ma) & is_three_up & big_move_up,
        "move_down": move_pct_down,
        "move_up": move_pct_up,
        "low3": low3,
        "high3": high3,
        "long_exit": long_exit,
        "short_exit": short_exit,
    }


def load_all(ex):
    d = {tf: e2.load(ex, tf) for tf in ("5m", "15m", "1h", "4h")}
    out = {}
    for tf, x in d.items():
        t = pd.to_datetime(x["ts"]).astype("int64").to_numpy()
        base = prepare_base(x)
        out[tf] = {
            "t": t,
            "step": int(np.median(np.diff(t))),
            "base": base,
            "yr": x["ts"].dt.year.to_numpy(),
            "months": (x["ts"].iloc[-1] - x["ts"].iloc[0]).days / 30,
        }
    return out


def align(e, h):
    idx = np.searchsorted(h["t"] + h["step"], e["t"] + e["step"], side="right") - 1
    h_bias = h["base"]["bias"]
    return np.where(idx >= 0, h_bias[np.clip(idx, 0, None)], 0)


def simulate_trades(e, min_move: float, sl_mode: str, st_filters, mtf_mode: str, D):
    b = e["base"]
    o, h, l, c = b["o"], b["h"], b["l"], b["c"]
    atr, low3, high3 = b["atr"], b["low3"], b["high3"]
    long_exit, short_exit = b["long_exit"], b["short_exit"]
    n = len(c)

    long_sig = b["base_long"] & (b["move_down"] >= min_move)
    short_sig = b["base_short"] & (b["move_up"] >= min_move)

    # Filter MTF (tanpa lookahead bias)
    ok_long = np.ones(n, dtype=bool)
    ok_short = np.ones(n, dtype=bool)
    for tf_h in st_filters:
        aligned_bias = align(e, D[tf_h])
        ok_long = ok_long & (aligned_bias == 1)
        ok_short = ok_short & (aligned_bias == -1)

    out = []
    pos = 0
    ep = 0.0
    sl_px = 0.0
    rd = 0.0
    je = 0

    for i in range(MA_LENGTH, n - 1):
        # 1. Evaluasi posisi aktif pada bar i (entry sudah masuk di open bar i)
        if pos == 1:
            # Cek apakah Hard Stop Loss tersentuh intrabar di bar i
            if l[i] <= sl_px:
                exit_px = min(o[i], sl_px)
                out.append((je, 1, ep, exit_px, rd, "sl"))
                pos = 0
            # Cek apakah sinyal exit normal terpenuhi di close bar i -> keluar di open i+1
            elif long_exit[i] or (mtf_mode == "B" and not ok_long[i]):
                out.append((je, 1, ep, o[i + 1], rd, "lepas"))
                pos = 0

        elif pos == -1:
            if h[i] >= sl_px:
                exit_px = max(o[i], sl_px)
                out.append((je, -1, ep, exit_px, rd, "sl"))
                pos = 0
            elif short_exit[i] or (mtf_mode == "B" and not ok_short[i]):
                out.append((je, -1, ep, o[i + 1], rd, "lepas"))
                pos = 0

        # 2. Jika posisi flat setelah evaluasi bar i, cek sinyal entry baru di close bar i -> masuk di open i+1
        if pos == 0:
            if long_sig[i] and ok_long[i]:
                entry_px = o[i + 1]
                if sl_mode == "bar3":
                    raw_sl = low3[i] * 0.9995  # buffer 0.05% di bawah low 3-bar
                else:
                    raw_sl = entry_px - 1.5 * atr[i]

                if raw_sl < entry_px * 0.998:  # pastikan jarak SL minimal 0.2% agar masuk akal
                    pos = 1
                    ep = entry_px
                    sl_px = raw_sl
                    rd = max((ep - sl_px) / ep, 1e-4)
                    je = i + 1

            elif short_sig[i] and ok_short[i]:
                entry_px = o[i + 1]
                if sl_mode == "bar3":
                    raw_sl = high3[i] * 1.0005  # buffer 0.05% di atas high 3-bar
                else:
                    raw_sl = entry_px + 1.5 * atr[i]

                if raw_sl > entry_px * 1.002:
                    pos = -1
                    ep = entry_px
                    sl_px = raw_sl
                    rd = max((sl_px - ep) / ep, 1e-4)
                    je = i + 1

    if pos != 0:
        out.append((je, pos, ep, c[-1], rd, "end"))
    return out


def run(ex):
    D = load_all(ex)
    fence = "`" * 3

    for var_name, cfg in VARIANTS.items():
        rows, yrs = [], []
        for tf in ORDER:
            e = D[tf]
            above = FILT[FILT.index(tf) + 1:] if tf in FILT else ()
            sets = [above[-k:] for k in range(1, len(above) + 1)]
            modes = [("sendiri", "none", ())] + [
                (f"MTF-{m} +{'/'.join(st)}", m, st) for st in sets for m in ("A", "B")
            ]

            for nm, md, st in modes:
                tr_list = simulate_trades(e, cfg["min_move"], cfg["sl_mode"], st, md, D)
                r, y = stats(tr_list, e["yr"], tf, nm, e["months"])
                if r:
                    rows.append(r)
                    yrs += y

        bt._emit(
            f"\n### MeanRev Optimized — {var_name} (fee 0,12% + slippage 0,10%)\n"
            + fence + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + fence
        )
        bt._emit(
            f"\n### Per tahun ({var_name.split()[0]} - bersihR)\n"
            + fence + "\n" + pd.DataFrame(yrs).to_string(index=False) + "\n" + fence
        )


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
