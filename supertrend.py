#!/usr/bin/env python3
"""
TRIPLE SUPERTREND (10-1, 11-2, 12-3), SEMUA TF + MTF + TV DETAIL METRICS.
"""
import numpy as np
import pandas as pd

import backtest as bt
import eventtree2 as e2
from zerolag import stats
from tv_metrics import generate_tv_report

ORDER = ("5m", "15m", "1h", "4h", "1d")
FILT = ("5m", "15m", "1h", "4h")
ST_PARAMS = (
    (10, 1.0),
    (11, 2.0),
    (12, 3.0),
)


def rma(series: np.ndarray, length: int) -> np.ndarray:
    out = np.full(len(series), np.nan, dtype=float)
    if len(series) < length:
        return out
    out[length - 1] = np.mean(series[:length])
    alpha = 1.0 / length
    for i in range(length, len(series)):
        out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
    return out


def calc_supertrend(df: pd.DataFrame, atr_len: int, mult: float):
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(c)

    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr_raw = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = rma(tr_raw, atr_len)

    hl2 = (h + l) / 2.0
    basic_ub = hl2 + mult * atr
    basic_lb = hl2 - mult * atr

    ub = np.copy(basic_ub)
    lb = np.copy(basic_lb)
    trend = np.zeros(n, dtype=int)

    start = atr_len
    if start >= n:
        return ub, lb, trend

    trend[start] = 1 if c[start] >= hl2[start] else -1

    for i in range(start + 1, n):
        ub[i] = basic_ub[i] if (basic_ub[i] < ub[i - 1] or c[i - 1] > ub[i - 1]) else ub[i - 1]
        lb[i] = basic_lb[i] if (basic_lb[i] > lb[i - 1] or c[i - 1] < lb[i - 1]) else lb[i - 1]

        if trend[i - 1] == 1:
            trend[i] = -1 if c[i] < lb[i] else 1
        else:
            trend[i] = 1 if c[i] > ub[i] else -1

    return ub, lb, trend


def build_triple_signals(df: pd.DataFrame):
    sts = [calc_supertrend(df, length, mult) for length, mult in ST_PARAMS]
    ub1, lb1, tr1 = sts[0]
    ub2, lb2, tr2 = sts[1]
    ub3, lb3, tr3 = sts[2]

    conf = np.where((tr1 == 1) & (tr2 == 1) & (tr3 == 1), 1,
           np.where((tr1 == -1) & (tr2 == -1) & (tr3 == -1), -1, 0))

    hold_A = conf.copy()

    hold_B = np.zeros(len(conf), dtype=int)
    cur = 0
    for i in range(1, len(conf)):
        if conf[i] != 0 and conf[i] != cur:
            cur = conf[i]
        elif cur == 1 and tr3[i] != 1:
            cur = 0
        elif cur == -1 and tr3[i] != -1:
            cur = 0
        hold_B[i] = cur

    tr_sum = tr1 + tr2 + tr3
    hold_C = np.zeros(len(conf), dtype=int)
    cur = 0
    for i in range(1, len(conf)):
        if conf[i] != 0 and conf[i] != cur:
            cur = conf[i]
        elif cur == 1 and tr_sum[i] < 1:
            cur = 0
        elif cur == -1 and tr_sum[i] > -1:
            cur = 0
        hold_C[i] = cur

    return {
        "conf": conf,
        "A": {"hold": hold_A, "up": ub1, "dn": lb1},
        "B": {"hold": hold_B, "up": ub3, "dn": lb3},
        "C": {"hold": hold_C, "up": ub2, "dn": lb2},
    }


def load_all(ex):
    d = {tf: e2.load(ex, tf) for tf in ("5m", "15m", "1h", "4h")}
    g = d["1h"].set_index("ts")[["open", "high", "low", "close"]].resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index()
    d["1d"] = g.iloc[1:-1].reset_index(drop=True)
    out = {}
    for tf, x in d.items():
        t = pd.to_datetime(x["ts"]).astype("int64").to_numpy()
        sig = build_triple_signals(x)
        out[tf] = {
            "df": x,
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
    idx = np.searchsorted(h["t"] + h["step"], e["t"] + e["step"], side="right") - 1
    h_conf = h["sig"]["conf"]
    return np.where(idx >= 0, h_conf[np.clip(idx, 0, None)], 0)


def build_target(conf, hold, ok_mtf, mtf_mode):
    if mtf_mode == "none":
        return hold.copy()
    if mtf_mode == "B":
        return np.where(ok_mtf, hold, 0)

    tg = np.zeros(len(hold), dtype=int)
    cur = 0
    for i in range(1, len(hold)):
        if cur != 0:
            if hold[i] == 0:
                cur = 0
            elif hold[i] != cur:
                cur = conf[i] if (conf[i] == hold[i] and ok_mtf[i]) else 0
        if cur == 0 and conf[i] != 0 and conf[i] != conf[i - 1]:
            if ok_mtf[i]:
                cur = conf[i]
        tg[i] = cur
    return tg


def trades(e, tg, up, dn):
    o, c = e["o"], e["c"]
    out_stats, out_tv = [], []
    pos, ep, rd, je = 0, 0.0, 0.0, 0
    for i in range(1, len(c) - 1):
        if tg[i] != tg[i - 1]:
            px = o[i + 1]
            if pos:
                reason = "flip" if tg[i] == -pos else "lepas"
                out_stats.append((je, pos, ep, px, rd, reason))
                out_tv.append((je, i + 1, pos, ep, px, rd, reason))
            pos = tg[i]
            if pos:
                ep, je = px, i + 1
                rd = max(abs((px - dn[i]) / px) if pos == 1 else abs((up[i] - px) / px), 1e-4)
    if pos:
        out_stats.append((je, pos, ep, c[-1], rd, "end"))
        out_tv.append((je, len(c) - 1, pos, ep, c[-1], rd, "end"))
    return out_stats, out_tv


def run(ex):
    D = load_all(ex)
    fence = "`" * 3
    exit_labels = {
        "A": "Exit-A (Cepat: 1 ST flip = keluar)",
        "B": "Exit-B (Tahan Tren: pita 12-3 flip = keluar)",
        "C": "Exit-C (Mayoritas: >=2 ST flip = keluar)",
    }

    for exit_code, exit_desc in exit_labels.items():
        rows, yrs, tv_blocks = [], [], []
        for tf in ORDER:
            e = D[tf]
            conf = e["sig"]["conf"]
            hold = e["sig"][exit_code]["hold"]
            up = e["sig"][exit_code]["up"]
            dn = e["sig"][exit_code]["dn"]

            above = FILT[FILT.index(tf) + 1:] if tf in FILT else ()
            sets = [above[-k:] for k in range(1, len(above) + 1)]
            modes = [("sendiri", "none", ())] + [
                (f"MTF-{m} +{'/'.join(st)}", m, st) for st in sets for m in ("A", "B")
            ]

            for nm, md, st in modes:
                ok = hold != 0
                for h in st:
                    ok = ok & (align(e, D[h]) == hold)
                tg = build_target(conf, hold, ok, md)
                tr_stats, tr_tv = trades(e, tg, up, dn)
                r, y = stats(tr_stats, e["yr"], tf, nm, e["months"])
                if r:
                    rows.append(r)
                    yrs += y
                    if r["bersihR_total"] > 0:
                        tv_blocks.append(generate_tv_report(tr_tv, e["df"], tf, f"{exit_code} - {nm}"))

        bt._emit(
            f"\n### Triple Supertrend (10-1, 11-2, 12-3) — {exit_desc}\n"
            + fence + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + fence
        )
        for blk in tv_blocks:
            bt._emit(blk)
        bt._emit(
            f"\n### Per tahun ({exit_code} - bersihR)\n"
            + fence + "\n" + pd.DataFrame(yrs).to_string(index=False) + "\n" + fence
        )


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
