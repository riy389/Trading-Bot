#!/usr/bin/env python3
"""
MACD_DH: adaptasi "How to Design a Simple Multi-Timeframe Trend Strategy on Bitcoin" (PapersWithBacktest).
Pakai data H1 asli + D1 hasil resample dari H1 yang sama (tanpa interpolasi, tanpa lookahead).

Aturan (varian "stop" dari paper, long-only):
- Filter: MACD(12,26,9) D1 bullish (MACD > signal), memakai HANYA candle D1 yang sudah close sebelum candle H1 mulai.
- Sinyal: MACD(12,26,9) H1 cross up di close candle H1 i. Entry di OPEN candle i+1 (konvensi mesin backtest.py).
- Exit: di CLOSE candle H1 yang close-nya < low candle H1 sebelumnya (dicek mulai candle entry). Tanpa SL intrabar (paper tidak punya).
- 1R = jarak entry ke low candle sinyal (hanya untuk ukuran posisi 1% dan perhitungan R; rugi bisa lewat -1R karena tidak ada SL).
- Biaya dipotong mesin saat trade ditutup (apply_costs), sama seperti strategi lain.
Pembanding: varian tanpa filter D1 (H1 MACD cross saja, exit sama).
"""

import os

import numpy as np
import pandas as pd

import backtest as bt

SYMBOL = "BTC/USDT:USDT"
TF = "1h"
OUTPUT_TRADES = "trades_macddh.csv"


def ema_std(x, length):
    """EMA standar: seed = SMA dari `length` nilai valid pertama, alpha = 2 / (length + 1)."""
    out = np.full(len(x), np.nan)
    valid = np.flatnonzero(~np.isnan(x))
    if len(valid) < length:
        return out
    s = valid[0] + length - 1
    out[s] = x[valid[0]:s + 1].mean()
    k = 2.0 / (length + 1)
    for i in range(s + 1, len(x)):
        out[i] = k * x[i] + (1 - k) * out[i - 1]
    return out


def macd_bull(close, fast=12, slow=26, sig=9):
    """Return (bull: MACD > signal, cross_up: baru cross ke atas di candle ini)."""
    macd = ema_std(close, fast) - ema_std(close, slow)
    signal = ema_std(macd, sig)
    with np.errstate(invalid="ignore"):
        bull = macd > signal
    bull1 = np.r_[False, bull[:-1]]
    return bull, bull & ~bull1


def daily_from_h1(dfh):
    """Candle D1 (UTC) dari candle H1. Hanya hari lengkap (24 candle H1) yang dipakai."""
    g = dfh.set_index("ts").resample("1D").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), n=("close", "size"))
    full = g[g["n"] == 24].drop(columns="n").reset_index()
    print(f"  D1 dari H1: {len(full)} hari lengkap, {int((g['n'] != 24).sum())} hari dibuang (tidak lengkap)")
    return full


def d1_filter_on_h1(dfh, dfd):
    """True per candle H1 kalau MACD D1 bullish, memakai HANYA candle D1 yang sudah close sebelum candle H1 itu mulai.
    dfd['ts'] = waktu buka candle D1, jadi candle itu baru close di ts + 1 hari (H1 jam 00:00 hari X sudah boleh
    memakai D1 hari X-1; H1 mana pun di hari X tidak boleh memakai D1 hari X)."""
    d_bull, _ = macd_bull(dfd["close"].to_numpy(dtype=float))
    td_close = (dfd["ts"] + pd.Timedelta(days=1)).dt.tz_convert(None).to_numpy("datetime64[ns]")
    th = dfh["ts"].dt.tz_convert(None).to_numpy("datetime64[ns]")
    idx = np.searchsorted(td_close, th, side="right") - 1
    return np.where(idx >= 0, d_bull[np.clip(idx, 0, None)], False)


def simulate_macddh(df, entry):
    """Sinyal entry[i] di close candle i -> entry di open candle i+1. Exit di close candle j (j >= i+1)
    pertama yang close-nya < low candle j-1. Satu posisi sekali waktu. Trade terbuka di akhir data ditutup di close terakhir."""
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    tss = df["ts"].tolist()
    trades, pos, pending = [], None, False

    def close_trade(p, i, reason):
        gross_ret = (c[i] - p["entry"]) / p["entry"]
        return {"side": "long", "entry_ts": str(p["ts"]), "exit_ts": str(tss[i]),
                "entry_price": p["entry"], "exit_price": c[i], "sl_pct": p["sl_pct"],
                "size": bt.RISK_PER_TRADE / p["sl_pct"], "gross_ret": gross_ret,
                "gross_R": gross_ret / p["sl_pct"], "exit_reason": reason, "mfe_R": p["mfe"]}

    for i in range(1, len(df)):
        if pending and pos is None:
            e = o[i]
            nl = e - l[i - 1]                       # l[i-1] = low candle sinyal
            if nl > 0 and nl / e >= bt.MIN_SL_PCT:  # sama dengan batas leverage mesin
                pos = {"entry": e, "ts": tss[i], "sl_pct": nl / e, "mfe": 0.0}
        pending = False
        if pos is not None:
            pos["mfe"] = max(pos["mfe"], (h[i] - pos["entry"]) / (pos["entry"] * pos["sl_pct"]))
            if c[i] < l[i - 1]:
                trades.append(close_trade(pos, i, "close_below_prevlow"))
                pos = None
        if entry[i] and pos is None:
            pending = True
    if pos is not None:
        trades.append(close_trade(pos, len(df) - 1, "end_data"))
    return trades


def check_macddh(exchange):
    """Cek cepat: 10 cross up H1 terakhir dan apakah filter D1 lolos (candle H1 terakhir belum close, dibuang)."""
    dfh = bt.fetch_ohlcv(exchange, SYMBOL, TF, 120)
    if dfh is None:
        return
    dfh = dfh.iloc[:-1].reset_index(drop=True)
    dfd = daily_from_h1(dfh)
    d_ok = d1_filter_on_h1(dfh, dfd)
    _, cross = macd_bull(dfh["close"].to_numpy(dtype=float))
    t = pd.DataFrame({"utc": dfh["ts"].dt.tz_localize(None),
                      "wib": (dfh["ts"] + pd.Timedelta(hours=7)).dt.tz_localize(None),
                      "close": dfh["close"], "d1_bull": d_ok, "entry": cross & d_ok})
    t = t[cross].tail(10)
    bt._emit("\n### Cek sinyal MACD_DH BTC 1h (waktu = waktu BUKA candle sinyal; entry di open candle sesudahnya). "
             "entry = True hanya kalau MACD D1 juga bullish\n```\n" + t.to_string(index=False) + "\n```")


def run_macddh(exchange):
    dfh = bt.fetch_ohlcv_cached(exchange, SYMBOL, TF, bt.DAYS_BACK_TF[TF])
    if dfh is None or len(dfh) < 1000:
        raise SystemExit("[!] Data H1 tidak cukup.")
    if any(g[2] > bt.MAX_GAP_CANDLES for g in bt.find_ohlcv_gaps(dfh, TF)):
        raise SystemExit("[!] Gap besar di data H1, periksa cache.")
    dfh = dfh.iloc[:-1].reset_index(drop=True)   # candle terakhir mungkin belum close

    dfd = daily_from_h1(dfh)
    d_ok = d1_filter_on_h1(dfh, dfd)
    _, cross = macd_bull(dfh["close"].to_numpy(dtype=float))
    variants = {"MACD_DH_D1filter": cross & d_ok, "MACD_H1_noD1": cross}

    split_ts = dfh["ts"].iloc[0] + (dfh["ts"].iloc[-1] - dfh["ts"].iloc[0]) / 2
    days = max((dfh["ts"].iloc[-1] - dfh["ts"].iloc[0]).days, 1)
    cost_net = bt.net_cost(SYMBOL)
    rows, trade_rows = [], []
    for vname, entry in variants.items():
        trades = simulate_macddh(dfh, entry)
        for cname, cost in (("net", cost_net), ("gross", 0.0)):
            costed = bt.apply_costs(trades, cost)
            cap = [x for x in bt.capital_sim(costed, vname, SYMBOL, TF)
                   if x["risk_pct"] == 2.0 and x["policy"] == "skip"][0]
            r = bt.fs_row(TF, 0, 0, vname, costed, split_ts, cap, days)
            r["cost"] = cname
            if cname == "net" and costed and r.get("pf_IS") is not None and r.get("pf_OOS") is not None:
                r["edge"] = int(r["pf_IS"] >= 1.2 and r["pf_OOS"] >= 1.2 and r["netR"] > 0
                                and r["n_IS"] >= 30 and r["n_OOS"] >= 30)
            rows.append(r)
            print(f"  {vname} {cname}: {len(costed)} trade")
            if cname == "net":
                trade_rows += [{"variant": vname, **t} for t in costed]

    pd.DataFrame(trade_rows).to_csv(OUTPUT_TRADES, index=False)
    fence = "`" * 3
    bt._emit("\n### MACD_DH, BTC 1h, long-only. D1filter = MACD D1 (12/26/9) bullish + MACD H1 cross up, exit di close candle H1 "
             "yang close-nya < low candle sebelumnya (tanpa SL intrabar). noD1 = pembanding tanpa filter D1. D1 dari resample H1 "
             "(tanpa lookahead). net = biaya " + f"{cost_net * 100:.2f}" + "%, gross = tanpa biaya. per_day = trade/hari, "
             "netR = rata-rata net_R, wr% = win rate net, med_h = lama tahan median (jam), eq$10 / dd% = simulasi modal $10 risiko 2%. "
             "edge = PF IS dan OOS >= 1,2, netR > 0, n >= 30 per paruh\n"
             + fence + "\n" + pd.DataFrame(rows).drop(columns=["tf", "A", "ATR"], errors="ignore").to_string(index=False)
             + "\n" + fence)


def main():
    import ccxt
    exchange = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    exchange.load_markets()
    check_macddh(exchange)
    if os.environ.get("CHECK_ONLY") == "1":
        return
    run_macddh(exchange)


if __name__ == "__main__":
    main()
