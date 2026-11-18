#!/usr/bin/env python3
"""
Regime filter (parameter TERKUNCI dari regime_filter_test.py) diuji di BTC/ETH/SOL 1h.
Varian: Base, Standby (ADX>22 & ER>0.22), Standby Ketat (ADX>25 & ER>0.28). Tidak ada tuning ulang.
"""
import pandas as pd

import backtest as bt
import regime_filter_test as rf
from zerolag import stats

PAIRS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
VARS = rf.VARIANTS[:3]
COLS = ["pair", "varian", "trade", "hit%", "kotorR/trade", "bersihR/trade", "bersihR_total", "R_21-24", "R_25-26"]


def load(ex, sym):
    d = bt.fetch_ohlcv_cached(ex, sym, "1h", bt.DAYS_BACK_TF["1h"])
    if d is None or len(d) < 500 or any(g[2] > bt.MAX_GAP_CANDLES for g in bt.find_ohlcv_gaps(d, "1h")):
        return None
    return d.iloc[:-1].reset_index(drop=True)


def run(ex):
    rows, yrs = [], []
    fence = "`" * 3
    for sym in PAIRS:
        pair = sym.split("/")[0]
        d = load(ex, sym)
        if d is None:
            bt._emit(f"\n{pair}: data tidak layak, dilewati")
            continue
        bt._emit(f"\n{pair}: {len(d)} candle 1h, {d['ts'].iloc[0]} s/d {d['ts'].iloc[-1]}")
        data = rf.prepare_data(d)
        for cfg in VARS:
            tr, _ = rf.simulate(data, cfg)
            r, y = stats(tr, data["yr"], "1h", cfg["name"], data["months"])
            if r:
                r["pair"] = pair
                rows.append(r)
                if cfg is VARS[2]:
                    for k in y:
                        k["pair"] = pair
                    yrs += y
    bt._emit("\n### Regime filter 1h BTC/ETH/SOL (fee 0,12% + slippage 0,10%)\n"
             + fence + "\n" + pd.DataFrame(rows)[COLS].to_string(index=False) + "\n" + fence)
    bt._emit("\n### Per tahun, Standby Ketat (bersihR)\n"
             + fence + "\n" + pd.DataFrame(yrs)[["pair", "tahun", "trade", "hit%", "bersihR"]].to_string(index=False) + "\n" + fence)


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
