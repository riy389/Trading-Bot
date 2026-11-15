#!/usr/bin/env python3
"""CEK DATA: unduh/lengkapi cache BTC 5m, 15m, 1h, 4h sesuai DAYS_BACK_TF di backtest.py, lalu tampilkan cakupannya."""
import ccxt
import pandas as pd

import backtest as bt


def main():
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    rows = []
    for tf in ("4h", "1h", "15m", "5m"):
        d = bt.fetch_ohlcv_cached(ex, bt.PAIRS[0], tf, bt.DAYS_BACK_TF[tf])
        if d is None:
            rows.append({"tf": tf, "candle": 0})
            continue
        gaps = [g for g in bt.find_ohlcv_gaps(d, tf) if g[2] > bt.MAX_GAP_CANDLES]
        rows.append({"tf": tf, "diminta_hari": bt.DAYS_BACK_TF[tf], "candle": len(d), "mulai": str(d["ts"].iloc[0])[:16],
                     "akhir": str(d["ts"].iloc[-1])[:16], "gap_besar": len(gaps), "gap_terbesar": max((g[2] for g in gaps), default=0)})
    fence = "`" * 3
    bt._emit("\n### Cek data BTC (gap_besar = lompatan lebih dari " + f"{bt.MAX_GAP_CANDLES}" + " candle; kalau > 0 loader skrip menolak data itu)\n"
             + fence + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + fence)


if __name__ == "__main__":
    main()
