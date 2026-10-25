#!/usr/bin/env python3
"""
Backtest engine sederhana: Trend-Following (EMA cross) vs Mean-Reversion (RSI)
Data asli dari Bitget via CCXT. Long-only, fee Bitget nyata, tanpa leverage.

Cara pakai:
    pip install ccxt pandas numpy --break-system-packages
    python3 backtest.py

Edit PAIRS / TIMEFRAMES / DAYS_BACK di bawah untuk ubah cakupan uji.
"""

import ccxt
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta, timezone

# ============== KONFIGURASI ==============
PAIRS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]  # Bitget perpetual swap
TIMEFRAMES = ["15m", "1h", "4h", "1d"]
DAYS_BACK = 365 * 2          # 2 tahun data
TAKER_FEE = 0.0006           # 0.06% per sisi (Bitget futures taker)
ROUND_TRIP_FEE = TAKER_FEE * 2
SLIPPAGE = 0.0005             # 0.05% asumsi slippage per sisi, konservatif

# Parameter strategi
EMA_FAST, EMA_SLOW = 21, 55
RSI_PERIOD = 14
RSI_OVERSOLD, RSI_EXIT = 30, 50

OUTPUT_JSON = "results.json"
OUTPUT_CSV = "results_summary.csv"
# ===========================================


def fetch_ohlcv(exchange, symbol, timeframe, days_back):
    """Ambil OHLCV lengkap dengan pagination, karena CCXT membatasi per-call.
    Berhenti hanya saat data sudah mencapai waktu sekarang, bukan saat satu
    call mengembalikan kurang dari `limit` candle (Bitget bisa saja balas
    sebagian meski masih ada data lebih lanjut)."""
    since = exchange.parse8601(
        (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    now_ms = exchange.milliseconds()
    all_candles = []
    limit = 1000
    stall_count = 0
    while since < now_ms:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        except Exception as e:
            print(f"  [!] Error fetch {symbol} {timeframe} (since={since}): {e}, retry dalam 3s...")
            time.sleep(3)
            stall_count += 1
            if stall_count > 5:
                print(f"  [!] Gagal berulang, hentikan fetch {symbol} {timeframe}.")
                break
            continue

        if not candles:
            # tidak ada data di jendela ini; majukan since 1 hari supaya tidak stuck
            since += 86400000
            stall_count += 1
            if stall_count > 10:
                break
            continue

        stall_count = 0
        all_candles.extend(candles)
        last_ts = candles[-1][0]
        if last_ts <= since:
            # exchange tidak maju sama sekali -> hindari infinite loop
            since += 86400000
        else:
            since = last_ts + 1
        print(f"    ...{len(all_candles)} candle terkumpul (sampai {candles[-1][0]})")
        time.sleep(exchange.rateLimit / 1000)
        if len(all_candles) > 300000:  # safety cap
            break

    if not all_candles:
        return None

    df = pd.DataFrame(all_candles, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    return df


def add_indicators(df):
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def run_trend_following(df):
    """Long-only EMA cross: masuk saat fast cross up slow, keluar saat fast cross down slow."""
    trades = []
    position = None
    for i in range(1, len(df)):
        prev_fast, prev_slow = df["ema_fast"].iloc[i - 1], df["ema_slow"].iloc[i - 1]
        cur_fast, cur_slow = df["ema_fast"].iloc[i], df["ema_slow"].iloc[i]
        price = df["close"].iloc[i]
        ts = df["ts"].iloc[i]

        cross_up = prev_fast <= prev_slow and cur_fast > cur_slow
        cross_down = prev_fast >= prev_slow and cur_fast < cur_slow

        if position is None and cross_up:
            entry_price = price * (1 + SLIPPAGE)
            position = {"entry_ts": ts, "entry_price": entry_price}
        elif position is not None and cross_down:
            exit_price = price * (1 - SLIPPAGE)
            pnl_pct = (exit_price - position["entry_price"]) / position["entry_price"] - ROUND_TRIP_FEE
            trades.append({
                "entry_ts": str(position["entry_ts"]), "exit_ts": str(ts),
                "entry_price": position["entry_price"], "exit_price": exit_price,
                "pnl_pct": pnl_pct,
            })
            position = None
    return trades


def run_mean_reversion(df):
    """Long-only RSI: masuk saat RSI < 30, keluar saat RSI > 50."""
    trades = []
    position = None
    for i in range(1, len(df)):
        rsi = df["rsi"].iloc[i]
        prev_rsi = df["rsi"].iloc[i - 1]
        price = df["close"].iloc[i]
        ts = df["ts"].iloc[i]

        if pd.isna(rsi) or pd.isna(prev_rsi):
            continue

        enter_signal = prev_rsi >= RSI_OVERSOLD and rsi < RSI_OVERSOLD
        exit_signal = prev_rsi <= RSI_EXIT and rsi > RSI_EXIT

        if position is None and enter_signal:
            entry_price = price * (1 + SLIPPAGE)
            position = {"entry_ts": ts, "entry_price": entry_price}
        elif position is not None and exit_signal:
            exit_price = price * (1 - SLIPPAGE)
            pnl_pct = (exit_price - position["entry_price"]) / position["entry_price"] - ROUND_TRIP_FEE
            trades.append({
                "entry_ts": str(position["entry_ts"]), "exit_ts": str(ts),
                "entry_price": position["entry_price"], "exit_price": exit_price,
                "pnl_pct": pnl_pct,
            })
            position = None
    return trades


def compute_metrics(trades, strategy_name, symbol, timeframe):
    if not trades:
        return {
            "strategy": strategy_name, "symbol": symbol, "timeframe": timeframe,
            "n_trades": 0, "total_return_pct": 0, "win_rate_pct": 0,
            "profit_factor": None, "max_drawdown_pct": 0, "avg_pnl_pct": 0,
        }

    pnls = [t["pnl_pct"] for t in trades]
    n = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / n * 100

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0)

    # compound equity curve utk total return & max drawdown
    equity = [1.0]
    for p in pnls:
        equity.append(equity[-1] * (1 + p))
    equity = np.array(equity)
    total_return = (equity[-1] - 1) * 100

    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min() * 100

    return {
        "strategy": strategy_name, "symbol": symbol, "timeframe": timeframe,
        "n_trades": n, "total_return_pct": round(total_return, 2),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
        "max_drawdown_pct": round(max_dd, 2),
        "avg_pnl_pct": round(np.mean(pnls) * 100, 3),
    }


def main():
    exchange = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    exchange.load_markets()

    all_results = []
    raw_trades = {}

    for symbol in PAIRS:
        for tf in TIMEFRAMES:
            print(f"\n=== Fetching {symbol} {tf} ({DAYS_BACK} hari) ===")
            df = fetch_ohlcv(exchange, symbol, tf, DAYS_BACK)
            if df is None or len(df) < 100:
                print(f"  [!] Data tidak cukup untuk {symbol} {tf} (dapat {0 if df is None else len(df)} candle), skip.")
                continue
            print(f"  -> SELESAI: {len(df)} candle didapat ({df['ts'].iloc[0]} s/d {df['ts'].iloc[-1]})")

            df = add_indicators(df)

            tf_trades = run_trend_following(df)
            mr_trades = run_mean_reversion(df)

            m1 = compute_metrics(tf_trades, "TrendFollowing_EMA21_55", symbol, tf)
            m2 = compute_metrics(mr_trades, "MeanReversion_RSI14", symbol, tf)
            all_results.append(m1)
            all_results.append(m2)

            raw_trades[f"{symbol}_{tf}_trend"] = tf_trades
            raw_trades[f"{symbol}_{tf}_meanrev"] = mr_trades

    # simpan hasil
    with open(OUTPUT_JSON, "w") as f:
        json.dump({"summary": all_results, "trades": raw_trades}, f, indent=2, default=str)

    df_summary = pd.DataFrame(all_results)
    df_summary = df_summary.sort_values(["timeframe", "profit_factor"], ascending=[True, False])
    df_summary.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 100)
    print("RINGKASAN HASIL (diurutkan per timeframe, profit factor tertinggi dulu)")
    print("=" * 100)
    print(df_summary.to_string(index=False))
    print(f"\nDisimpan ke: {OUTPUT_CSV} dan {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
