from pathlib import Path
import ccxt
import pandas as pd
import numpy as np

from backtest import fetch_ohlcv_cached, find_ohlcv_gaps

SYMBOL = "BTC/USDT:USDT"
TIMEFRAME = "15m"
DAYS = 365
PIVOT_LEN = 5
ATR_LEN = 14
RR_TARGETS = (2.0, 3.0, 3.5)
ATR_BUFFER = 0.1


def prepare(df):
    df = df.copy().reset_index(drop=True)
    body = (df["close"] - df["open"]).abs()
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.rolling(ATR_LEN).mean()
    df["mean_body"] = body.rolling(PIVOT_LEN).mean()
    df["displacement_up"] = (
        (body > df["mean_body"]) &
        ((df["high"] - df[["close", "open"]].max(axis=1)) < body * 0.36) &
        ((df[["close", "open"]].min(axis=1) - df["low"]) < body * 0.36) &
        (df["close"] > df["open"])
    )
    df["displacement_down"] = (
        (body > df["mean_body"]) &
        ((df["high"] - df[["close", "open"]].max(axis=1)) < body * 0.36) &
        ((df[["close", "open"]].min(axis=1) - df["low"]) < body * 0.36) &
        (df["close"] < df["open"])
    )

    # Pine pivots are confirmed one candle after the pivot candle.
    df["pivot_high"] = np.nan
    df["pivot_low"] = np.nan
    for i in range(PIVOT_LEN, len(df) - 1):
        p = i - 1
        left = df.iloc[p - PIVOT_LEN:p]
        if df.iloc[p]["high"] > left["high"].max():
            df.at[i, "pivot_high"] = df.iloc[p]["high"]
        if df.iloc[p]["low"] < left["low"].min():
            df.at[i, "pivot_low"] = df.iloc[p]["low"]

    return df


def build_pools(df):
    highs, lows = [], []
    buy_pools, sell_pools = {}, {}
    for i, row in df.iterrows():
        atr = row["atr"]
        if not np.isfinite(atr):
            continue
        if np.isfinite(row["pivot_high"]):
            highs.append((i, row["pivot_high"]))
            nearby = [p for _, p in highs[:-1] if abs(p - row["pivot_high"]) <= atr / 2.5]
            if len(nearby) >= 2:
                vals = nearby + [row["pivot_high"]]
                buy_pools[i] = (min(vals) - atr / 2, max(vals) + atr / 2)
        if np.isfinite(row["pivot_low"]):
            lows.append((i, row["pivot_low"]))
            nearby = [p for _, p in lows[:-1] if abs(p - row["pivot_low"]) <= atr / 2.5]
            if len(nearby) >= 2:
                vals = nearby + [row["pivot_low"]]
                sell_pools[i] = (min(vals) - atr / 2, max(vals) + atr / 2)
    return buy_pools, sell_pools


def find_setups(df):
    buy_pools, sell_pools = build_pools(df)
    active_buy, active_sell = [], []
    swings_high, swings_low = [], []
    setups = []
    mss_dir = 0
    state = None

    for i in range(len(df)):
        r = df.iloc[i]

        if i in buy_pools:
            active_buy.append((i, *buy_pools[i]))
        if i in sell_pools:
            active_sell.append((i, *sell_pools[i]))

        if np.isfinite(r["pivot_high"]):
            swings_high.append((i, r["pivot_high"]))
        if np.isfinite(r["pivot_low"]):
            swings_low.append((i, r["pivot_low"]))

        # Sweep above buy-side liquidity / below sell-side liquidity.
        swept_high = next(
            ((idx, lo, hi) for idx, lo, hi in reversed(active_buy)
             if r["high"] > hi and r["close"] < hi), None
        )
        swept_low = next(
            ((idx, lo, hi) for idx, lo, hi in reversed(active_sell)
             if r["low"] < lo and r["close"] > lo), None
        )

        if swept_high:
            state = {"side": -1, "sweep_i": i, "extreme": r["high"]}
        elif swept_low:
            state = {"side": 1, "sweep_i": i, "extreme": r["low"]}

        # MSS: close breaks the latest confirmed opposite swing.
        if len(swings_high) >= 1 and r["close"] > swings_high[-1][1] and mss_dir < 1:
            mss_dir = 1
            if state and state["side"] == 1 and i > state["sweep_i"]:
                state["mss_i"] = i
        elif len(swings_low) >= 1 and r["close"] < swings_low[-1][1] and mss_dir > -1:
            mss_dir = -1
            if state and state["side"] == -1 and i > state["sweep_i"]:
                state["mss_i"] = i

        if not state or "mss_i" not in state or i <= state["mss_i"]:
            continue

        # FVG uses displacement on previous candle and a three-candle gap.
        if state["side"] == 1 and i >= 2:
            if (df.iloc[i - 1]["displacement_up"] and
                    r["low"] > df.iloc[i - 2]["high"]):
                state["fvg"] = (df.iloc[i - 2]["high"], r["low"])
                state["fvg_i"] = i
        elif state["side"] == -1 and i >= 2:
            if (df.iloc[i - 1]["displacement_down"] and
                    r["high"] < df.iloc[i - 2]["low"]):
                state["fvg"] = (r["high"], df.iloc[i - 2]["low"])
                state["fvg_i"] = i

        if "fvg" not in state or i <= state["fvg_i"]:
            continue

        bottom, top = state["fvg"]
        if state["side"] == 1:
            if r["low"] < bottom:
                state = None
                continue
            if r["low"] <= top and r["high"] >= bottom:
                entry = top
                sl = state["extreme"] - ATR_BUFFER * r["atr"]
                risk = entry - sl
                if risk > 0:
                    setups.append((i, 1, entry, sl, risk))
                state = None
        else:
            if r["high"] > top:
                state = None
                continue
            if r["high"] >= bottom and r["low"] <= top:
                entry = bottom
                sl = state["extreme"] + ATR_BUFFER * r["atr"]
                risk = sl - entry
                if risk > 0:
                    setups.append((i, -1, entry, sl, risk))
                state = None

    return setups


def backtest(df, setups, rr):
    trades = []
    for start, side, entry, sl, risk in setups:
        tp = entry + side * risk * rr
        result = None
        exit_i = None

        for j in range(start + 1, len(df)):
            row = df.iloc[j]
            if side == 1:
                if row["low"] <= sl:
                    result, exit_i = -1, j
                    break
                if row["high"] >= tp:
                    result, exit_i = rr, j
                    break
            else:
                if row["high"] >= sl:
                    result, exit_i = -1, j
                    break
                if row["low"] <= tp:
                    result, exit_i = rr, j
                    break

        if result is not None:
            trades.append({
                "entry_time": df.iloc[start]["timestamp"],
                "exit_time": df.iloc[exit_i]["timestamp"],
                "side": "long" if side == 1 else "short",
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "result_R": result
            })

    return pd.DataFrame(trades)


def main():
    exchange = ccxt.bitget({"enableRateLimit": True})
    exchange.options["defaultType"] = "swap"
    df = fetch_ohlcv_cached(exchange, SYMBOL, TIMEFRAME, DAYS)
    if df is None or df.empty:
        raise RuntimeError("OHLCV kosong")
    gaps = find_ohlcv_gaps(df, TIMEFRAME)
    print(f"Data: {len(df)} candles | gaps: {len(gaps)}")

    df = prepare(df)
    setups = find_setups(df)
    print(f"Setups: {len(setups)}")

    summaries = []
    all_trades = []
    for rr in RR_TARGETS:
        trades = backtest(df, setups, rr)
        trades["rr_target"] = rr
        all_trades.append(trades)
        if trades.empty:
            summaries.append({"RR": rr, "trades": 0, "win_rate": 0,
                              "PF": 0, "total_R": 0})
            continue
        wins = trades.loc[trades.result_R > 0, "result_R"].sum()
        losses = -trades.loc[trades.result_R < 0, "result_R"].sum()
        summaries.append({
            "RR": rr,
            "trades": len(trades),
            "win_rate": round((trades.result_R > 0).mean() * 100, 2),
            "PF": round(wins / losses, 3) if losses else float("inf"),
            "total_R": round(trades.result_R.sum(), 2)
        })

    pd.DataFrame(summaries).to_csv("ict_summary.csv", index=False)
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv("ict_trades.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
