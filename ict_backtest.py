from pathlib import Path
import ccxt
import pandas as pd
import numpy as np

from backtest import fetch_ohlcv_cached, find_ohlcv_gaps

SYMBOL = "BTC/USDT:USDT"
TIMEFRAME = "15m"
DAYS = 365
PIVOT_LEN = 5
ATR_LEN = 10
RR_TARGETS = (2.0, 3.0, 3.5)
ATR_BUFFER = 0.1


def pine_rma(series, length):
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) >= length:
        seed = values[:length]
        if np.isfinite(seed).all():
            out[length - 1] = seed.mean()
            for i in range(length, len(values)):
                out[i] = (out[i - 1] * (length - 1) + values[i]) / length
    return pd.Series(out, index=series.index)


def prepare(df):
    df = df.copy().reset_index(drop=True)
    body = (df["close"] - df["open"]).abs()
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)

    df["atr"] = pine_rma(tr, ATR_LEN)
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
        if df.iloc[p]["high"] > left["high"].max() and df.iloc[p]["high"] > df.iloc[i]["high"]:
            df.at[i, "pivot_high"] = df.iloc[p]["high"]
        if df.iloc[p]["low"] < left["low"].min() and df.iloc[p]["low"] < df.iloc[i]["low"]:
            df.at[i, "pivot_low"] = df.iloc[p]["low"]

    return df


def build_pools(df):
    # ZigZag LuxAlgo: array terbaru berada di index 0, maksimum 50 pivot.
    zz_d, zz_x, zz_y = [], [], []
    buy_pools, sell_pools, mss_events = {}, {}, {}
    mss_dir = 0

    for i, row in df.iterrows():
        atr = row["atr"]
        ph = row["pivot_high"]
        pl = row["pivot_low"]

        if np.isfinite(ph):
            direction = zz_d[0] if zz_d else 0
            if direction < 1:
                zz_d.insert(0, 1)
                zz_x.insert(0, i - 1)
                zz_y.insert(0, float(ph))
            elif direction == 1 and ph > zz_y[0]:
                zz_x[0] = i - 1
                zz_y[0] = float(ph)

            del zz_d[50:]
            del zz_x[50:]
            del zz_y[50:]

            # LuxAlgo liquidity clustering: pivot searah dalam jarak ATR / 2.5.
            if np.isfinite(atr):
                tolerance = atr / 2.5
                clustered = []
                for d, x, y in zip(zz_d, zz_x, zz_y):
                    if d != 1:
                        continue
                    if y > ph + tolerance:
                        break
                    if ph - tolerance < y < ph + tolerance:
                        clustered.append(y)

                if len(clustered) > 2:
                    center = (min(clustered) + max(clustered)) / 2
                    buy_pools[i] = (
                        center - tolerance,
                        center + tolerance
                    )

        if np.isfinite(pl):
            direction = zz_d[0] if zz_d else 0
            if direction > -1:
                zz_d.insert(0, -1)
                zz_x.insert(0, i - 1)
                zz_y.insert(0, float(pl))
            elif direction == -1 and pl < zz_y[0]:
                zz_x[0] = i - 1
                zz_y[0] = float(pl)

            del zz_d[50:]
            del zz_x[50:]
            del zz_y[50:]

            if np.isfinite(atr):
                tolerance = atr / 2.5
                clustered = []
                for d, x, y in zip(zz_d, zz_x, zz_y):
                    if d != -1:
                        continue
                    if y < pl - tolerance:
                        break
                    if pl - tolerance < y < pl + tolerance:
                        clustered.append(y)

                if len(clustered) > 2:
                    center = (min(clustered) + max(clustered)) / 2
                    sell_pools[i] = (
                        center - tolerance,
                        center + tolerance
                    )

        # MSS LuxAlgo: gunakan pivot ZigZag pada index 1 atau 2.
        if len(zz_d) >= 2:
            i_h = 2 if len(zz_d) > 2 and zz_d[2] == 1 else 1
            i_l = 2 if len(zz_d) > 2 and zz_d[2] == -1 else 1

            if (len(zz_d) > i_h and zz_d[i_h] == 1
                    and row["close"] > zz_y[i_h] and mss_dir < 1):
                mss_dir = 1
                mss_events[i] = 1

            elif (len(zz_d) > i_l and zz_d[i_l] == -1
                    and row["close"] < zz_y[i_l] and mss_dir > -1):
                mss_dir = -1
                mss_events[i] = -1

    return buy_pools, sell_pools, mss_events


def find_setups(df):
    buy_pools, sell_pools, mss_events = build_pools(df)
    active_buy, active_sell = [], []
    setups = []
    state = None

    for i, row in df.iterrows():
        if i in buy_pools:
            active_buy.append((i, *buy_pools[i]))
        if i in sell_pools:
            active_sell.append((i, *sell_pools[i]))

        # Sweep buy-side liquidity: high melewati zona lalu close kembali di bawah.
        swept_high = next(
            ((idx, lo, hi) for idx, lo, hi in reversed(active_buy)
             if row["high"] > hi and row["close"] < hi),
            None
        )
        # Sweep sell-side liquidity: low melewati zona lalu close kembali di atas.
        swept_low = next(
            ((idx, lo, hi) for idx, lo, hi in reversed(active_sell)
             if row["low"] < lo and row["close"] > lo),
            None
        )

        if swept_high:
            state = {"side": -1, "sweep_i": i, "extreme": row["high"]}
            active_buy = [p for p in active_buy if p[0] != swept_high[0]]
        elif swept_low:
            state = {"side": 1, "sweep_i": i, "extreme": row["low"]}
            active_sell = [p for p in active_sell if p[0] != swept_low[0]]

        # MSS harus terjadi setelah sweep dan searah pembalikan.
        mss = mss_events.get(i)
        if state and i > state["sweep_i"]:
            if mss == state["side"]:
                state["mss_i"] = i

        if not state or "mss_i" not in state or i <= state["mss_i"]:
            continue

        # FVG: displacement pada candle sebelumnya dan gap tiga candle.
        if state["side"] == 1 and i >= 2:
            if (df.iloc[i - 1]["displacement_up"]
                    and row["low"] > df.iloc[i - 2]["high"]):
                state["fvg"] = (
                    df.iloc[i - 2]["high"],
                    row["low"]
                )
                state["fvg_i"] = i

        elif state["side"] == -1 and i >= 2:
            if (df.iloc[i - 1]["displacement_down"]
                    and row["high"] < df.iloc[i - 2]["low"]):
                state["fvg"] = (
                    row["high"],
                    df.iloc[i - 2]["low"]
                )
                state["fvg_i"] = i

        if "fvg" not in state or i <= state["fvg_i"]:
            continue

        bottom, top = state["fvg"]

        if state["side"] == 1:
            if row["low"] < bottom:
                state = None
                continue
            if row["low"] <= top and row["high"] >= bottom:
                entry = top
                sl = state["extreme"] - ATR_BUFFER * row["atr"]
                risk = entry - sl
                if risk > 0:
                    setups.append((i, 1, entry, sl, risk))
                state = None

        else:
            if row["high"] > bottom:
                state = None
                continue
            if row["high"] >= top and row["low"] <= bottom:
                entry = top
                sl = state["extreme"] + ATR_BUFFER * row["atr"]
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
                "entry_time": df.iloc[start]["ts"],
                "exit_time": df.iloc[exit_i]["ts"],
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
