#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd

from backtest import PAIRS, fetch_ohlcv_cached, find_ohlcv_gaps, MAX_GAP_CANDLES

TIMEFRAME = "2h"
DAYS_BACK = 730

TENKAN = 9
KIJUN = 26
CHIKOU = 52
DISPLACEMENT = 26

OOS_PF_MIN = 1.5
OOS_MIN_TRADES = 30

TRADE_COLUMNS = [
    "symbol", "side", "entry_ts", "exit_ts",
    "entry_price", "exit_price", "return_pct", "exit_reason"
]


def ichimoku_signals(df):
    high = df["high"]
    low = df["low"]

    def donchian(length):
        return (
            high.rolling(length).max()
            + low.rolling(length).min()
        ) / 2

    tenkan = donchian(TENKAN)
    kijun = donchian(KIJUN)

    senkou_a = (tenkan + kijun) / 2
    senkou_b = donchian(CHIKOU)

    # Pine: senkouA[displacement - 1]
    cloud_a = senkou_a.shift(DISPLACEMENT - 1)
    cloud_b = senkou_b.shift(DISPLACEMENT - 1)

    cross_up = (
        (tenkan > kijun)
        & (tenkan.shift(1) <= kijun.shift(1))
    )

    cross_down = (
        (tenkan < kijun)
        & (tenkan.shift(1) >= kijun.shift(1))
    )

    strong_bull = (tenkan > cloud_a) & (tenkan > cloud_b)
    neutral_bull = (
        ((tenkan > cloud_a) & (tenkan < cloud_b))
        | ((tenkan < cloud_a) & (tenkan > cloud_b))
    )
    weak_bull = (tenkan < cloud_a) & (tenkan < cloud_b)

    strong_bear = (tenkan < cloud_a) & (tenkan < cloud_b)
    neutral_bear = (
        ((tenkan > cloud_a) & (tenkan < cloud_b))
        | ((tenkan < cloud_a) & (tenkan > cloud_b))
    )
    weak_bear = (tenkan > cloud_a) & (tenkan > cloud_b)

    bullish = cross_up & (
        strong_bull | neutral_bull | weak_bull
    )
    bearish = cross_down & (
        strong_bear | neutral_bear | weak_bear
    )

    return (
        bullish.fillna(False).to_numpy(),
        bearish.fillna(False).to_numpy()
    )


def close_trade(pos, ts, price, reason):
    sign = 1 if pos["side"] == "long" else -1
    ret = sign * (price - pos["entry_price"]) / pos["entry_price"]

    return {
        "side": pos["side"],
        "entry_ts": str(pos["entry_ts"]),
        "exit_ts": str(ts),
        "entry_price": float(pos["entry_price"]),
        "exit_price": float(price),
        "return_pct": ret * 100,
        "exit_reason": reason,
    }


def simulate(df, bullish, bearish):
    opens = df["open"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    timestamps = df["ts"].tolist()

    trades = []
    pos = None
    pending = None

    for i in range(1, len(df)):
        # Eksekusi sinyal candle sebelumnya pada open candle sekarang.
        if pending is not None:
            if pos is not None and pos["side"] != pending:
                trades.append(
                    close_trade(pos, timestamps[i], opens[i], "reverse")
                )
                pos = None

            if pos is None:
                pos = {
                    "side": pending,
                    "entry_ts": timestamps[i],
                    "entry_price": opens[i],
                }

            pending = None

        # Sinyal baru diketahui setelah candle sekarang selesai.
        if bullish[i]:
            pending = "long"
        elif bearish[i]:
            pending = "short"

    if pos is not None:
        trades.append(
            close_trade(pos, timestamps[-1], closes[-1], "end_data")
        )

    return trades


def profit_factor(returns):
    wins = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())

    if losses == 0:
        return float("inf") if wins > 0 else None

    return float(wins / losses)


def metrics(trades, symbol, period):
    returns = np.array(
        [t["return_pct"] / 100 for t in trades],
        dtype=float
    )

    if len(returns) == 0:
        return {
            "strategy": "Ichimoku_Clouds",
            "symbol": symbol,
            "timeframe": TIMEFRAME,
            "mode": "long_short",
            "stop_loss": "disabled",
            "take_profit": "disabled",
            "period": period,
            "n_trades": 0,
            "total_return_pct": 0,
            "win_rate_pct": 0,
            "profit_factor": None,
            "max_drawdown_pct": 0,
            "avg_pnl_pct": 0,
        }

    equity = np.r_[1.0, np.cumprod(1 + returns)]
    peak = np.maximum.accumulate(equity)
    drawdown = (equity / peak - 1).min()

    return {
        "strategy": "Ichimoku_Clouds",
        "symbol": symbol,
        "timeframe": TIMEFRAME,
        "mode": "long_short",
        "stop_loss": "disabled",
        "take_profit": "disabled",
        "period": period,
        "n_trades": len(returns),
        "total_return_pct": round((equity[-1] - 1) * 100, 2),
        "win_rate_pct": round(float((returns > 0).mean() * 100), 2),
        "profit_factor": (
            round(profit_factor(returns), 2)
            if profit_factor(returns) is not None
            and np.isfinite(profit_factor(returns))
            else "inf" if profit_factor(returns) == float("inf")
            else None
        ),
        "max_drawdown_pct": round(float(drawdown * 100), 2),
        "avg_pnl_pct": round(float(returns.mean() * 100), 3),
    }


def evaluate(rows):
    results = []

    for symbol in PAIRS:
        is_row = next(
            (r for r in rows if r["symbol"] == symbol and r["period"] == "IS"),
            None
        )
        oos_row = next(
            (r for r in rows if r["symbol"] == symbol and r["period"] == "OOS"),
            None
        )

        if is_row is None or oos_row is None:
            continue

        is_pf = is_row["profit_factor"]
        oos_pf = oos_row["profit_factor"]

        if is_pf == "inf":
            is_pf = float("inf")
        if oos_pf == "inf":
            oos_pf = float("inf")

        passed = (
            isinstance(is_pf, (int, float))
            and is_pf > 1
            and isinstance(oos_pf, (int, float))
            and oos_pf > OOS_PF_MIN
            and oos_row["n_trades"] >= OOS_MIN_TRADES
        )

        results.append({
            "symbol": symbol,
            "pf_is": is_pf,
            "pf_oos": oos_pf,
            "trades_oos": oos_row["n_trades"],
            "return_oos": oos_row["total_return_pct"],
            "dd_oos": oos_row["max_drawdown_pct"],
            "pass": passed,
        })

    passed = sum(r["pass"] for r in results)
    total = len(results)
    overall = total > 0 and passed > total / 2

    lines = [
        "# Ichimoku Clouds — Evaluasi",
        "",
        f"Pair lulus: {passed}/{total}",
        f"Status mayoritas: {'LULUS' if overall else 'TIDAK LULUS'}",
        "",
        f"Kriteria: OOS PF > {OOS_PF_MIN}, OOS trade >= {OOS_MIN_TRADES}, IS PF > 1.",
        "",
        "| Pair | PF IS | PF OOS | Trade OOS | Return OOS | DD OOS | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    for r in results:
        lines.append(
            f"| {r['symbol']} | {r['pf_is']} | {r['pf_oos']} "
            f"| {r['trades_oos']} | {r['return_oos']}% "
            f"| {r['dd_oos']}% | {'LULUS' if r['pass'] else 'GAGAL'} |"
        )

    report = "\n".join(lines) + "\n"
    print(report)

    with open("ichimoku_evaluation.md", "w", encoding="utf-8") as f:
        f.write(report)

    return results


def main():
    import ccxt

    exchange = ccxt.bitget({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })

    all_summary = []
    all_trades = []

    for symbol in PAIRS:
        print(f"\n=== Ichimoku {symbol} {TIMEFRAME} ===")

        df = fetch_ohlcv_cached(
            exchange, symbol, TIMEFRAME, DAYS_BACK
        )

        if df is None or len(df) < 200:
            print("  [!] Data tidak cukup, skip.")
            continue

        # Abaikan candle 2h yang belum selesai.
        now = pd.Timestamp.now(tz="UTC")
        df = df[
            df["ts"] + pd.Timedelta(TIMEFRAME) <= now
        ].reset_index(drop=True)

        if len(df) < 200:
            print("  [!] Candle tertutup tidak cukup, skip.")
            continue

        gaps = find_ohlcv_gaps(df, TIMEFRAME)
        if any(g[2] > MAX_GAP_CANDLES for g in gaps):
            print("  [!] Gap data melebihi batas, skip.")
            continue

        split_ts = (
            df["ts"].iloc[0]
            + (df["ts"].iloc[-1] - df["ts"].iloc[0]) / 2
        )

        bullish, bearish = ichimoku_signals(df)
        trades = simulate(df, bullish, bearish)

        for trade in trades:
            trade["symbol"] = symbol
            all_trades.append(trade)

        for period in ("full", "IS", "OOS"):
            if period == "full":
                selected = trades
            elif period == "IS":
                selected = [
                    t for t in trades
                    if pd.Timestamp(t["entry_ts"]) < split_ts
                ]
            else:
                selected = [
                    t for t in trades
                    if pd.Timestamp(t["entry_ts"]) >= split_ts
                ]

            all_summary.append(metrics(selected, symbol, period))

        print(
            f"  Candle: {len(df)} | Sinyal long: {bullish.sum()} "
            f"| Sinyal short: {bearish.sum()} | Trade: {len(trades)}"
        )

    if not all_summary:
        raise SystemExit("Tidak ada hasil backtest.")

    pd.DataFrame(all_summary).to_csv(
        "ichimoku_summary.csv", index=False
    )

    pd.DataFrame(
        all_trades, columns=TRADE_COLUMNS
    ).to_csv("ichimoku_trades.csv", index=False)

    with open("ichimoku_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {"summary": all_summary},
            f,
            indent=2,
            default=str,
            allow_nan=False
        )

    evaluate(all_summary)


if __name__ == "__main__":
    main()
