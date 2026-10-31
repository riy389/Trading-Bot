#!/usr/bin/env python3
"""
ETH Momentum Breakout v15 — Python backtest for Bitget USDT-M perpetual.
Uses backtest.py's existing Bitget OHLCV fetcher/cache; native 2h candles.
Period: 2020-10-01 through 2026-02-28 (UTC).

Signal at candle close, entry at next candle open.
Initial SL/TP anchored to signal candle close.
Intrabar SL is checked before TP.
BE/trailing updates at candle close apply from next candle.
This is an OHLC approximation, not a tick replay.
"""

import json
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

from backtest import fetch_ohlcv_cached, find_ohlcv_gaps

SYMBOL = "ETH/USDT:USDT"
TIMEFRAME = "2h"
DAYS_BACK = 2300
START = pd.Timestamp("2020-10-01 00:00:00", tz="UTC")
END = pd.Timestamp("2026-03-01 00:00:00", tz="UTC")

BB_LEN, BB_MULT = 20, 2.0
KC_LEN, KC_MULT = 20, 1.5
SQZ_MIN_BARS, SQZ_MAX_BARS = 3, 6
BREAKOUT_ATR, MIN_BODY_PCT = 0.3, 40.0
RSI_LEN, RSI_LONG_MAX = 14, 75.0
VOL_LEN, VOL_MULT = 20, 1.8
MOM_LEN = 12
ATR_LEN = 14
SL_MULT, TP2_MULT, TRAIL_MULT = 2.0, 3.5, 2.5
BE_TRIGGER, BE_OFFSET = 1.2, 0.2
COOLDOWN, MAX_BARS = 12, 100
STAGN_HOT, STAGN_NORMAL, STAGN_COLD, STAGN_ATR = 15, 10, 7, 0.5
EMA_DIST_MAX = 35.0
DAILY_VOL_MIN, DAILY_VOL_MAX = 0.3, 1.5
ADAPT_VOL_LEN, ADAPT_EMA_LEN = 100, 50
HOT_VOL, HOT_TREND, COLD_VOL, COLD_TREND = 60.0, 1.5, 30.0, 0.5
HOT_SCALE, COLD_SCALE = 1.35, 0.75


def rma(s, n):
    """Wilder RMA seeded with first n-value SMA."""
    x = s.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    valid = np.where(np.isfinite(x))[0]
    if len(valid) < n:
        return pd.Series(out, index=s.index)
    first = valid[0]
    if first + n > len(x) or not np.isfinite(x[first:first + n]).all():
        return pd.Series(out, index=s.index)
    out[first + n - 1] = x[first:first + n].mean()
    for i in range(first + n, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return pd.Series(out, index=s.index)


def atr(df, n):
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs()
    ], axis=1).max(axis=1)
    return rma(tr, n)


def rsi(close, n):
    delta = close.diff()
    up = rma(delta.clip(lower=0), n)
    down = rma((-delta).clip(lower=0), n)
    rs = up / down.replace(0, np.nan)
    result = 100 - 100 / (1 + rs)
    result = result.where(down != 0, 100)
    result = result.where(up != 0, 0)
    return result.fillna(50)


def linreg_endpoint(series, n):
    """Pine ta.linreg(source, length, 0): fitted value at last x."""
    x = np.arange(n, dtype=float)
    xm = x.mean()
    denominator = ((x - xm) ** 2).sum()

    def calc(values):
        if not np.isfinite(values).all():
            return np.nan
        slope = ((x - xm) * (values - values.mean())).sum() / denominator
        return values.mean() + slope * (n - 1 - xm)

    return series.rolling(n, min_periods=n).apply(calc, raw=True)


def daily_regime(df):
    """Use completed UTC daily candles only."""
    daily_source = df.set_index("ts")[
        ["open", "high", "low", "close", "volume"]
    ]

    daily = daily_source.resample(
        "1D", label="left", closed="left"
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()

    daily_atr = atr(daily, 14)
    daily_ema = daily["close"].ewm(
        span=200, adjust=False, min_periods=200
    ).mean()
    daily_vol_ratio = daily_atr / daily_atr.rolling(
        100, min_periods=100
    ).mean()

    daily_features = pd.DataFrame({
        "available_at": daily.index + pd.Timedelta(days=1),
        "daily_ema200": daily_ema.to_numpy(),
        "daily_vol_ratio": daily_vol_ratio.to_numpy()
    }).dropna(subset=["available_at"])

    base = df[["ts", "close"]].sort_values("ts")
    merged = pd.merge_asof(
        base,
        daily_features.sort_values("available_at"),
        left_on="ts",
        right_on="available_at",
        direction="backward"
    )

    return (
        (merged["daily_ema200"] > 0)
        & (
            (merged["close"] - merged["daily_ema200"]).abs()
            / merged["daily_ema200"] * 100 <= EMA_DIST_MAX
        )
        & merged["daily_vol_ratio"].between(
            DAILY_VOL_MIN, DAILY_VOL_MAX
        )
    ).to_numpy()


def build_signals(df):
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    atr_values = atr(df, ATR_LEN)

    bb_mid = close.rolling(BB_LEN).mean()
    bb_dev = close.rolling(BB_LEN).std(ddof=0) * BB_MULT
    bb_upper = bb_mid + bb_dev
    bb_lower = bb_mid - bb_dev

    kc_mid = close.rolling(KC_LEN).mean()
    kc_range = atr(df, KC_LEN) * KC_MULT
    squeeze = (
        (bb_lower > kc_mid - kc_range)
        & (bb_upper < kc_mid + kc_range)
    )

    recent_squeeze = np.zeros(len(df), dtype=bool)
    duration = 0
    bars_since = 999
    last_valid = False

    for i, is_squeeze in enumerate(squeeze.fillna(False).to_numpy()):
        previous_duration = duration

        if is_squeeze:
            duration += 1
            bars_since = 0
        else:
            if bars_since == 0:
                last_valid = previous_duration >= SQZ_MIN_BARS
            bars_since += 1
            duration = 0

        recent_squeeze[i] = (
            0 < bars_since <= SQZ_MAX_BARS and last_valid
        )

    momentum = linreg_endpoint(
        close - close.rolling(MOM_LEN).mean(), MOM_LEN
    )
    rsi_values = rsi(close, RSI_LEN)

    volume_ok = volume > volume.rolling(VOL_LEN).mean() * VOL_MULT
    body_pct = (
        (open_ - close).abs()
        / (high - low).replace(0, np.nan) * 100
    )
    breakout = close > bb_upper + atr_values * BREAKOUT_ATR
    momentum_up = (momentum > 0) & (momentum > momentum.shift(1))
    atr_pct_ok = atr_values / close * 100 >= 0.3

    vol_rank = atr_values.rolling(
        ADAPT_VOL_LEN, min_periods=ADAPT_VOL_LEN
    ).apply(
        lambda values: (
            100.0 * np.count_nonzero(values <= values[-1]) / len(values)
        ),
        raw=True
    )

    adaptive_ema = close.ewm(
        span=ADAPT_EMA_LEN, adjust=False,
        min_periods=ADAPT_EMA_LEN
    ).mean()

    trend_score = (close - adaptive_ema) / atr_values
    hot = (vol_rank > HOT_VOL) & (trend_score > HOT_TREND)
    cold = (vol_rank < COLD_VOL) | (trend_score < COLD_TREND)
    regime = np.where(hot, 1, np.where(cold, -1, 0))

    daily_ok = daily_regime(df)

    signal = (
        recent_squeeze
        & momentum_up
        & volume_ok
        & breakout
        & (body_pct >= MIN_BODY_PCT)
        & (rsi_values < RSI_LONG_MAX)
        & atr_pct_ok
        & daily_ok
    )

    return atr_values.to_numpy(), signal.fillna(False).to_numpy(), regime


def run_backtest(df):
    atr_values, signals, regimes = build_signals(df)

    opens = df["open"].to_numpy(float)
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    closes = df["close"].to_numpy(float)
    timestamps = df["ts"].astype(str).to_numpy()

    trades = []
    position = None
    pending_entry = False
    bars_since_exit = 999

    for i in range(1, len(df)):
        closed_this_bar = False

        # Entry signal at previous close, fill at current open.
        if pending_entry and position is None and np.isfinite(atr_values[i - 1]):
            scale = (
                HOT_SCALE if regimes[i - 1] == 1
                else COLD_SCALE if regimes[i - 1] == -1
                else 1.0
            )
            entry_reference = closes[i - 1]
            entry_atr = atr_values[i - 1]

            position = {
                "signal_ts": timestamps[i - 1],
                "entry_ts": timestamps[i],
                "entry": opens[i],
                "entry_reference": entry_reference,
                "entry_atr": entry_atr,
                "sl": entry_reference - entry_atr * SL_MULT * scale,
                "tp": entry_reference + entry_atr * TP2_MULT * scale,
                "trail_mult": TRAIL_MULT * scale,
                "scale": scale,
                "stagn_bars": (
                    STAGN_HOT if regimes[i - 1] == 1
                    else STAGN_COLD if regimes[i - 1] == -1
                    else STAGN_NORMAL
                ),
                "bars": 0,
                "mfe_close": 0.0,
                "be": False,
                "regime": int(regimes[i - 1])
            }
            pending_entry = False

        if position is not None:
            stop = position["sl"]
            exit_price = None
            exit_reason = None

            # Conservative priority: SL before TP.
            if lows[i] <= stop:
                exit_price = min(opens[i], stop)
                exit_reason = "stop_loss"
            elif highs[i] >= position["tp"]:
                exit_price = max(opens[i], position["tp"])
                exit_reason = "take_profit"

            if exit_reason:
                pnl = exit_price / position["entry"] - 1
                trades.append({
                    **position,
                    "exit_ts": timestamps[i],
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "return_pct": pnl * 100
                })
                position = None
                bars_since_exit = 0
                closed_this_bar = True
            else:
                position["bars"] += 1
                position["mfe_close"] = max(
                    position["mfe_close"],
                    closes[i] - position["entry"]
                )

                if (
                    not position["be"]
                    and closes[i] - position["entry"]
                    >= position["entry_atr"] * BE_TRIGGER
                ):
                    position["be"] = True
                    position["sl"] = max(
                        position["sl"],
                        position["entry"] + position["entry_atr"] * BE_OFFSET
                    )

                if np.isfinite(atr_values[i]):
                    position["sl"] = max(
                        position["sl"],
                        closes[i] - atr_values[i] * position["trail_mult"]
                    )

                if position["bars"] >= MAX_BARS:
                    pending_exit = "timeout"
                elif (
                    position["bars"] >= position["stagn_bars"]
                    and position["mfe_close"]
                    < position["entry_atr"] * STAGN_ATR
                ):
                    pending_exit = "stagnation"
                else:
                    pending_exit = None

                if pending_exit and i + 1 < len(df):
                    next_open = opens[i + 1]
                    trades.append({
                        **position,
                        "exit_ts": timestamps[i + 1],
                        "exit_price": next_open,
                        "exit_reason": pending_exit,
                        "return_pct": (
                            next_open / position["entry"] - 1
                        ) * 100
                    })
                    position = None
                    bars_since_exit = 0
                    closed_this_bar = True

        if position is None:
            if not closed_this_bar:
                bars_since_exit += 1

            if signals[i] and bars_since_exit >= COOLDOWN:
                pending_entry = True

    if position is not None:
        trades.append({
            **position,
            "exit_ts": timestamps[-1],
            "exit_price": closes[-1],
            "exit_reason": "end_data",
            "return_pct": (
                closes[-1] / position["entry"] - 1
            ) * 100
        })

    return trades


def main():
    print(f"Fetching {SYMBOL} native {TIMEFRAME} candles from Bitget...")

    exchange = ccxt.bitget({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"}
    })
    exchange.load_markets()

    raw = fetch_ohlcv_cached(
        exchange, SYMBOL, TIMEFRAME, DAYS_BACK
    )

    if raw is None or raw.empty:
        raise SystemExit("ERROR: Bitget returned no candles.")

    df = raw[
        (raw["ts"] >= START) & (raw["ts"] < END)
    ].copy().reset_index(drop=True)

    if df.empty:
        raise SystemExit("ERROR: no candles in requested period.")

    print(
        f"Actual data: {df.ts.iloc[0]} -> {df.ts.iloc[-1]} "
        f"({len(df)} candles)"
    )

    gaps = [
        gap for gap in find_ohlcv_gaps(df, TIMEFRAME)
        if gap[2] > 3
    ]

    print(f"Large gaps (>3 candles): {len(gaps)}")

    if gaps:
        for gap in gaps[:10]:
            print(
                f"  GAP {gap[0]} -> {gap[1]}: "
                f"{gap[2]} candles missing"
            )
        raise SystemExit(
            "ERROR: data gaps exceed tolerance; refusing to publish result."
        )

    trades = run_backtest(df)
    trade_df = pd.DataFrame(trades)
    trade_df.to_csv("eth_momentum_trades.csv", index=False)

    if len(trade_df):
        wins = trade_df[trade_df.return_pct > 0]
        losses = trade_df[trade_df.return_pct <= 0]

        gross_profit = wins.return_pct.sum()
        gross_loss = abs(losses.return_pct.sum())

        equity = (1 + trade_df.return_pct / 100).cumprod()
        drawdown = equity / equity.cummax() - 1

        summary = {
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,
            "period_start": str(df.ts.iloc[0]),
            "period_end": str(df.ts.iloc[-1]),
            "candles": int(len(df)),
            "trades": int(len(trade_df)),
            "win_rate_pct": round(
                len(wins) / len(trade_df) * 100, 2
            ),
            "profit_factor": (
                round(gross_profit / gross_loss, 4)
                if gross_loss else None
            ),
            "total_return_pct_compounded": round(
                (equity.iloc[-1] - 1) * 100, 2
            ),
            "max_drawdown_pct_trade_equity": round(
                drawdown.min() * 100, 2
            ),
            "exit_reasons": trade_df.exit_reason.value_counts().to_dict(),
            "data_gaps_over_3_candles": len(gaps),
            "note": (
                "OHLC approximation, not tick replay. "
                "Intrabar SL is prioritized over TP."
            )
        }
    else:
        summary = {
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,
            "trades": 0,
            "period_start": str(df.ts.iloc[0]),
            "period_end": str(df.ts.iloc[-1])
        }

    Path("eth_momentum_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(json.dumps(summary, indent=2))
    print(
        "Saved: eth_momentum_trades.csv, "
        "eth_momentum_summary.json"
    )


if __name__ == "__main__":
    main()
