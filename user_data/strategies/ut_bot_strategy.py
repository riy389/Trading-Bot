# UT Bot Alerts port — Pine v4 logic (key=a, ATR period=c, Heikin Ashi h=false)
# Filter: EMA200 self-timeframe (ikut TF entry masing-masing, bukan dipaku 4h)
# Exit: garis trailing stop UT sendiri (tanpa TP tetap, tanpa BEP/lock)
# Broaden: 3 TF (15m/1h/4h) x A(4,5,6,7) = 12 class, dijalankan terhadap
# pair_whitelist BTC/ETH/SOL sekaligus lewat config.
# Catatan: Freqtrade proses exit lalu entry per-candle terpisah (bukan reverse
# instan satu candle seperti Pine stop-and-reverse). Efeknya hasil bisa geser
# tipis vs mesin lama (backtest.py) — dicatat untuk pembacaan hasil nanti.

import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy


def _ut_bot(df: DataFrame, a: float, c: int) -> DataFrame:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / c, adjust=False).mean()
    n_loss = a * atr

    src = close.values
    nloss = n_loss.values
    stop = np.zeros(len(df))
    stop[0] = src[0] - nloss[0]

    for i in range(1, len(df)):
        prev_stop = stop[i - 1]
        if src[i] > prev_stop and src[i - 1] > prev_stop:
            stop[i] = max(prev_stop, src[i] - nloss[i])
        elif src[i] < prev_stop and src[i - 1] < prev_stop:
            stop[i] = min(prev_stop, src[i] + nloss[i])
        elif src[i] > prev_stop:
            stop[i] = src[i] - nloss[i]
        else:
            stop[i] = src[i] + nloss[i]

    df["ut_stop"] = stop
    df["ut_buy"] = (close > df["ut_stop"]) & (close.shift() <= df["ut_stop"].shift())
    df["ut_sell"] = (close < df["ut_stop"]) & (close.shift() >= df["ut_stop"].shift())
    df["ut_long_side"] = close > df["ut_stop"]
    return df


class _UTBotBase(IStrategy):
    can_short = True
    UT_A = 6
    UT_C = 10
    minimal_roi = {"0": 100}
    stoploss = -0.99  # exit dikendalikan logic UT, bukan stoploss tetap
    use_exit_signal = True
    exit_profit_only = False

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df = _ut_bot(df, a=self.UT_A, c=self.UT_C)
        df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[df["ut_buy"] & (df["close"] > df["ema200"]), "enter_long"] = 1
        df.loc[df["ut_sell"] & (df["close"] < df["ema200"]), "enter_short"] = 1
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[~df["ut_long_side"], "exit_long"] = 1
        df.loc[df["ut_long_side"], "exit_short"] = 1
        return df


# ---- 15m ----
class UTBotA4_15m(_UTBotBase):
    timeframe = "15m"
    UT_A = 4


class UTBotA5_15m(_UTBotBase):
    timeframe = "15m"
    UT_A = 5


class UTBotA6_15m(_UTBotBase):
    timeframe = "15m"
    UT_A = 6


class UTBotA7_15m(_UTBotBase):
    timeframe = "15m"
    UT_A = 7


# ---- 1h ----
class UTBotA4_1h(_UTBotBase):
    timeframe = "1h"
    UT_A = 4


class UTBotA5_1h(_UTBotBase):
    timeframe = "1h"
    UT_A = 5


class UTBotA6_1h(_UTBotBase):
    timeframe = "1h"
    UT_A = 6


class UTBotA7_1h(_UTBotBase):
    timeframe = "1h"
    UT_A = 7


# ---- 4h ----
class UTBotA4_4h(_UTBotBase):
    timeframe = "4h"
    UT_A = 4


class UTBotA5_4h(_UTBotBase):
    timeframe = "4h"
    UT_A = 5


class UTBotA6_4h(_UTBotBase):
    timeframe = "4h"
    UT_A = 6


class UTBotA7_4h(_UTBotBase):
    timeframe = "4h"
    UT_A = 7
