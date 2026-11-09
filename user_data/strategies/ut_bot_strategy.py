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


# ---- Analisis fitur sinyal UT Bot (tanpa filter). Dump CSV lewat env UT_DUMP ----
import os
import talib.abstract as ta


class UTBotFeat(_UTBotBase):
    timeframe = "1h"
    UT_A = float(os.environ.get("UT_A", 5))
    UT_C = int(os.environ.get("UT_C", 10))
    PAIRS = [(3, 8), (5, 13), (10, 20), (20, 50)]
    FEE = 0.0012  # 0.06% x 2 sisi

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df = super().populate_indicators(df, metadata)
        h, l, c, o = df["high"], df["low"], df["close"], df["open"]
        for p in (3, 5, 8, 10, 13, 20, 50):
            df[f"ema{p}"] = c.ewm(span=p, adjust=False).mean()
        atr = ta.ATR(df, timeperiod=14)
        flip = (df["ut_buy"] | df["ut_sell"]).values
        d = np.where(df["ut_buy"].values, 1, -1)
        n = len(df)

        f = pd.DataFrame(index=df.index)
        f["side"] = d
        f["adx"] = ta.ADX(df, timeperiod=14)
        f["rsi"] = ta.RSI(df, timeperiod=14)
        f["atr_pct"] = atr / c * 100
        f["body_atr"] = (c - o) * d / atr
        f["rng10_atr"] = (h.rolling(10).max() - l.rolling(10).min()) / atr
        f["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        f["chop"] = 100 * np.log10(tr.rolling(14).sum() / (h.rolling(14).max() - l.rolling(14).min())) / np.log10(14)
        f["d50"] = (c - df["ema50"]) * d / atr
        f["d200"] = (c - df["ema200"]) * d / atr
        for a, b in self.PAIRS:
            f[f"e{a}_{b}"] = (df[f"ema{a}"] - df[f"ema{b}"]) * d / atr
        f["hour"] = df["date"].dt.hour

        # label (pakai data ke depan, hanya untuk analisis, bukan untuk trading)
        ev = {p: df[f"ema{p}"].values for p in (3, 5, 8, 10, 13, 20, 50)}
        cl = c.values
        idx = np.flatnonzero(flip)
        end = np.append(idx[1:], n - 1)
        nan = lambda: np.full(n, np.nan)
        raw, leg, prev = nan(), nan(), nan()
        lag = {k: nan() for k in self.PAIRS}
        ret = {k: nan() for k in self.PAIRS}
        for m, (i, j) in enumerate(zip(idx, end)):
            s = d[i]
            raw[i] = s * (cl[j] / cl[i] - 1) - self.FEE
            leg[i] = j - i
            if m:
                prev[i] = i - idx[m - 1]
            for a, b in self.PAIRS:
                ok = (ev[a][i:j] - ev[b][i:j]) * s > 0
                if ok.any():
                    k = i + int(ok.argmax())
                    lag[(a, b)][i] = k - i
                    ret[(a, b)][i] = s * (cl[j] / cl[k] - 1) - self.FEE
        f["prev_leg"], f["leg"], f["ret_raw"] = prev, leg, raw
        for a, b in self.PAIRS:
            f[f"c{a}_{b}_lag"] = lag[(a, b)]
            f[f"c{a}_{b}_ret"] = ret[(a, b)]

        out = os.environ.get("UT_DUMP")
        if out:
            sig = f[flip].round(4)
            sig.insert(0, "pair", metadata["pair"])
            sig.insert(0, "date", df.loc[flip, "date"])
            sig.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        return df
