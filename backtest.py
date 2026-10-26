#!/usr/bin/env python3
"""
Backtest engine sederhana: Trend-Following (EMA cross) vs Mean-Reversion (RSI)
vs Regime-Switch (ADX menentukan mode mana yang aktif).
Data asli dari Bitget via CCXT. Fee Bitget nyata, tanpa leverage.

Cara pakai:
    pip install ccxt pandas numpy --break-system-packages
    python3 backtest.py

Edit PAIRS / TIMEFRAMES / DAYS_BACK di bawah untuk ubah cakupan uji.

Asumsi eksekusi:
- Sinyal dihitung dari close candle i. SEMUA aksi (entry, exit sinyal, reversal)
  dieksekusi di OPEN candle i+1, bukan di close candle sinyal.
- Stop-loss dicek mulai candle entry (low untuk long, high untuk short). Kalau
  open gap melewati SL, fill di open (lebih buruk dari harga SL).
- Dua mode: long_only dan long_short (short saat sinyal berlawanan).
  Funding rate perpetual TIDAK dimodelkan.
- Hasil dilaporkan 3 periode: full, IS (paruh pertama), OOS (paruh kedua).
  Trade dimasukkan ke IS/OOS berdasarkan waktu entry.
- Regime-Switch: ADX>ADX_TREND_THRESHOLD pakai sinyal EMA cross (trend-following),
  ADX<ADX_RANGE_THRESHOLD pakai sinyal RSI (mean-reversion), di antara keduanya
  tidak ada entry baru (posisi yang sudah terbuka tetap jalan sampai exit normal).
"""

import ccxt
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta, timezone

# ============== KONFIGURASI ==============
PAIRS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
    "BNB/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT",
    "ADA/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT",
    "LTC/USDT:USDT", "DOT/USDT:USDT", "TRX/USDT:USDT",
]  # Bitget perpetual swap
TIMEFRAMES = ["1h", "4h"]
DAYS_BACK = 365 * 2          # 2 tahun data
TAKER_FEE = 0.0006           # 0.06% per sisi (Bitget futures taker)
ROUND_TRIP_FEE = TAKER_FEE * 2
SLIPPAGE = 0.0005             # 0.05% asumsi slippage per sisi, konservatif

# Parameter strategi
EMA_FAST, EMA_SLOW = 21, 55
RSI_PERIOD = 14
RSI_OVERSOLD, RSI_OVERBOUGHT, RSI_EXIT = 30, 70, 50

# Parameter ADX (regime filter)
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25   # ADX > ini -> regime trending, pakai EMA cross
ADX_RANGE_THRESHOLD = 20   # ADX < ini -> regime ranging, pakai RSI mean-reversion

# Stop-loss: beberapa level diuji sekaligus (persentase dari harga entry).
# None berarti tanpa SL (baseline lama, untuk perbandingan).
SL_LEVELS = [None, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]

# Mode arah: nama -> boleh short?
MODES = {"long_only": False, "long_short": True}

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

    df["adx"] = _compute_adx(df, ADX_PERIOD)
    return df


def _compute_adx(df, period):
    """ADX klasik (Wilder smoothing) dari high/low/close."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx


def _open_position(side, ts, open_, sl_pct):
    """Buka posisi di open candle. Slippage selalu merugikan."""
    if side == "long":
        entry_price = open_ * (1 + SLIPPAGE)
        sl_price = entry_price * (1 - sl_pct) if sl_pct else None
    else:
        entry_price = open_ * (1 - SLIPPAGE)
        sl_price = entry_price * (1 + sl_pct) if sl_pct else None
    return {"side": side, "entry_ts": ts, "entry_price": entry_price, "sl_price": sl_price}


def _sl_fill(position, open_, high, low):
    """Harga fill mentah (sebelum slippage) kalau SL tersentuh, selain itu None.
    Kalau open gap melewati SL, fill di open (lebih buruk dari harga SL)."""
    sl = position["sl_price"]
    if sl is None:
        return None
    if position["side"] == "long":
        return min(open_, sl) if low <= sl else None
    return max(open_, sl) if high >= sl else None


def _close_position(position, ts, raw_exit_price, reason):
    if position["side"] == "long":
        exit_price = raw_exit_price * (1 - SLIPPAGE)
        gross = (exit_price - position["entry_price"]) / position["entry_price"]
    else:
        exit_price = raw_exit_price * (1 + SLIPPAGE)
        gross = (position["entry_price"] - exit_price) / position["entry_price"]
    return {
        "side": position["side"],
        "entry_ts": str(position["entry_ts"]), "exit_ts": str(ts),
        "entry_price": position["entry_price"], "exit_price": exit_price,
        "pnl_pct": gross - ROUND_TRIP_FEE, "exit_reason": reason,
    }


def _simulate(df, sl_pct, allow_short, get_signal):
    """Loop eksekusi bersama untuk semua strategi.
    get_signal(i, position) dipanggil di close candle i dan mengembalikan target
    untuk dieksekusi di open candle i+1: "long", "short", "flat", atau None.
    "short" saat allow_short=False hanya menutup long (tidak membuka short)."""
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    tss = df["ts"].tolist()

    trades = []
    position = None
    pending = None
    for i in range(1, len(df)):
        # 1) Eksekusi sinyal dari candle sebelumnya di open candle ini
        if pending is not None:
            if position is not None and position["side"] != pending:
                trades.append(_close_position(position, tss[i], opens[i], "signal"))
                position = None
            if position is None and (pending == "long" or (pending == "short" and allow_short)):
                position = _open_position(pending, tss[i], opens[i], sl_pct)
            pending = None

        # 2) Cek stop-loss dalam candle ini (termasuk candle entry)
        if position is not None:
            fill = _sl_fill(position, opens[i], highs[i], lows[i])
            if fill is not None:
                trades.append(_close_position(position, tss[i], fill, "stop_loss"))
                position = None

        # 3) Sinyal baru dari close candle ini
        pending = get_signal(i, position)
    return trades


def run_trend_following(df, sl_pct=None, allow_short=False):
    """EMA cross: fast cross up slow -> long, cross down -> short (atau hanya
    tutup long kalau allow_short=False). Selalu reversal saat sinyal berlawanan."""
    fast = df["ema_fast"].to_numpy()
    slow = df["ema_slow"].to_numpy()

    def get_signal(i, position):
        if fast[i - 1] <= slow[i - 1] and fast[i] > slow[i]:
            return "long"
        if fast[i - 1] >= slow[i - 1] and fast[i] < slow[i]:
            return "short"
        return None

    return _simulate(df, sl_pct, allow_short, get_signal)


def run_mean_reversion(df, sl_pct=None, allow_short=False):
    """RSI: cross di bawah 30 -> long, keluar saat cross di atas 50.
    Kalau allow_short: cross di atas 70 -> short, keluar saat cross di bawah 50."""
    rsi = df["rsi"].to_numpy()

    def get_signal(i, position):
        r, p = rsi[i], rsi[i - 1]
        if np.isnan(r) or np.isnan(p):
            return None
        side = position["side"] if position is not None else None
        if side == "long" and p <= RSI_EXIT < r:
            return "flat"
        if side == "short" and p >= RSI_EXIT > r:
            return "flat"
        if p >= RSI_OVERSOLD and r < RSI_OVERSOLD:
            return "long"
        if p <= RSI_OVERBOUGHT and r > RSI_OVERBOUGHT:
            return "short"
        return None

    return _simulate(df, sl_pct, allow_short, get_signal)


def run_regime_switch(df, sl_pct=None, allow_short=False):
    """ADX menentukan strategi mana yang aktif di candle ini:
    - ADX > ADX_TREND_THRESHOLD -> sinyal EMA cross (trend-following)
    - ADX < ADX_RANGE_THRESHOLD -> sinyal RSI (mean-reversion)
    - di antara keduanya -> tidak ada entry baru (posisi existing tetap exit normal)
    Exit tetap mengikuti logika strategi yang membuka posisi tsb (EMA reversal
    untuk posisi trend, RSI=50 cross untuk posisi mean-reversion)."""
    fast = df["ema_fast"].to_numpy()
    slow = df["ema_slow"].to_numpy()
    rsi = df["rsi"].to_numpy()
    adx = df["adx"].to_numpy()

    # Lacak strategi mana yang membuka posisi saat ini, supaya exit pakai logika yang sama
    open_by = {"strategy": None}

    def get_signal(i, position):
        a = adx[i]
        if position is not None:
            # posisi masih terbuka -> exit mengikuti strategi yang membukanya
            if open_by["strategy"] == "trend":
                if fast[i - 1] <= slow[i - 1] and fast[i] > slow[i]:
                    return "long"
                if fast[i - 1] >= slow[i - 1] and fast[i] < slow[i]:
                    return "short"
                return None
            if open_by["strategy"] == "meanrev":
                r, p = rsi[i], rsi[i - 1]
                if np.isnan(r) or np.isnan(p):
                    return None
                side = position["side"]
                if side == "long" and p <= RSI_EXIT < r:
                    return "flat"
                if side == "short" and p >= RSI_EXIT > r:
                    return "flat"
                return None
            return None

        # tidak ada posisi -> entry baru hanya kalau regime jelas (trending/ranging)
        if np.isnan(a):
            return None
        if a > ADX_TREND_THRESHOLD:
            if fast[i - 1] <= slow[i - 1] and fast[i] > slow[i]:
                open_by["strategy"] = "trend"
                return "long"
            if fast[i - 1] >= slow[i - 1] and fast[i] < slow[i]:
                open_by["strategy"] = "trend"
                return "short"
        elif a < ADX_RANGE_THRESHOLD:
            r, p = rsi[i], rsi[i - 1]
            if np.isnan(r) or np.isnan(p):
                return None
            if p >= RSI_OVERSOLD and r < RSI_OVERSOLD:
                open_by["strategy"] = "meanrev"
                return "long"
            if p <= RSI_OVERBOUGHT and r > RSI_OVERBOUGHT:
                open_by["strategy"] = "meanrev"
                return "short"
        return None

    return _simulate(df, sl_pct, allow_short, get_signal)


def compute_metrics(trades, strategy_name, symbol, timeframe, sl_label, period="full", mode="long_only"):
    base = {
        "strategy": strategy_name, "symbol": symbol, "timeframe": timeframe,
        "mode": mode, "stop_loss": sl_label, "period": period,
    }
    if not trades:
        return {
            **base,
            "n_trades": 0, "total_return_pct": 0, "win_rate_pct": 0,
            "profit_factor": None, "max_drawdown_pct": 0, "avg_pnl_pct": 0,
            "sl_hit_rate_pct": 0,
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

    sl_hits = sum(1 for t in trades if t.get("exit_reason") == "stop_loss")
    sl_hit_rate = sl_hits / n * 100

    return {
        **base,
        "n_trades": n, "total_return_pct": round(total_return, 2),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
        "max_drawdown_pct": round(max_dd, 2),
        "avg_pnl_pct": round(np.mean(pnls) * 100, 3),
        "sl_hit_rate_pct": round(sl_hit_rate, 1),
    }


def split_by_period(trades, split_ts):
    """Pisahkan trade jadi IS (entry sebelum split_ts) dan OOS (entry sesudahnya)."""
    split = str(split_ts)
    is_trades = [t for t in trades if t["entry_ts"] < split]
    oos_trades = [t for t in trades if t["entry_ts"] >= split]
    return {"full": trades, "IS": is_trades, "OOS": oos_trades}


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
            # Titik tengah rentang waktu: sebelum = in-sample, sesudah = out-of-sample
            split_ts = df["ts"].iloc[0] + (df["ts"].iloc[-1] - df["ts"].iloc[0]) / 2

            for sl in SL_LEVELS:
                sl_label = "none" if sl is None else f"{int(sl * 100)}%"

                for mode, allow_short in MODES.items():
                    tf_trades = run_trend_following(df, sl_pct=sl, allow_short=allow_short)
                    mr_trades = run_mean_reversion(df, sl_pct=sl, allow_short=allow_short)
                    rs_trades = run_regime_switch(df, sl_pct=sl, allow_short=allow_short)

                    for period, trades in split_by_period(tf_trades, split_ts).items():
                        all_results.append(compute_metrics(trades, "TrendFollowing_EMA21_55", symbol, tf, sl_label, period, mode))
                    for period, trades in split_by_period(mr_trades, split_ts).items():
                        all_results.append(compute_metrics(trades, "MeanReversion_RSI14", symbol, tf, sl_label, period, mode))
                    for period, trades in split_by_period(rs_trades, split_ts).items():
                        all_results.append(compute_metrics(trades, "RegimeSwitch_ADX", symbol, tf, sl_label, period, mode))

                    raw_trades[f"{symbol}_{tf}_trend_sl{sl_label}_{mode}"] = tf_trades
                    raw_trades[f"{symbol}_{tf}_meanrev_sl{sl_label}_{mode}"] = mr_trades
                    raw_trades[f"{symbol}_{tf}_regimeswitch_sl{sl_label}_{mode}"] = rs_trades

    # simpan hasil
    with open(OUTPUT_JSON, "w") as f:
        json.dump({"summary": all_results, "trades": raw_trades}, f, indent=2, default=str)

    df_summary = pd.DataFrame(all_results)
    df_summary = df_summary.sort_values(["timeframe", "strategy", "mode", "stop_loss", "symbol", "period"])
    df_summary.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 100)
    print("RINGKASAN HASIL (per timeframe, strategi, mode, SL, pair, periode full/IS/OOS)")
    print("=" * 100)
    print(df_summary.to_string(index=False))
    print(f"\nDisimpan ke: {OUTPUT_CSV} dan {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
