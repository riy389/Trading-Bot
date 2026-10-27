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
- Regime-Switch (dua varian): ADX>ADX_TREND_THRESHOLD pakai sinyal EMA cross
  (trend-following), ADX<ADX_RANGE_THRESHOLD pakai sinyal RSI (mean-reversion),
  di antara keduanya tidak ada entry baru. Posisi yang sudah terbuka ditutup
  PAKSA ("flat") begitu regime jelas berbalik ke lawannya (trend->ranging atau
  sebaliknya), supaya tidak nyangkut lama menunggu sinyal strategi asal (bug
  yang ditemukan: EMA cross sangat lambat bereaksi saat tren melemah pelan).
  V1 (run_regime_switch) hanya exit paksa + SL dari SL_LEVELS seperti biasa.
  V2 (run_regime_switch_v2) sama, ditambah SL tetap (REGIME_SWITCH_TREND_SL)
  khusus posisi trend sebagai pengaman tambahan; posisi meanrev tetap pakai
  sl_pct dari SL_LEVELS seperti biasa (tanpa SL tambahan di luar itu).
"""

import ccxt
import pandas as pd
import numpy as np
import os
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

# SL tetap khusus posisi trend di run_regime_switch_v2 (independen dari SL_LEVELS loop utama)
REGIME_SWITCH_TREND_SL = 0.02

# Stop-loss: beberapa level diuji sekaligus (persentase dari harga entry).
# None berarti tanpa SL (baseline lama, untuk perbandingan).
SL_LEVELS = [None, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]

# Mode arah: nama -> boleh short?
MODES = {"long_only": False, "long_short": True}

# Validasi data: gap lebih dari ini (dalam jumlah candle hilang) -> pair/timeframe di-skip
# supaya SL tidak disimulasikan melewati data yang hilang. Gap kecil ditoleransi.
MAX_GAP_CANDLES = 3

CACHE_DIR = "cache"          # cache candle per pair/timeframe (CSV)
OUTPUT_JSON = "results.json"
OUTPUT_CSV = "results_summary.csv"
# ===========================================


def fetch_ohlcv(exchange, symbol, timeframe, days_back, since_ms=None):
    """Ambil OHLCV lengkap dengan pagination, karena CCXT membatasi per-call.
    Berhenti hanya saat data sudah mencapai waktu sekarang, bukan saat satu
    call mengembalikan kurang dari `limit` candle (Bitget bisa saja balas
    sebagian meski masih ada data lebih lanjut)."""
    since = since_ms if since_ms is not None else exchange.parse8601(
        (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    now_ms = exchange.milliseconds()
    all_candles = []
    # Bitget history-candles: maks 200 candle per call. Kalau limit lebih besar,
    # window (limit x timeframe) tetap dicakup tapi hanya 200 candle TERAKHIR
    # yang dikembalikan -> sisanya hilang (penyebab gap ~800 candle).
    limit = 200
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


def fetch_ohlcv_cached(exchange, symbol, timeframe, days_back):
    """fetch_ohlcv dengan cache CSV: hanya ambil candle sejak candle terakhir di
    cache (candle terakhir diambil ulang karena bisa saja belum selesai)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{symbol.split(':')[0].replace('/', '_')}_{timeframe}.csv")
    cached, since_ms = None, None
    if os.path.exists(path):
        cached = pd.read_csv(path)
        cached["ts"] = pd.to_datetime(cached["ts"], utc=True)
        if len(cached):
            since_ms = int(cached["ts"].iloc[-1].timestamp() * 1000)
    new = fetch_ohlcv(exchange, symbol, timeframe, days_back, since_ms)
    parts = [d for d in (cached, new) if d is not None and len(d)]
    if not parts:
        return None
    df = pd.concat(parts).drop_duplicates(subset="ts", keep="last").sort_values("ts")
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back)
    df = df[df["ts"] >= cutoff].reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"  [cache] {0 if cached is None else len(cached)} dari cache + {0 if new is None else len(new)} baru")
    return df


def find_ohlcv_gaps(df, timeframe):
    """Daftar (ts_sebelum, ts_sesudah, jumlah_candle_hilang) untuk tiap lompatan
    waktu yang lebih besar dari satu periode timeframe."""
    expected = pd.Timedelta(timeframe)
    deltas = df["ts"].diff()
    gaps = []
    for idx in deltas[deltas > expected].index:
        missing = int(round(deltas[idx] / expected)) - 1
        if missing > 0:
            gaps.append((df["ts"].iloc[idx - 1], df["ts"].iloc[idx], missing))
    return gaps


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
    """V1 (A): ADX menentukan strategi mana yang aktif di candle ini:
    - ADX > ADX_TREND_THRESHOLD -> sinyal EMA cross (trend-following)
    - ADX < ADX_RANGE_THRESHOLD -> sinyal RSI (mean-reversion)
    - di antara keduanya -> tidak ada entry baru (posisi existing tetap exit normal)
    Exit mengikuti logika strategi yang membuka posisi tsb (EMA reversal untuk
    posisi trend, RSI=50 cross untuk posisi mean-reversion), DITAMBAH exit paksa
    kalau regime sudah jelas berbalik total (posisi trend ditutup segera saat
    ADX turun di bawah ADX_RANGE_THRESHOLD; posisi meanrev ditutup segera saat
    ADX naik di atas ADX_TREND_THRESHOLD) -- supaya posisi tidak nyangkut lama
    menunggu sinyal strategi asal sementara regime sudah berubah."""
    fast = df["ema_fast"].to_numpy()
    slow = df["ema_slow"].to_numpy()
    rsi = df["rsi"].to_numpy()
    adx = df["adx"].to_numpy()

    # Lacak strategi mana yang membuka posisi saat ini, supaya exit pakai logika yang sama
    open_by = {"strategy": None}

    def get_signal(i, position):
        a = adx[i]
        if position is not None:
            # exit paksa kalau regime sudah jelas berbalik ke lawannya
            if open_by["strategy"] == "trend" and not np.isnan(a) and a < ADX_RANGE_THRESHOLD:
                return "flat"
            if open_by["strategy"] == "meanrev" and not np.isnan(a) and a > ADX_TREND_THRESHOLD:
                return "flat"

            # selain itu, exit mengikuti logika strategi yang membukanya
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


def run_regime_switch_v2(df, sl_pct=None, allow_short=False):
    """V2 (B): sama seperti run_regime_switch (exit paksa saat regime berbalik),
    DITAMBAH SL tetap REGIME_SWITCH_TREND_SL (independen dari sl_pct) khusus
    untuk posisi trend, sebagai pengaman kalau exit paksa telat terdeteksi
    (ADX baru update per candle). Posisi meanrev tetap tanpa SL tambahan di
    luar sl_pct, sesuai temuan backtest sebelumnya (SL ketat merusak
    mean-reversion)."""
    fast = df["ema_fast"].to_numpy()
    slow = df["ema_slow"].to_numpy()
    rsi = df["rsi"].to_numpy()
    adx = df["adx"].to_numpy()

    open_by = {"strategy": None}

    def get_signal(i, position):
        a = adx[i]
        if position is not None:
            if open_by["strategy"] == "trend" and not np.isnan(a) and a < ADX_RANGE_THRESHOLD:
                return "flat"
            if open_by["strategy"] == "meanrev" and not np.isnan(a) and a > ADX_TREND_THRESHOLD:
                return "flat"

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

    # SL per-posisi berbeda (trend pakai SL tetap, meanrev pakai sl_pct dari loop utama)
    # -> _simulate hanya terima satu sl_pct global, jadi dijalankan dengan wrapper
    # yang memilih sl_pct saat open_position terjadi.
    trades = _simulate_regime_v2(df, sl_pct, allow_short, get_signal, open_by)

    return trades


def _simulate_regime_v2(df, sl_pct, allow_short, get_signal, open_by):
    """Varian _simulate khusus run_regime_switch_v2: SL posisi trend selalu
    REGIME_SWITCH_TREND_SL, SL posisi meanrev memakai sl_pct seperti biasa."""
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    tss = df["ts"].tolist()

    trades = []
    position = None
    pending = None
    for i in range(1, len(df)):
        if pending is not None:
            if position is not None and position["side"] != pending:
                trades.append(_close_position(position, tss[i], opens[i], "signal"))
                position = None
            if position is None and (pending == "long" or (pending == "short" and allow_short)):
                effective_sl = REGIME_SWITCH_TREND_SL if open_by["strategy"] == "trend" else sl_pct
                position = _open_position(pending, tss[i], opens[i], effective_sl)
            pending = None

        if position is not None:
            fill = _sl_fill(position, opens[i], highs[i], lows[i])
            if fill is not None:
                trades.append(_close_position(position, tss[i], fill, "stop_loss"))
                position = None

        pending = get_signal(i, position)
    return trades


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
            df = fetch_ohlcv_cached(exchange, symbol, tf, DAYS_BACK)
            if df is None or len(df) < 100:
                print(f"  [!] Data tidak cukup untuk {symbol} {tf} (dapat {0 if df is None else len(df)} candle), skip.")
                continue
            print(f"  -> SELESAI: {len(df)} candle didapat ({df['ts'].iloc[0]} s/d {df['ts'].iloc[-1]})")

            gaps = find_ohlcv_gaps(df, tf)
            if gaps:
                print(f"  [gap] {len(gaps)} gap, total {sum(g[2] for g in gaps)} candle hilang, terbesar {max(g[2] for g in gaps)}")
            big_gaps = [g for g in gaps if g[2] > MAX_GAP_CANDLES]
            if big_gaps:
                for before, after, missing in big_gaps[:3]:
                    print(f"      {before} -> {after} (~{missing} candle hilang)")
                print(f"  [!] {len(big_gaps)} gap besar (> {MAX_GAP_CANDLES} candle) di {symbol} {tf}, skip agar SL tidak disimulasikan melewati data hilang.")
                continue

            df = add_indicators(df)
            # Titik tengah rentang waktu: sebelum = in-sample, sesudah = out-of-sample
            split_ts = df["ts"].iloc[0] + (df["ts"].iloc[-1] - df["ts"].iloc[0]) / 2

            for sl in SL_LEVELS:
                sl_label = "none" if sl is None else f"{int(sl * 100)}%"

                for mode, allow_short in MODES.items():
                    tf_trades = run_trend_following(df, sl_pct=sl, allow_short=allow_short)
                    mr_trades = run_mean_reversion(df, sl_pct=sl, allow_short=allow_short)
                    rs_trades = run_regime_switch(df, sl_pct=sl, allow_short=allow_short)
                    rs2_trades = run_regime_switch_v2(df, sl_pct=sl, allow_short=allow_short)

                    for period, trades in split_by_period(tf_trades, split_ts).items():
                        all_results.append(compute_metrics(trades, "TrendFollowing_EMA21_55", symbol, tf, sl_label, period, mode))
                    for period, trades in split_by_period(mr_trades, split_ts).items():
                        all_results.append(compute_metrics(trades, "MeanReversion_RSI14", symbol, tf, sl_label, period, mode))
                    for period, trades in split_by_period(rs_trades, split_ts).items():
                        all_results.append(compute_metrics(trades, "RegimeSwitch_ADX", symbol, tf, sl_label, period, mode))
                    for period, trades in split_by_period(rs2_trades, split_ts).items():
                        all_results.append(compute_metrics(trades, "RegimeSwitch_V2_ADX", symbol, tf, sl_label, period, mode))

                    raw_trades[f"{symbol}_{tf}_trend_sl{sl_label}_{mode}"] = tf_trades
                    raw_trades[f"{symbol}_{tf}_meanrev_sl{sl_label}_{mode}"] = mr_trades
                    raw_trades[f"{symbol}_{tf}_regimeswitch_sl{sl_label}_{mode}"] = rs_trades
                    raw_trades[f"{symbol}_{tf}_regimeswitchv2_sl{sl_label}_{mode}"] = rs2_trades

    if not all_results:
        raise SystemExit("[!] Tidak ada pair/timeframe yang lolos validasi data, tidak ada hasil untuk disimpan.")

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
