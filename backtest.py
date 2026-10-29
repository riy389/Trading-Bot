#!/usr/bin/env python3
"""
Backtest engine: UT Bot Alerts (TradingView, Pine v4) diport persis ke Python.
Data asli dari Bitget via CCXT. Ukuran posisi dari risiko 1% ekuitas.

Tahap 1 = port persis (a=1, ATR 10, tanpa Heikin Ashi). Tuning baru setelah hasil dilihat.

Aturan eksekusi:
- Sinyal (buy/sell UT) dihitung di CLOSE candle i, entry di OPEN candle i+1.
- 1R = jarak SL = UT_A x ATR candle sinyal. Ukuran posisi = RISK_PER_TRADE / jarak SL (persen),
  TANPA fee di dalamnya dan tanpa batas leverage, supaya rugi di SL selalu -1R (gross).
- Exit hanya lewat SL tetap (1R) atau trailing TP (garis trailing stop UT dari candle
  sebelumnya), dicek intrabar pakai low/high. Gap -> fill di open. Sinyal berlawanan saat posisi
  masih hidup: tutup di open lalu buka arah baru.
- Fee + slippage TIDAK ada di strategi: mesin memotongnya saat trade ditutup
  (net_R = gross_R - fee_R). Versi 'net' (0.22%) dan 'gross' (0) dijalankan keduanya.
- Dua sesi: '24h' (entry 07:00-05:00 WIB, tutup paksa 06:00 WIB) dan 'free' (tanpa batas waktu).
- Funding rate tidak dimodelkan.
"""

import ccxt
import pandas as pd
import numpy as np
import os
import time
import json
import math
from datetime import datetime, timedelta, timezone

# ============== KONFIGURASI ==============
PAIRS = ["BTC/USDT:USDT"]  # Bitget perpetual swap (sementara 1 pair; 12 pair ada di git history)
TIMEFRAMES = ["5m", "15m", "1h", "4h"]
DAYS_BACK = 365 * 2          # 2 tahun data
TAKER_FEE = 0.0006           # 0.06% per sisi (Bitget futures taker)
ROUND_TRIP_FEE = TAKER_FEE * 2
SLIPPAGE = 0.0005            # 0.05% per sisi
COST_VERSIONS = {"net": ROUND_TRIP_FEE + 2 * SLIPPAGE, "gross": 0.0}   # % dari nilai posisi

# UT Bot Alerts (default TradingView)
UT_A = 1                     # Key Value (sensitivitas)
UT_C = 10                    # ATR period
UT_HA = False                # sinyal dari close Heikin Ashi

RISK_PER_TRADE = 0.01        # 1R = 1% ekuitas
MIN_SL_PCT = 0.0             # jarak SL minimum (% harga); 0 = mati

SESSIONS = ["24h", "free"]
TP_R = 1.5                   # TP tetap = TP_R x 1R dari entry (gross)
# varian exit: nama -> (pakai trailing UT, pakai TP tetap)
VARIANTS = {"trail": (True, False), "tp15": (False, True), "trail_tp15": (True, True),
            "trail_be1": (True, False), "trail_be05": (True, False)}
# BEP: SL pindah ke entry +/- biaya setelah profit tertinggi (candle sebelumnya) >= trigger R
BE_R = {"trail_be1": 1.0, "trail_be05": 0.5}
BE_COST = COST_VERSIONS["net"]
SESSION_LAST_ENTRY_MIN = 22 * 60   # 22:00 UTC = 05:00 WIB, entry terakhir
SESSION_CLOSE_MIN = 23 * 60        # 23:00 UTC = 06:00 WIB, tutup paksa

MODES = {"long_short": True}
MAX_GAP_CANDLES = 3          # gap data lebih besar -> pair/timeframe di-skip

CACHE_DIR = "cache"
OUTPUT_JSON = "results.json"
OUTPUT_CSV = "results_summary.csv"
# ===========================================

def fetch_ohlcv(exchange, symbol, timeframe, days_back, since_ms=None, until_ms=None):
    """Ambil OHLCV lengkap dengan pagination, karena CCXT membatasi per-call.
    Berhenti hanya saat data sudah mencapai waktu sekarang, bukan saat satu
    call mengembalikan kurang dari `limit` candle (Bitget bisa saja balas
    sebagian meski masih ada data lebih lanjut)."""
    since = since_ms if since_ms is not None else exchange.parse8601(
        (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    now_ms = until_ms if until_ms is not None else exchange.milliseconds()
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
    old = None
    first_cached = cached["ts"].iloc[0] if cached is not None and len(cached) else None
    want = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back)
    if first_cached is not None and first_cached > want + pd.Timedelta(days=2):
        # cache lebih pendek dari yang diminta: tarik data lama (backfill) sampai awal cache
        print(f"  [backfill] cache mulai {first_cached}, ambil sejak {want}")
        old = fetch_ohlcv(exchange, symbol, timeframe, days_back, int(want.timestamp() * 1000),
                          until_ms=int(first_cached.timestamp() * 1000))
    parts = [d for d in (old, cached, new) if d is not None and len(d)]
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


# ---------- UT Bot Alerts (port persis dari Pine v4) ----------

def _rma_atr(high, low, close, length):
    """atr() Pine v4 = RMA(TR, length), seed = SMA(TR) di bar ke-length."""
    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum.reduce([
        high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])])
    out = np.full(n, np.nan)
    if n >= length:
        out[length - 1] = tr[:length].mean()
        for i in range(length, n):
            out[i] = (out[i - 1] * (length - 1) + tr[i]) / length
    return out


def ut_bot(df, a=UT_A, c=UT_C, ha=UT_HA):
    """Return (stop, buy, sell, nloss), array per candle (dihitung di close candle).
    stop = xATRTrailingStop. buy/sell = alertcondition 'UT Long'/'UT Short'.
    ha=True: src = close Heikin Ashi (o+h+l+c)/4; ATR tetap dari candle biasa."""
    o, h, l, cl = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    src = (o + h + l + cl) / 4 if ha else cl
    nloss = a * _rma_atr(h, l, cl, c)
    n = len(src)
    stop = np.full(n, np.nan)
    prev = float("nan")
    for i in range(n):
        nl = nloss[i]
        p = 0.0 if math.isnan(prev) else prev            # nz(stop[1], 0)
        s = src[i]
        s1 = src[i - 1] if i > 0 else float("nan")
        if math.isnan(nl):
            cur = float("nan")                            # max/min dengan na = na
        elif s > p and s1 > p:
            cur = max(p, s - nl)
        elif s < p and s1 < p:
            cur = min(p, s + nl)
        elif s > p:
            cur = s - nl
        else:
            cur = s + nl
        stop[i] = cur
        prev = cur
    stop1 = np.r_[np.nan, stop[:-1]]
    src1 = np.r_[np.nan, src[:-1]]
    ema = src                                             # ema(src, 1) == src
    ema1 = np.r_[np.nan, ema[:-1]]
    with np.errstate(invalid="ignore"):
        above = (ema > stop) & (ema1 <= stop1)            # crossover(ema, stop)
        below = (stop > ema) & (stop1 <= ema1)            # crossover(stop, ema)
        buy = (src > stop) & above
        sell = (src < stop) & below
    return stop, buy, sell, nloss


# ---------- Eksekusi: SL tetap 1R + trailing TP (garis UT), biaya dipotong di luar strategi ----------

def _time_arrays(df):
    ts = df["ts"]
    tod = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
    day = ((ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(days=1)).to_numpy()
    return tod, day


def _close(pos, ts, price, reason):
    sign = 1 if pos["side"] == "long" else -1
    gross_ret = sign * (price - pos["entry"]) / pos["entry"]
    return {
        "side": pos["side"], "entry_ts": str(pos["ts"]), "exit_ts": str(ts),
        "entry_price": pos["entry"], "exit_price": price,
        "sl_pct": pos["sl_pct"], "size": pos["size"],
        "gross_ret": gross_ret, "gross_R": gross_ret / pos["sl_pct"], "exit_reason": reason,
        "mfe_R": pos["mfe"],
    }


def apply_costs(trades, cost):
    """Biaya (fee + slippage, % dari nilai posisi) dipotong saat trade ditutup, di luar strategi.
    pnl_pct = persen ekuitas; fee_R = biaya dalam kelipatan R; net_R = gross_R - fee_R."""
    out = []
    for t in trades:
        fee_r = cost / t["sl_pct"]
        out.append({**t, "fee_R": fee_r, "net_R": t["gross_R"] - fee_r,
                    "pnl_pct": t["size"] * (t["gross_ret"] - cost)})
    return out


def simulate_ut(df, ut, session, allow_short, trail_on=True, tp_on=False, be_r=None):
    """Sinyal di close candle i-1 dieksekusi di open candle i.
    Exit hanya lewat: SL tetap (entry -/+ nloss sinyal, 1R) atau trailing TP (garis UT candle
    sebelumnya), keduanya dicek intrabar (low/high); gap -> fill di open. Sinyal berlawanan
    saat posisi masih hidup: tutup di open lalu buka arah baru (kalau diizinkan).
    session '24h': entry hanya sampai 22:00 UTC (05:00 WIB), tutup paksa 23:00 UTC (06:00 WIB)."""
    o, h, l = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low"))
    stop, buy, sell, nloss = ut
    tod, day = _time_arrays(df)
    tss = df["ts"].tolist()
    trades, pos, pending = [], None, None
    for i in range(1, len(df)):
        # 1) tutup paksa akhir sesi
        if pos is not None and session == "24h" and (day[i] > pos["day"] or tod[i] >= SESSION_CLOSE_MIN):
            trades.append(_close(pos, tss[i], o[i], "session_end"))
            pos = None

        # 2) eksekusi sinyal candle sebelumnya di open candle ini
        if pending is not None:
            if pos is not None and pos["side"] != pending:
                trades.append(_close(pos, tss[i], o[i], "reverse"))
                pos = None
            if pos is None and (pending == "long" or allow_short):
                nl, tr_sig, e = nloss[i - 1], stop[i - 1], o[i]
                sign = 1 if pending == "long" else -1
                ok = (nl > 0 and nl / e >= MIN_SL_PCT and not math.isnan(tr_sig)
                      and sign * (e - tr_sig) > 0                    # open belum melewati garis UT
                      and (session == "free" or tod[i] <= SESSION_LAST_ENTRY_MIN))
                if ok:
                    sl_pct = nl / e
                    pos = {"side": pending, "ts": tss[i], "entry": e, "sl": e - sign * nl, "tp": e + sign * TP_R * nl, "mfe": 0.0,
                           "sl_pct": sl_pct, "size": RISK_PER_TRADE / sl_pct, "day": day[i]}
            pending = None

        # 3) SL tetap / trailing TP / TP tetap, dicek intrabar (termasuk candle entry)
        #    SL dan TP tetap kena di candle yang sama -> SL dianggap duluan
        if pos is not None:
            sign = 1 if pos["side"] == "long" else -1
            sl = pos["sl"]
            be_on = be_r is not None and pos["mfe"] >= be_r
            if be_on:
                sl = pos["entry"] * (1 + sign * BE_COST)
            trail = stop[i - 1]
            use_trail = trail_on and sign * (trail - sl) > 0
            lvl = trail if use_trail else sl
            if sign == 1:
                fill = min(o[i], lvl) if l[i] <= lvl else None
            else:
                fill = max(o[i], lvl) if h[i] >= lvl else None
            reason = None
            if fill is not None:
                if not use_trail:
                    reason = "be_stop" if be_on else "stop_loss"
                else:
                    reason = "trail_win" if sign * (fill - pos["entry"]) > 0 else "trail_loss"
            elif tp_on:
                if sign == 1 and h[i] >= pos["tp"]:
                    fill, reason = max(o[i], pos["tp"]), "tp_fixed"
                elif sign == -1 and l[i] <= pos["tp"]:
                    fill, reason = min(o[i], pos["tp"]), "tp_fixed"
            if reason is not None:
                trades.append(_close(pos, tss[i], fill, reason))
                pos = None
            else:
                ext = h[i] if sign == 1 else l[i]
                pos["mfe"] = max(pos["mfe"], sign * (ext - pos["entry"]) / (pos["entry"] * pos["sl_pct"]))

        # 4) sinyal baru dari close candle ini
        if buy[i]:
            pending = "long"
        elif sell[i]:
            pending = "short"
    return trades


REASONS = {"stop_loss": "n_sl", "tp_fixed": "n_tp_fixed", "trail_win": "n_trail_win",
           "trail_loss": "n_trail_loss", "session_end": "n_session", "reverse": "n_reverse",
           "be_stop": "n_be"}


def _reason_counts(trades):
    out = {col: sum(1 for t in trades if t["exit_reason"] == r) for r, col in REASONS.items()}
    # trade rugi (gross) yang sebelumnya sempat untung >= k R (MFE intrabar, candle exit tidak dihitung)
    losers = [t for t in trades if t["gross_R"] < 0]
    out["n_loss"] = len(losers)
    for k, nm in ((0.5, "05"), (1.0, "10"), (1.5, "15")):
        out["loss_mfe" + nm] = sum(1 for t in losers if t["mfe_R"] >= k)
    return out


def compute_metrics(trades, strategy_name, symbol, timeframe, sl_label, tp_label, period="full", mode="long_only"):
    base = {
        "strategy": strategy_name, "symbol": symbol, "timeframe": timeframe,
        "mode": mode, "stop_loss": sl_label, "take_profit": tp_label, "period": period,
    }
    if not trades:
        return {
            **base,
            "n_trades": 0, "total_return_pct": 0, "win_rate_pct": 0,
            "profit_factor": None, "max_drawdown_pct": 0, "avg_pnl_pct": 0,
            "sl_hit_rate_pct": 0, "tp_hit_rate_pct": 0,
            "avg_gross_R": 0, "avg_fee_R": 0, "avg_net_R": 0, "avg_lev_x": 0, "max_lev_x": 0,
            **_reason_counts(trades),
        }

    pnls = [t["pnl_pct"] for t in trades]
    n = len(pnls)
    wins = [p for p in pnls if p > 1e-9]
    losses = [p for p in pnls if p <= 1e-9]
    win_rate = len(wins) / n * 100

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0)

    equity = [1.0]
    for p in pnls:
        equity.append(equity[-1] * (1 + p))
    equity = np.array(equity)
    total_return = (equity[-1] - 1) * 100
    running_max = np.maximum.accumulate(equity)
    max_dd = ((equity - running_max) / running_max).min() * 100

    sl_hits = sum(1 for t in trades if t["exit_reason"] == "stop_loss")
    tp_hits = sum(1 for t in trades if t["exit_reason"] in ("tp_fixed", "trail_win"))
    return {
        **base,
        "n_trades": n, "total_return_pct": round(total_return, 2),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
        "max_drawdown_pct": round(max_dd, 2),
        "avg_pnl_pct": round(np.mean(pnls) * 100, 3),
        "sl_hit_rate_pct": round(sl_hits / n * 100, 1),
        "tp_hit_rate_pct": round(tp_hits / n * 100, 1),
        "avg_gross_R": round(float(np.mean([t["gross_R"] for t in trades])), 3),
        "avg_fee_R": round(float(np.mean([t["fee_R"] for t in trades])), 3),
        "avg_net_R": round(float(np.mean([t["net_R"] for t in trades])), 3),
        "avg_lev_x": round(float(np.mean([t["size"] for t in trades])), 2),
        "max_lev_x": round(float(np.max([t["size"] for t in trades])), 2),
        **_reason_counts(trades),
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
    for symbol in PAIRS:
        for tf in TIMEFRAMES:
            print(f"\n=== Fetching {symbol} {tf} ({DAYS_BACK} hari) ===")
            df = fetch_ohlcv_cached(exchange, symbol, tf, DAYS_BACK)
            if df is None or len(df) < 100:
                print(f"  [!] Data tidak cukup untuk {symbol} {tf}, skip.")
                continue
            print(f"  -> {len(df)} candle ({df['ts'].iloc[0]} s/d {df['ts'].iloc[-1]})")
            big_gaps = [g for g in find_ohlcv_gaps(df, tf) if g[2] > MAX_GAP_CANDLES]
            if big_gaps:
                print(f"  [!] {len(big_gaps)} gap besar (> {MAX_GAP_CANDLES} candle) di {symbol} {tf}, skip.")
                continue

            ut = ut_bot(df)
            split_ts = df["ts"].iloc[0] + (df["ts"].iloc[-1] - df["ts"].iloc[0]) / 2
            for session in SESSIONS:
                for mode, allow_short in MODES.items():
                    for vname, (trail_on, tp_on) in VARIANTS.items():
                        trades = simulate_ut(df, ut, session, allow_short, trail_on, tp_on, BE_R.get(vname))
                        tp_label = "+".join(x for x in ("trail_UT" if trail_on else "", f"TP{TP_R:g}R" if tp_on else "", f"BE{BE_R[vname]:g}R" if vname in BE_R else "") if x)
                        for cost_name, cost in COST_VERSIONS.items():
                            suffix = "" if cost_name == "net" else "_gross"
                            name = f"UT_a{UT_A:g}_c{UT_C}_{session}_{vname}{suffix}"
                            for period, tr in split_by_period(apply_costs(trades, cost), split_ts).items():
                                all_results.append(compute_metrics(
                                    tr, name, symbol, tf, "SL1R_fixed", tp_label, period, mode))

    if not all_results:
        raise SystemExit("[!] Tidak ada pair/timeframe yang lolos validasi data, tidak ada hasil untuk disimpan.")

    with open(OUTPUT_JSON, "w") as f:
        json.dump({"summary": all_results}, f, indent=2, default=str)
    df_summary = pd.DataFrame(all_results)
    df_summary = df_summary.sort_values(["timeframe", "strategy", "mode", "symbol", "period"])
    df_summary.to_csv(OUTPUT_CSV, index=False)
    print("\n" + "=" * 100)
    print(df_summary.to_string(index=False))
    print(f"\nDisimpan ke: {OUTPUT_CSV} dan {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
