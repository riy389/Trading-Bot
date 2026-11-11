#!/usr/bin/env python3
"""
UT_SWING: UT Bot BTC 1h (A=3, ATR 10) dengan SL di swing terakhir sebelum sinyal.
Spesifikasi: Issue #1, "Checkpoint 10 Nov 2026 (malam)".

- Sinyal UT di close candle i, entry di OPEN candle i+1.
- SL: long = swing low terakhir sebelum sinyal, short = swing high terakhir. 1R = jarak entry ke SL.
  Swing = pivot SWING_N candle kiri dan kanan, baru dipakai SWING_N candle setelah pivot (tanpa lookahead).
  Entry dilewati kalau swing ada di sisi salah dari harga entry atau jarak SL < MIN_SL_PCT.
- Varian: flip = SL swing + keluar di sinyal balik. lock1R = flip + begitu high/low candle sebelumnya sudah +1R,
  SL dipasang di +1R, lalu trailing garis UT (stop[i-1]) kalau lebih menguntungkan. Kunci berlaku mulai candle berikutnya.
- Pembanding: sinyal acak (jumlah sama) dengan aturan SL dan exit yang sama.
- Biaya dipotong mesin saat trade ditutup (bt.apply_costs), net dan gross.
"""

import os

import numpy as np
import pandas as pd

import backtest as bt

SYMBOL, TF = "BTC/USDT:USDT", "1h"
UT_A, UT_C = 3, 10
SWING_N = 5          # dikunci sebelum lihat hasil
LOCK_R = 1.0
OUTPUT_TRADES = "trades_utswing.csv"


def swing_levels(h, l, n):
    """(swing_high, swing_low) terakhir yang sudah terkonfirmasi di close candle i. Pivot j butuh candle j+n, jadi baru ada di i = j+n."""
    N = len(h)
    sh, sl = np.full(N, np.nan), np.full(N, np.nan)
    cur_h = cur_l = np.nan
    for i in range(N):
        j = i - n
        if j >= n:
            if h[j] == h[j - n:j + n + 1].max():
                cur_h = h[j]
            if l[j] == l[j - n:j + n + 1].min():
                cur_l = l[j]
        sh[i], sl[i] = cur_h, cur_l
    return sh, sl


def simulate_swing(df, buy, sell, sh, sl, stop, lock):
    o, h, l = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low"))
    tss = df["ts"].tolist()
    trades, pos, pending = [], None, None

    def close(p, i, price, reason):
        gross_ret = p["sign"] * (price - p["entry"]) / p["entry"]
        return {"side": "long" if p["sign"] == 1 else "short", "entry_ts": str(p["ts"]), "exit_ts": str(tss[i]),
                "entry_price": p["entry"], "exit_price": price, "sl_pct": p["sl_pct"], "size": bt.RISK_PER_TRADE / p["sl_pct"],
                "gross_ret": gross_ret, "gross_R": gross_ret / p["sl_pct"], "exit_reason": reason, "mfe_R": p["mfe"]}

    for i in range(1, len(df)):
        if pending is not None:
            sign = 1 if pending == "long" else -1
            pending = None
            if pos is not None and pos["sign"] != sign:
                trades.append(close(pos, i, o[i], "reverse"))
                pos = None
            if pos is None:
                e = o[i]
                sw = sl[i - 1] if sign == 1 else sh[i - 1]
                nl = sign * (e - sw)                      # jarak entry ke swing, harus > 0
                if np.isfinite(sw) and nl > 0 and nl / e >= bt.MIN_SL_PCT:
                    pos = {"sign": sign, "entry": e, "ts": tss[i], "sl": sw, "nl": nl, "sl_pct": nl / e, "mfe": 0.0}
        if pos is not None:
            sign, lvl, reason = pos["sign"], pos["sl"], "stop_loss"
            if lock and pos["mfe"] >= LOCK_R:
                lk = pos["entry"] + sign * LOCK_R * pos["nl"]
                if sign * (lk - lvl) > 0:
                    lvl, reason = lk, "lock_stop"
                if sign * (stop[i - 1] - lvl) > 0:
                    lvl, reason = stop[i - 1], "trail_win"
            if (l[i] <= lvl) if sign == 1 else (h[i] >= lvl):
                fill = min(o[i], lvl) if sign == 1 else max(o[i], lvl)   # gap -> fill di open
                trades.append(close(pos, i, fill, reason))
                pos = None
            else:
                ext = h[i] if sign == 1 else l[i]
                pos["mfe"] = max(pos["mfe"], sign * (ext - pos["entry"]) / pos["nl"])
        if buy[i]:
            pending = "long"
        elif sell[i]:
            pending = "short"
    if pos is not None:
        trades.append(close(pos, len(df) - 1, float(df["close"].iloc[-1]), "end_data"))
    return trades


def check_utswing(exchange):
    """Cek cepat: 8 sinyal UT terakhir dan level SL swing-nya, untuk dicocokkan dengan swing yang dilingkari di chart."""
    df = bt.fetch_ohlcv(exchange, SYMBOL, TF, 30)
    if df is None:
        return
    df = df.iloc[:-1].reset_index(drop=True)
    _, buy, sell, _ = bt.ut_bot(df, UT_A, UT_C)
    sh, sl = swing_levels(df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float), SWING_N)
    t = pd.DataFrame({"wib": (df["ts"] + pd.Timedelta(hours=7)).dt.tz_localize(None),
                      "sig": np.where(buy, "B", np.where(sell, "S", "")), "close": df["close"],
                      "swing_low": sl, "swing_high": sh})
    t = t[t["sig"] != ""].tail(8)
    bt._emit("\n### Cek sinyal UT_SWING BTC 1h A=3 ATR=10 (waktu = waktu BUKA candle sinyal; B pakai swing_low, S pakai swing_high)\n```\n"
             + t.to_string(index=False) + "\n```")


def run_utswing(exchange):
    df = bt.fetch_ohlcv_cached(exchange, SYMBOL, TF, bt.DAYS_BACK_TF[TF])
    if df is None or len(df) < 1000 or any(g[2] > bt.MAX_GAP_CANDLES for g in bt.find_ohlcv_gaps(df, TF)):
        raise SystemExit("[!] Data 1h tidak layak.")
    df = df.iloc[:-1].reset_index(drop=True)   # candle terakhir mungkin belum close
    n = len(df)
    stop, buy, sell, _ = bt.ut_bot(df, UT_A, UT_C)
    sh, sl = swing_levels(df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float), SWING_N)
    nsig = int(buy.sum() + sell.sum())
    rng = np.random.default_rng(1)
    hit = rng.random(n) < nsig / n
    side = rng.random(n) < 0.5
    sets = {"UT": (buy, sell), "random": (hit & side, hit & ~side)}

    split_ts = df["ts"].iloc[0] + (df["ts"].iloc[-1] - df["ts"].iloc[0]) / 2
    days = max((df["ts"].iloc[-1] - df["ts"].iloc[0]).days, 1)
    cost_net = bt.net_cost(SYMBOL)
    rows, trade_rows = [], []
    for sname, (b, s) in sets.items():
        for vname, lock in (("flip", False), ("lock1R", True)):
            name = f"{sname}_{vname}"
            trades = simulate_swing(df, b, s, sh, sl, stop, lock)
            for cname, cost in (("net", cost_net), ("gross", 0.0)):
                costed = bt.apply_costs(trades, cost)
                cap = [x for x in bt.capital_sim(costed, name, SYMBOL, TF)
                       if x["risk_pct"] == 2.0 and x["policy"] == "skip"][0]
                r = bt.fs_row(TF, UT_A, UT_C, name, costed, split_ts, cap, days)
                r["cost"] = cname
                if costed:
                    pct = lambda rs: round(float(np.mean([t["exit_reason"] in rs for t in costed]) * 100), 1)
                    r["sl%"] = round(float(np.mean([t["sl_pct"] for t in costed]) * 100), 2)
                    r["pct_lock"], r["pct_trail"], r["pct_rev"] = pct(("lock_stop",)), pct(("trail_win",)), pct(("reverse",))
                if cname == "net" and costed and r.get("pf_IS") is not None and r.get("pf_OOS") is not None:
                    r["edge"] = int(r["pf_IS"] >= 1.2 and r["pf_OOS"] >= 1.2 and r["netR"] > 0
                                    and r["n_IS"] >= 30 and r["n_OOS"] >= 30)
                rows.append(r)
                print(f"  {name} {cname}: {len(costed)} trade")
                if cname == "net":
                    trade_rows += [{"variant": name, **t} for t in costed]

    pd.DataFrame(trade_rows).to_csv(OUTPUT_TRADES, index=False)
    fence = "`" * 3
    bt._emit(f"\n### UT_SWING, BTC 1h, UT A={UT_A} ATR={UT_C}, SL di swing terakhir sebelum sinyal (pivot {SWING_N} candle kiri/kanan), long dan short. "
             "flip = SL swing + keluar di sinyal balik. lock1R = flip + SL dikunci di +1R lalu trailing garis UT. "
             f"random = sinyal acak (jumlah sinyal sama, {nsig}) dengan aturan sama. net = biaya {cost_net * 100:.2f}%, gross = tanpa biaya. "
             "sl% = rata-rata jarak SL (% harga), pct_lock / pct_trail / pct_rev = % exit lewat kunci +1R / trailing UT / sinyal balik. "
             "edge = PF IS dan OOS >= 1,2, netR > 0, n >= 30 per paruh\n"
             + fence + "\n" + pd.DataFrame(rows).drop(columns=["tf", "A", "ATR"], errors="ignore").to_string(index=False)
             + "\n" + fence)


def main():
    import ccxt
    exchange = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    exchange.load_markets()
    check_utswing(exchange)
    if os.environ.get("CHECK_ONLY") == "1":
        return
    run_utswing(exchange)


if __name__ == "__main__":
    main()
