#!/usr/bin/env python3
"""
PAPER TRADE UT BOT SWING 4h. Aturan beku dari utswing4h.py (UT A=6, ATR 10, EMA200 4h, SL awal = R, trailing garis UT, sinyal balik menutup).
Tanpa uang sungguhan. Jalan tiap 4 jam di Actions: candle 4h Bitget, 14 koin (BTC hanya referensi saat evaluasi).
Entry/exit virtual di open candle berikutnya. State: paper/state.json (hanya berubah kalau ada entry/exit), log: paper/trades.csv
(R kotor, biaya R = 0,22% / jarak SL; funding dihitung belakangan dari lama tahan). Telegram opsional lewat env TG_TOKEN dan TG_CHAT.
"""
import json
import os
import urllib.parse
import urllib.request

import ccxt
import numpy as np
import pandas as pd

from datacoins import SYMS
from derivrule1h import COST
from utswing4h import MAIN, WARM, ut

STATE, LOG, H4 = "paper/state.json", "paper/trades.csv", pd.Timedelta(hours=4)


def tg(msg):
    print(msg)
    t, c = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT")
    if t and c:
        try:
            urllib.request.urlopen(f"https://api.telegram.org/bot{t}/sendMessage", urllib.parse.urlencode({"chat_id": c, "text": msg}).encode(), timeout=20)
        except Exception as e:
            print("telegram gagal:", e)


def candles(ex, s):
    d = pd.DataFrame(ex.fetch_ohlcv(f"{s}/USDT:USDT", "4h", limit=1000), columns=["ms", "o", "h", "l", "c", "v"])
    d["ts"] = pd.to_datetime(d["ms"], unit="ms")
    return d.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def close(s, p, x, t, st):
    r, e, d = p["r"], p["e"], p["d"]
    row = {"coin": s, "dir": d, "t_in": p["t"], "t_out": str(t), "entry": e, "exit": x, "R_gross": round(d * (x - e) / r, 4), "R_fee": round(COST * e / r, 4)}
    os.makedirs("paper", exist_ok=True)
    pd.DataFrame([row]).to_csv(LOG, mode="a", header=not os.path.exists(LOG), index=False)
    del st["pos"][s]
    tg(f"EXIT {'LONG' if d > 0 else 'SHORT'} {s} @ {x:g} (masuk {e:g}) R kotor {row['R_gross']:+.2f}")


def step(s, d, st, now):
    cl = d[d["ts"] + H4 <= now].reset_index(drop=True)
    if len(cl) < 400:
        print(s, "data kurang")
        return
    nxt = d[d["ts"] == cl["ts"].iloc[-1] + H4]
    jmax = len(cl) - 1 if len(nxt) else len(cl) - 2
    ts, o = cl["ts"], np.r_[cl["o"].to_numpy(float), nxt["o"].to_numpy(float)]
    h, l, c = (cl[x].to_numpy(float) for x in ("h", "l", "c"))
    last = st["last"].get(s)
    if last is None:
        st["last"][s] = str(ts.iloc[jmax])
        return
    stop, nl, sig = ut(c, h, l, MAIN)
    ema = pd.Series(c).ewm(span=200, adjust=False).mean().to_numpy()
    ev = False
    for j in range(jmax + 1):
        if ts.iloc[j] <= pd.Timestamp(last):
            continue
        p = st["pos"].get(s)
        if p:
            eff = max(p["sl"], stop[j - 1]) if p["d"] > 0 else min(p["sl"], stop[j - 1])
            if p["d"] * (o[j] - eff) <= 0:
                close(s, p, o[j], ts.iloc[j], st)
                ev = True
            elif p["d"] * ((l[j] if p["d"] > 0 else h[j]) - eff) <= 0:
                close(s, p, eff, ts.iloc[j] + H4, st)
                ev = True
        p = st["pos"].get(s)
        if p and sig[j] == -p["d"]:
            close(s, p, o[j + 1], ts.iloc[j] + H4, st)
            ev = True
        d_ = int(sig[j])
        if s not in st["pos"] and d_ != 0 and j >= WARM and ((d_ > 0 and c[j] > ema[j]) or (d_ < 0 and c[j] < ema[j])) \
                and not (d_ > 0 and o[j + 1] <= stop[j]) and not (d_ < 0 and o[j + 1] >= stop[j]):
            e, r = float(o[j + 1]), float(nl[j])
            st["pos"][s] = {"d": d_, "e": e, "r": r, "sl": e - d_ * r, "t": str(ts.iloc[j] + H4)}
            ev = True
            tg(f"ENTRY {'LONG' if d_ > 0 else 'SHORT'} {s} @ {e:g}, SL awal {e - d_ * r:g}, R {r / e * 100:.1f}%")
    if ev:
        st["last"][s] = str(ts.iloc[jmax])


def main():
    os.makedirs("paper", exist_ok=True)
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"pos": {}, "last": {}}
    before = json.dumps(st, sort_keys=True)
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    now = pd.Timestamp.now("UTC").tz_localize(None)
    for s in SYMS:
        try:
            step(s, candles(ex, s), st, now)
        except Exception as e:
            print(s, "GAGAL:", e)
    print("posisi terbuka:", {k: ("LONG" if v["d"] > 0 else "SHORT") for k, v in st["pos"].items()} or "tidak ada")
    if json.dumps(st, sort_keys=True) != before:
        json.dump(st, open(STATE, "w"), indent=1)


if __name__ == "__main__":
    main()
