#!/usr/bin/env python3
"""Cek data multi-koin (Binance perp USDT): klines 1h, funding, metrics (OI, L/S, taker) per jam. Hanya unduh + laporan cakupan."""
from datetime import datetime, timedelta, timezone

import pandas as pd

import backtest as bt
import derivtree as dt

SYMS = "BTC ETH BNB SOL XRP ADA AVAX LINK DOT LTC BCH TRX ATOM NEAR".split()


def parse_k1h(r):
    r = r[r[0].str.fullmatch(r"\d+", na=False)].copy()
    ot = r[0].astype("int64")
    idx = pd.to_datetime(ot, unit="us" if ot.max() > 1e14 else "ms")
    cols = (("open", 1), ("high", 2), ("low", 3), ("close", 4), ("volume", 5), ("taker_buy_volume", 9))
    return pd.DataFrame({k: pd.to_numeric(r[i], errors="coerce").to_numpy() for k, i in cols}, index=idx)


def parse_m1h(r):
    return dt.parse_m(r).resample("1h").last()


def main():
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    days = [(dt.START + timedelta(days=i)).isoformat() for i in range((end - dt.START).days + 1)]
    months = sorted({d[:7] for d in days})[:-1]
    mk, dk = (lambda t: t.strftime("%Y-%m")), (lambda t: t.date().isoformat())
    rows = []
    for s in SYMS:
        p = s + "USDT"
        for nm, tpl, fn, keys, ko, step in (
            ("klines1h", f"{dt.B}/monthly/klines/{p}/1h/{p}-1h-{{d}}.zip", parse_k1h, months, mk, "1h"),
            ("funding", f"{dt.B}/monthly/fundingRate/{p}/{p}-fundingRate-{{d}}.zip", dt.parse_f, months, mk, "8h"),
            ("metrics1h", f"{dt.B}/daily/metrics/{p}/{p}-metrics-{{d}}.zip", parse_m1h, days, dk, "1h"),
        ):
            try:
                df = dt.load_set(f"{s}_{nm}", tpl, fn, keys, ko).dropna(how="all")
                n = (df.index[-1] - df.index[0]) / pd.Timedelta(step) + 1
                rows.append({"koin": s, "data": nm, "awal": str(df.index[0])[:10], "akhir": str(df.index[-1])[:10],
                             "baris": len(df), "bolong%": round((1 - len(df) / n) * 100, 1)})
            except Exception as e:
                rows.append({"koin": s, "data": nm, "awal": f"GAGAL: {e}"[:60]})
    f = "`" * 3
    bt._emit("\n### Cek data multi-koin (Binance perp). bolong% = baris yang hilang dari yang seharusnya ada (funding bisa negatif kalau intervalnya 4 jam)\n"
             + f + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + f)


if __name__ == "__main__":
    main()
