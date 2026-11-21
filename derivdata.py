#!/usr/bin/env python3
"""
CEK DATA DERIVATIF (belum strategi). Sumber: data.binance.vision (BTCUSDT perp) + funding history Bitget via ccxt.
Cek: tanggal pertama/terakhir, hari ada/hilang per dataset, contoh isi kolom, likuidasi ada atau tidak.
"""
import io
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import pandas as pd

import backtest as bt
import eventtree as et

B = "https://data.binance.vision/data/futures/um"
START = date(2019, 9, 1)
DAILY = {
    "metrics (OI, long/short, taker ratio) 5m": B + "/daily/metrics/BTCUSDT/BTCUSDT-metrics-{d}.zip",
    "klines 5m (ada taker buy volume)": B + "/daily/klines/BTCUSDT/5m/BTCUSDT-5m-{d}.zip",
}
LIQ = B + "/daily/liquidationSnapshot/BTCUSDT/BTCUSDT-liquidationSnapshot-{d}.zip"
FUND = B + "/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{d}.zip"


def head(url):
    for _ in range(3):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=20).status == 200
        except urllib.error.HTTPError:
            return False
        except Exception:
            continue
    return False


def sample(url):
    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=60).read()))
    return pd.read_csv(z.open(z.namelist()[0]))


def presence(tpl, days):
    with ThreadPoolExecutor(24) as ex:
        ok = list(ex.map(lambda x: head(tpl.format(d=x.isoformat())), days))
    return [x for x, o in zip(days, ok) if o]


def summary(name, have, days):
    if not have:
        return {"dataset": name, "pertama": "-", "terakhir": "-", "ada": 0, "hilang": "-", "gap_terbesar_hari": "-"}
    sub = [x for x in days if have[0] <= x <= have[-1]]
    miss = sorted(set(sub) - set(have))
    run = best = 0
    prev = None
    for m in miss:
        run = run + 1 if prev and (m - prev).days == 1 else 1
        best, prev = max(best, run), m
    return {"dataset": name, "pertama": have[0].isoformat(), "terakhir": have[-1].isoformat(), "ada": len(have), "hilang": len(miss), "gap_terbesar_hari": best}


def run(ex):
    f = "`" * 3
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    days = [START + timedelta(days=i) for i in range((end - START).days + 1)]
    rows, samp = [], []
    for name, tpl in DAILY.items():
        have = presence(tpl, days)
        rows.append(summary(name, have, days))
        if have:
            s = sample(tpl.format(d=have[len(have) // 2].isoformat()))
            samp.append(f"{name} (contoh {have[len(have) // 2]}): {len(s)} baris, kolom = {list(s.columns)}\n{s.head(2).to_string(index=False)}")
    months = sorted({date(x.year, x.month, 1) for x in days})
    fm = presence(FUND, [m.isoformat()[:7] and m for m in months]) if False else []
    with ThreadPoolExecutor(24) as pool:
        okm = list(pool.map(lambda m: head(FUND.format(d=m.strftime("%Y-%m"))), months))
    fm = [m for m, o in zip(months, okm) if o]
    rows.append({"dataset": "fundingRate Binance (bulanan)", "pertama": fm[0].isoformat()[:7] if fm else "-", "terakhir": fm[-1].isoformat()[:7] if fm else "-",
                 "ada": len(fm), "hilang": len(months) - len(fm), "gap_terbesar_hari": "-"})
    liq = [date(y, m, 15) for y in (2020, 2021, 2022, 2024, 2026) for m in (1, 6)]
    liq_ok = [x.isoformat() for x in liq if head(LIQ.format(d=x.isoformat()))]
    try:
        fr, since = [], ex.parse8601("2021-01-01T00:00:00Z")
        for _ in range(400):
            b = ex.fetch_funding_rate_history(et.SYMBOL, since=since, limit=100)
            if not b or b[-1]["timestamp"] + 1 <= since:
                break
            fr += b
            since = b[-1]["timestamp"] + 1
        bg = (f"Bitget funding: {len(fr)} baris, {pd.to_datetime(fr[0]['timestamp'], unit='ms')} s/d {pd.to_datetime(fr[-1]['timestamp'], unit='ms')}" if fr else "Bitget funding: kosong")
    except Exception as e:
        bg = f"Bitget funding gagal: {type(e).__name__} {str(e)[:120]}"
    bt._emit("\n### Cek data derivatif (BTCUSDT perp, harian s/d kemarin)\n" + f + "\n" + pd.DataFrame(rows).to_string(index=False) + "\n" + f)
    bt._emit("\n### Contoh isi file\n" + f + "\n" + "\n\n".join(samp) + "\n" + f)
    bt._emit("\n### Likuidasi dan funding Bitget\n" + f + f"\nliquidationSnapshot ada di tanggal tes: {liq_ok if liq_ok else 'tidak ada satupun'}\n{bg}\n" + f)


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
