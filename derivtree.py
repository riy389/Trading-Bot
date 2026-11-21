#!/usr/bin/env python3
"""
EVENT STUDY DERIVATIF + VOLUME. Bukan strategi, tidak ada entry/SL/TP.
Event = definisi eventtree (harga sentuh +-2,7% lebih dulu dalam N candle; 15m N=16, 1h N=12). Harga/candle dari Bitget.
Fitur derivatif dari data.binance.vision BTCUSDT perp (OI, rasio long/short, taker, funding), dipakai hanya setelah candle close
(metrics diberi lag 5 menit, lalu digeser 1 candle lagi di build X). Dua uji:
  ARAH : di antara event, fitur membedakan NAIK vs TURUN?
  EVENT: fitur membedakan event (naik/turun) vs diam (kontrol)?
Skor stabil = min(|AUC-0,5|) di paruh 1 dan 2 (0 kalau arahnya beda). Batas acak = skor terbaik dari label yang digeser (persentil 95).
"""
import io
import os
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

import backtest as bt
import eventtree as et
import eventtree2 as e2

B = "https://data.binance.vision/data/futures/um"
START, CACHE, SHIFTS = date(2020, 9, 1), "cache", 20
KC = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
MC = ["create_time", "symbol", "sum_open_interest", "sum_open_interest_value", "count_toptrader_long_short_ratio",
      "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]
GR = ["OI", "LONGSHORT", "TAKER", "FUNDING", "VOLUME", "HARGA"]


def get(url):
    for _ in range(3):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except Exception:
            pass
    return None


def raw_csv(url):
    b = get(url)
    if b is None:
        return None
    z = zipfile.ZipFile(io.BytesIO(b))
    return pd.read_csv(z.open(z.namelist()[0]), header=None, dtype=str)


def parse_k(r):
    r = r[r[0].str.fullmatch(r"\d+", na=False)].copy()
    r.columns = KC[:r.shape[1]]
    ot = r["open_time"].astype("int64")
    idx = pd.to_datetime(ot, unit="us" if ot.max() > 1e14 else "ms")
    out = pd.DataFrame({k: pd.to_numeric(r[k], errors="coerce").to_numpy() for k in ("volume", "count", "taker_buy_volume")}, index=idx)
    return out


def parse_m(r):
    r = r[r[0] != "create_time"].copy()
    r.columns = MC[:r.shape[1]]
    idx = pd.to_datetime(r["create_time"])
    return pd.DataFrame({k: pd.to_numeric(r[k], errors="coerce").to_numpy() for k in MC[2:]}, index=idx)


def parse_f(r):
    r = r[r[0].str.fullmatch(r"\d+", na=False)]
    return pd.DataFrame({"rate": pd.to_numeric(r[2], errors="coerce").to_numpy()}, index=pd.to_datetime(r[0].astype("int64"), unit="ms"))


def load_set(name, tpl, parse, keys, key_of):
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/bn_{name}.csv.gz"
    old = pd.read_csv(path, index_col=0, parse_dates=True) if os.path.exists(path) else None
    have = {key_of(t) for t in old.index} if old is not None and len(old) else set()
    todo = [k for k in keys if k not in have]

    def one(k):
        r = raw_csv(tpl.format(d=k))
        return parse(r) if r is not None else None
    with ThreadPoolExecutor(32) as ex:
        parts = [p for p in ex.map(one, todo) if p is not None and len(p)]
    df = pd.concat(([old] if old is not None else []) + parts).sort_index()
    df = df[~df.index.duplicated()]
    df.index.name = "t"
    df.to_csv(path)
    print(f"  {name}: {len(df)} baris, {df.index[0]} s/d {df.index[-1]} (unduh baru {len(todo)} file)")
    return df


def naive(ts):
    return ts.dt.tz_convert(None) if ts.dt.tz is not None else ts


def asof(tc, M, tol):
    left = pd.DataFrame({"t": pd.to_datetime(tc.to_numpy()).astype("datetime64[ns]")})
    M2 = M.copy()
    M2.index = pd.to_datetime(M2.index).astype("datetime64[ns]")
    M2.index.name = "t"
    return pd.merge_asof(left, M2.reset_index(), on="t", direction="backward", tolerance=pd.Timedelta(tol))


def feats(df, tf, kl, met, fund):
    mins = {"15m": 15, "1h": 60}[tf]
    n4, n24 = 240 // mins, 1440 // mins
    t0 = naive(df["ts"])
    tc = t0 + pd.Timedelta(minutes=mins)
    o, h, l, c = (df[k].astype(float) for k in ("open", "high", "low", "close"))
    F = pd.DataFrame(index=df.index)
    pc = c.shift(1)
    trp = pd.Series(np.maximum.reduce([h - l, (h - pc).abs(), (l - pc).abs()]), index=df.index) / c
    F["ret1"], F["ret4h"], F["ret24h"] = c / o - 1, c / c.shift(n4) - 1, c / c.shift(n24) - 1
    F["range_pct"], F["atr_pct"] = (h - l) / o, trp.rolling(14).mean()
    F["atr_ratio"] = F["atr_pct"] / F["atr_pct"].rolling(100).mean()
    kb = kl.resample(f"{mins}min").agg({"volume": "sum", "count": "sum", "taker_buy_volume": "sum"}).reindex(pd.DatetimeIndex(t0))
    v = pd.Series(kb["volume"].to_numpy(), index=df.index).where(lambda s: s > 0)
    tb = pd.Series(kb["taker_buy_volume"].to_numpy(), index=df.index).where(v.notna())
    nt = pd.Series(kb["count"].to_numpy(), index=df.index).where(v.notna())
    F["vol_rel"] = v / v.rolling(20).mean()
    F["vol_rel4"] = F["vol_rel"].rolling(4).mean()
    F["vol_z"] = (v - v.rolling(100).mean()) / v.rolling(100).std()
    F["vol_trades_rel"] = nt / nt.rolling(20).mean()
    dl = 2 * tb - v
    F["taker_delta"], F["taker_delta_4h"] = dl / v, dl.rolling(n4).sum() / v.rolling(n4).sum()
    F["taker_delta_24h"] = dl.rolling(n24).sum() / v.rolling(n24).sum()
    m = met.copy()
    m.index = pd.to_datetime(m.index) + pd.Timedelta(minutes=5)
    oi = m["sum_open_interest"]
    M = pd.DataFrame({"oi_chg_1h": oi / oi.shift(12) - 1, "oi_chg_4h": oi / oi.shift(48) - 1, "oi_chg_24h": oi / oi.shift(288) - 1,
                      "ls_acct": np.log(m["count_long_short_ratio"]), "ls_top_pos": np.log(m["sum_toptrader_long_short_ratio"]),
                      "ls_top_cnt": np.log(m["count_toptrader_long_short_ratio"]), "taker_ratio": np.log(m["sum_taker_long_short_vol_ratio"])})
    M["ls_acct_chg_4h"], M["ls_top_pos_chg_4h"] = M["ls_acct"] - M["ls_acct"].shift(48), M["ls_top_pos"] - M["ls_top_pos"].shift(48)
    M = M.replace([np.inf, -np.inf], np.nan)
    a = asof(tc, M, "30min")
    for k in M.columns:
        F[k] = a[k].to_numpy()
    F["oi_x_ret"] = F["oi_chg_4h"] * np.sign(F["ret4h"])
    fr = fund["rate"]
    Z = pd.DataFrame({"fund": fr, "fund_z": (fr - fr.rolling(90).mean()) / fr.rolling(90).std(), "fund_chg": fr - fr.shift(1)})
    a = asof(tc, Z, "9h")
    for k in Z.columns:
        F[k] = a[k].to_numpy()
    return F


def grp(n):
    for p, g in (("oi_", "OI"), ("ls_", "LONGSHORT"), ("taker_", "TAKER"), ("fund", "FUNDING"), ("vol_", "VOLUME")):
        if n.startswith(p):
            return g
    return "HARGA"


def analyse(tf, N, df, kl, met, fund):
    o, h, l = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low"))
    X = feats(df, tf, kl, met, fund).shift(1)
    names, Xv = list(X.columns), X.to_numpy()
    y, _ = et.label_events(o, h, l, N)
    fin = np.isfinite(Xv).all(axis=1)
    fin[:et.WARM] = False
    ts = df["ts"]
    half = ts[fin].iloc[0] + (ts[fin].iloc[-1] - ts[fin].iloc[0]) / 2
    early, yr = (ts < half).to_numpy(), ts.dt.year.to_numpy()

    def masks(yy, kind):
        if kind == "ARAH":
            return ((yy == 1) | (yy == -1)), (yy == 1)
        return ((yy == 0) | (yy == 1) | (yy == -1)), (yy != 0)

    def sc(yy, kind):
        ev, pos = masks(yy, kind)
        mi, mo = fin & ev & early, fin & ev & ~early
        a = np.array([et.auc(Xv[mi, j], pos[mi]) for j in range(len(names))]) - 0.5
        b = np.array([et.auc(Xv[mo, j], pos[mo]) for j in range(len(names))]) - 0.5
        st = np.where(a * b > 0, np.minimum(np.abs(a), np.abs(b)), 0.0)
        return a, b, st

    rng, out = np.random.default_rng(1), []
    for kind in ("ARAH", "EVENT"):
        a, b, st = sc(y, kind)
        thr = float(np.percentile([sc(np.roll(y, int(s)), kind)[2].max() for s in rng.integers(3000, len(y) - 3000, SHIFTS)], 95))
        ev, pos = masks(y, kind)
        base = fin & ev
        summ = []
        for g in GR:
            idx = [j for j, n in enumerate(names) if grp(n) == g]
            summ.append({"tf": tf, "uji": kind, "grup": g, "n_fitur": len(idx), "di_atas_batas_acak": int(sum(st[j] > thr for j in idx)),
                         "skor_terbaik": round(float(max(st[j] for j in idx)), 3)})
        rows = []
        for j in np.argsort(-st)[:10]:
            sgn = np.sign(a[j] + b[j])
            ok_y = tot = 0
            for Y in np.unique(yr[fin]):
                m = base & (yr == Y)
                if pos[m].sum() >= 30 and (~pos[m]).sum() >= 30:
                    tot += 1
                    ok_y += int(np.sign(et.auc(Xv[m, j], pos[m]) - 0.5) == sgn)
            rows.append({"tf": tf, "uji": kind, "fitur": names[j], "grup": grp(names[j]), "AUC_IS": round(0.5 + a[j], 3), "AUC_OOS": round(0.5 + b[j], 3),
                         "skor": round(float(st[j]), 3), "thn_searah": f"{ok_y}/{tot}",
                         "med_pos": round(float(np.nanmedian(Xv[base & pos, j])), 4), "med_neg": round(float(np.nanmedian(Xv[base & ~pos, j])), 4)})
        out.append((summ, rows, thr, int((base & early).sum()), int((base & ~early).sum())))
    return out


def run(ex):
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    days = [(START + timedelta(days=i)).isoformat() for i in range((end - START).days + 1)]
    months = sorted({d[:7] for d in days})[:-1]
    kl = load_set("klines5m", B + "/daily/klines/BTCUSDT/5m/BTCUSDT-5m-{d}.zip", parse_k, days, lambda t: t.date().isoformat())
    met = load_set("metrics", B + "/daily/metrics/BTCUSDT/BTCUSDT-metrics-{d}.zip", parse_m, days, lambda t: t.date().isoformat())
    fund = load_set("funding", B + "/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{d}.zip", parse_f, months, lambda t: t.strftime("%Y-%m"))
    f = "`" * 3
    S, R, notes = [], [], []
    for tf, (N, _) in et.TFS.items():
        df = e2.load(ex, tf)
        for summ, rows, thr, ni, no in analyse(tf, N, df, kl, met, fund):
            S += summ
            R += rows
            notes.append(f"{tf} {summ[0]['uji']}: baris IS {ni}, OOS {no}, batas acak skor = {thr:.3f}")
    bt._emit("\n### Event study derivatif + volume. Event = sentuh +-2,7% lebih dulu dalam N candle (15m N=16, 1h N=12). "
             "ARAH = di antara event, naik vs turun (AUC>0,5: nilai lebih tinggi sebelum NAIK). EVENT = event vs diam (AUC>0,5: nilai lebih tinggi sebelum event). "
             "skor stabil = min(|AUC-0,5|) di dua paruh, 0 kalau arahnya beda. Funding Binance bulanan, jadi ±3 minggu terakhir tidak ikut\n"
             + f + "\n" + "\n".join(notes) + "\n" + f)
    bt._emit("\n### Ringkasan per grup fitur\n" + f + "\n" + pd.DataFrame(S).to_string(index=False) + "\n" + f)
    bt._emit("\n### 10 fitur teratas per TF dan uji (thn_searah = tahun dengan arah AUC sama, dari tahun yang punya data cukup)\n"
             + f + "\n" + pd.DataFrame(R).to_string(index=False) + "\n" + f)


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    run(ex)


if __name__ == "__main__":
    main()
