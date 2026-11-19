#!/usr/bin/env python3
"""TES KRONOS di BTC 1h: apakah prediksi Kronos (bobot pretrained, tanpa fine-tune) berguna?
Konteks 400 candle 1h (OHLC saja), prediksi 12 candle. Arah prediksi vs arah sebenarnya.
Pembanding acak: prediksi digeser melingkar 500x. Kriteria dikunci sebelum lihat hasil."""
import sys, time
sys.path.insert(0, "Kronos")
import numpy as np
import pandas as pd

import backtest as bt
import eventtree2 as e2
from model import Kronos, KronosTokenizer, KronosPredictor

LB, H, SC, BATCH = 400, 12, 3, 8
BUDGET_MIN, MIN_N, FEE = 220, 300, 0.16
CLEAN_FROM = pd.Timestamp("2025-09-01")
OUT = "kronos_preds.csv"


def forecast(pr, ohlc, ts, idxs):
    xs = [ohlc.iloc[i - LB + 1:i + 1].reset_index(drop=True) for i in idxs]
    xt = [ts.iloc[i - LB + 1:i + 1].reset_index(drop=True) for i in idxs]
    yt = [pd.Series(pd.date_range(ts.iloc[i] + pd.Timedelta(hours=1), periods=H, freq="h")) for i in idxs]
    out = pr.predict_batch(df_list=xs, x_timestamp_list=xt, y_timestamp_list=yt,
                           pred_len=H, T=1.0, top_p=0.9, sample_count=SC, verbose=False)
    return [float(o["close"].iloc[-1]) for o in out]


def analyze(df):
    df = df.sort_values("idx").reset_index(drop=True)
    pred = (df.pred_close / df.c0 - 1).to_numpy()
    act = (df.act_close / df.c0 - 1).to_numpy()
    t = pd.to_datetime(df["ts"])
    n = len(df)
    sg = np.sign(pred)
    ret = sg * act
    ks = np.random.default_rng(1).integers(1, n, 500)
    null = np.array([(np.roll(sg, k) * act).mean() for k in ks])
    m = ret.mean()
    z = (m - null.mean()) / (null.std() + 1e-12)
    ic = pd.Series(pred).corr(pd.Series(act), method="spearman")
    nic = np.array([pd.Series(np.roll(pred, k)).corr(pd.Series(act), method="spearman") for k in ks[:200]])
    zic = (ic - nic.mean()) / (nic.std() + 1e-12)
    half = t.median()
    m_is = ret[(t <= half).to_numpy()].mean()
    m_oos = ret[(t > half).to_numpy()].mean()
    yr = t.dt.year.to_numpy()
    yn = yp = 0
    for y in np.unique(yr):
        k = yr == y
        if k.sum() < 10:
            continue
        yn += 1
        yp += int(ret[k].mean() > 0)
    up = pred > 0
    cl = (t >= CLEAN_FROM).to_numpy()
    ok = z >= 2 and m_is > 0 and m_oos > 0 and yp >= yn - 1
    res = {
        "n_prediksi": n,
        "prediksi_naik%": round(up.mean() * 100, 1),
        "akurasi_arah%": round((sg == np.sign(act)).mean() * 100, 1),
        "mean_searah%": round(m * 100, 3),
        "info_net%": round(m * 100 - FEE, 3),
        "long_saja%": round(act[up].mean() * 100, 3) if up.any() else "",
        "short_saja%": round(-act[~up].mean() * 100, 3) if (~up).any() else "",
        "buy_hold_rata2%": round(act.mean() * 100, 3),
        "IS%": round(m_is * 100, 3),
        "OOS%": round(m_oos * 100, 3),
        "tahun_ok": f"{yp}/{yn}",
        "z": round(float(z), 2),
        "IC_spearman": round(float(ic), 3),
        "z_IC": round(float(zic), 2),
        "bersih_n(>=Sep2025)": int(cl.sum()),
        "bersih_mean%": round(ret[cl].mean() * 100, 3) if cl.sum() else "",
        "LULUS": "YA" if ok else "tidak",
    }
    fence = "`" * 3
    bt._emit("\n### Tes Kronos-small BTC 1h (prediksi 12 jam)\n" + fence + "\n"
             + pd.Series(res).to_string() + "\n" + fence)


def main():
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    d = e2.load(ex, "1h")
    ts = pd.to_datetime(d["ts"], utc=True).dt.tz_localize(None).reset_index(drop=True)
    ohlc = d[["open", "high", "low", "close"]].astype(float).reset_index(drop=True)
    c = ohlc["close"].to_numpy()
    cand = np.arange(LB, len(c) - H)

    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    pr = KronosPredictor(mdl, tok, max_context=512)

    t0 = time.time()
    forecast(pr, ohlc, ts, cand[::len(cand) // BATCH][:BATCH])
    per = (time.time() - t0) / BATCH
    N = min(int(BUDGET_MIN * 60 / per), len(cand) // 24)
    print(f"BENCH: {per:.1f} detik/prediksi -> muat {N} prediksi dalam {BUDGET_MIN} menit", flush=True)
    if N < MIN_N:
        print(f"TERLALU LAMBAT: hanya muat {N} (< {MIN_N}). Berhenti.")
        sys.exit(1)
    sel = cand[::len(cand) // N][:N]
    order = np.random.default_rng(7).permutation(sel)

    rows, start = [], time.time()
    for b in range(0, len(order), BATCH):
        idxs = order[b:b + BATCH]
        for i, p in zip(idxs, forecast(pr, ohlc, ts, idxs)):
            rows.append({"idx": int(i), "ts": str(ts.iloc[i]), "c0": c[i], "pred_close": p, "act_close": c[i + H]})
        pd.DataFrame(rows).to_csv(OUT, index=False)
        el = time.time() - start
        print(f"{len(rows)}/{N} selesai, {el/60:.0f} menit", flush=True)
        if el > BUDGET_MIN * 60 * 1.3:
            print("Batas waktu, analisis dengan data yang ada.")
            break
    analyze(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
