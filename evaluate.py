#!/usr/bin/env python3
"""Evaluasi results_summary.csv dengan kriteria terkunci (tidak diubah per run).

Satu pair LULUS untuk sebuah kombinasi (strategi, mode, timeframe, SL) kalau:
  OOS PF > OOS_PF_MIN, OOS trade >= OOS_MIN_TRADES,
  dan IS PF > 1.
Kombinasi LULUS kalau mayoritas pair (> 50%) lulus.
Output dicetak ke stdout dan ditambahkan ke $GITHUB_STEP_SUMMARY kalau ada.
"""
import json
import os
import sys
import numpy as np
import pandas as pd

CSV = "results_summary.csv"
OOS_PF_MIN = 1.5
OOS_MIN_TRADES = 30
TOP_N = 15
KEY = ["strategy", "mode", "timeframe", "stop_loss", "take_profit"]


def load(path):
    df = pd.read_csv(path)
    df["profit_factor"] = pd.to_numeric(df["profit_factor"].replace("inf", np.inf), errors="coerce")
    return df


def evaluate(df):
    is_ = df[df["period"] == "IS"].set_index(KEY + ["symbol"])
    oos = df[df["period"] == "OOS"].set_index(KEY + ["symbol"])
    j = oos.join(is_[["profit_factor"]], rsuffix="_is", how="inner")
    j["pass"] = (
        (j["profit_factor"] > OOS_PF_MIN)
        & (j["n_trades"] >= OOS_MIN_TRADES)
        & (j["profit_factor_is"] > 1)
    )
    g = j.reset_index().groupby(KEY)
    out = g.agg(
        pairs=("symbol", "count"),
        pass_pairs=("pass", "sum"),
        med_pf_is=("profit_factor_is", "median"),
        med_pf_oos=("profit_factor", "median"),
        med_wr_oos=("win_rate_pct", "median"),
        med_ret_oos=("total_return_pct", "median"),
        med_dd_oos=("max_drawdown_pct", "median"),
        med_n_oos=("n_trades", "median"),
    ).reset_index()
    out["lulus"] = out["pass_pairs"] > out["pairs"] / 2
    return out.sort_values(["pass_pairs", "med_pf_oos"], ascending=False)


def render(out):
    lines = [
        f"## Evaluasi (OOS PF>{OOS_PF_MIN}, trade>={OOS_MIN_TRADES}, IS PF>1; WR hanya info; lulus = mayoritas pair)",
        f"Kombinasi diuji: {len(out)} | LULUS: {int(out['lulus'].sum())}",
        "",
        "### Per strategi (jumlah kombinasi lulus / total)",
    ]
    per = out.groupby("strategy")["lulus"].agg(["sum", "count"])
    for name, r in per.iterrows():
        lines.append(f"- {name}: {int(r['sum'])}/{int(r['count'])}")
    lines += ["", f"### Top {TOP_N} (urut jumlah pair lulus)", "```"]
    cols = ["strategy", "mode", "timeframe", "stop_loss", "take_profit", "pass_pairs", "pairs",
            "med_pf_is", "med_pf_oos", "med_wr_oos", "med_ret_oos", "med_dd_oos", "med_n_oos"]
    top = out.head(TOP_N)[cols].copy()
    top["strategy"] = top["strategy"].str.replace("RegimeSwitch", "RS").str.replace("TrendFollowing_EMA21_55", "Trend").str.replace("MeanReversion_RSI14", "MeanRev")
    lines.append(top.round(2).to_string(index=False))
    lines.append("```")
    return "\n".join(lines)


def mtf_report(df):
    d = df[(df["period"] == "OOS") & df["strategy"].str.startswith(("MTF_", "FTS_")) & (df["n_trades"] > 0)].copy()
    if d.empty:
        return ""
    d["w"] = d["win_rate_pct"] * d["n_trades"]
    d["wtp"] = d["tp_hit_rate_pct"] * d["n_trades"]
    d["wsl"] = d["sl_hit_rate_pct"] * d["n_trades"]
    t = d.groupby(["strategy", "timeframe", "mode", "take_profit"]).agg(
        pairs=("symbol", "count"), n_oos=("n_trades", "sum"),
        pairs_profit=("total_return_pct", lambda x: int((x > 0).sum())),
        med_pf=("profit_factor", "median"), avg_ret=("total_return_pct", "mean"),
        med_dd=("max_drawdown_pct", "median"), w=("w", "sum"),
        wtp=("wtp", "sum"), wsl=("wsl", "sum"),
    ).reset_index()
    t["wr_info"] = t["w"] / t["n_oos"]
    t["tp_hit"] = t["wtp"] / t["n_oos"]   # % trade keluar kena TP
    t["sl_hit"] = t["wsl"] / t["n_oos"]   # % trade keluar kena SL (termasuk SL yang sudah geser ke impas)
    t = t.drop(columns=["w", "wtp", "wsl"])
    return "\n### MTF per kondisi pasar (OOS, WR hanya info)\n```\n" + t.round(2).to_string(index=False) + "\n```"


def excursion_report(path="excursion.csv"):
    """Kualitas entry vs entry acak (kondisi pasar sama, SL 1R, tanpa TP/BE, time-stop 24 jam).
    Selisih = peluang strategi bergerak +kR searah sebelum kena SL dikurangi entry acak (poin persen)."""
    if not os.path.exists(path):
        return ""
    r = pd.read_csv(path)
    ks = [c for c in r.columns if c.startswith("k")]
    g = r.groupby(["timeframe", "regime", "side", "period", "source"])[["n"] + ks].sum()
    j = g.xs("strategi", level="source").join(g.xs("acak", level="source"), lsuffix="_s", rsuffix="_a", how="inner").reset_index()
    j = j[(j["n_s"] >= 30) & (j["n_a"] >= 30)]
    if j.empty:
        return ""
    out = j[["timeframe", "regime", "side", "period"]].copy()
    out["n_str"], out["n_acak"] = j["n_s"], j["n_a"]
    for k in ks:
        out["d" + k[1:] + "R"] = 100 * (j[k + "_s"] / j["n_s"] - j[k + "_a"] / j["n_a"])
    out["str_1R"] = 100 * j["k1_s"] / j["n_s"]
    out["acak_1R"] = 100 * j["k1_a"] / j["n_a"]
    p = (j["k1_s"] + j["k1_a"]) / (j["n_s"] + j["n_a"])
    out["se_1R"] = 100 * np.sqrt(p * (1 - p) * (1 / j["n_s"] + 1 / j["n_a"]))
    out = out.sort_values(["timeframe", "regime", "side", "period"])
    return ("\n### Kualitas entry vs entry acak (SL 1R, tanpa TP/BE, 24 jam). dXR = selisih peluang bergerak +X R sebelum SL, poin persen."
            " >0 lebih baik dari acak, ~0 tanpa edge arah, <0 lebih buruk (entry telat). Selisih nyata kalau > 2 x se_1R\n```\n"
            + out.round(1).to_string(index=False) + "\n```")


def regime_report(path="regime_forward.csv"):
    if not os.path.exists(path):
        return ""
    r = pd.read_csv(path)
    key = ["symbol", "timeframe", "horizon_h", "period"]
    base = r[r["regime"] == "ALL"].set_index(key)[["mean_fwd_pct", "mean_abs_pct"]]
    r = r[(r["regime"] != "ALL") & (r["n"] > 0)].join(base, on=key, rsuffix="_all")
    sign = r["regime"].map({"TREND_UP": 1, "TREND_DOWN": -1})
    r["edge"] = np.where(r["regime"] == "RANGE",
                         r["mean_abs_pct_all"] - r["mean_abs_pct"],
                         sign * (r["mean_fwd_pct"] - r["mean_fwd_pct_all"]))
    t = r.groupby(["timeframe", "horizon_h", "regime", "period"]).agg(
        pairs=("symbol", "count"), n=("n", "sum"), avg_fwd=("mean_fwd_pct", "mean"),
        avg_edge=("edge", "mean"), pairs_ok=("edge", lambda x: int((x > 0).sum())),
    ).reset_index()
    return ("\n### Tes label regime (return ke depan; edge>0 = label benar; lulus: OOS avg_edge>0 dan pairs_ok>=7)\n```\n"
            + t.round(4).to_string(index=False) + "\n```")


def ut_report(df):
    """UT BTC: hitungan exit per varian (gross, 2 tahun penuh) + PF IS/OOS."""
    d = df[df["strategy"].str.startswith("UT_") & df["strategy"].str.endswith("_gross")]
    if d.empty or d["symbol"].nunique() > 1:
        return ""
    keys = ["strategy", "timeframe"]
    cols = ["n_trades", "n_sl", "n_tp_fixed", "n_trail_win", "n_trail_loss", "n_be", "n_session", "n_reverse",
            "win_rate_pct", "profit_factor", "avg_gross_R"]
    if not set(cols) <= set(d.columns):
        return ""
    t = d[d["period"] == "full"].set_index(keys)[cols].copy()
    pf = d.pivot_table(index=keys, columns="period", values="profit_factor", aggfunc="median")
    t["pf_is"], t["pf_oos"] = pf.get("IS"), pf.get("OOS")
    nt = df[df["strategy"].str.startswith("UT_") & ~df["strategy"].str.endswith("_gross") & (df["period"] == "full")].copy()
    nt["strategy"] = nt["strategy"] + "_gross"
    nt = nt.set_index(keys)[["profit_factor", "avg_net_R"]]
    t["pf_net"], t["netR"] = nt["profit_factor"], nt["avg_net_R"]
    t = t.reset_index()
    t["strategy"] = t["strategy"].str.replace(r"^UT_a([^_]+)_c(\d+)_", r"\1/\2 ", regex=True).str.replace("_gross", "", regex=False)
    t["timeframe"] = pd.Categorical(t["timeframe"], ["5m", "15m", "1h", "4h"], ordered=True)
    t = t.sort_values(["strategy", "timeframe"]).rename(columns={
        "n_trades": "n", "n_sl": "sl", "n_tp_fixed": "tp", "n_trail_win": "trW", "n_trail_loss": "trL", "n_be": "be",
        "n_session": "sess", "n_reverse": "rev", "win_rate_pct": "wr", "profit_factor": "pf", "avg_gross_R": "avgR"})
    return ("\n### UT BTC: exit per varian (gross, 2 tahun). sl=kena SL 1R, tp=TP tetap, trW/trL=trailing untung/rugi,"
            " sess=tutup paksa, rev=sinyal balik, wr=% trade untung\n```\n" + t.round(2).to_string(index=False) + "\n```")


def mfe_report(df):
    """Dari trade rugi (gross, varian trail), berapa yang sebelumnya sempat untung >= 0.5R / 1R / 1.5R."""
    d = df[df["strategy"].str.startswith("UT_") & df["strategy"].str.endswith("_trail_gross") & (df["period"] == "full")]
    if d.empty or d["symbol"].nunique() > 1 or "loss_mfe10" not in d.columns:
        return ""
    t = d[["strategy", "timeframe", "n_trades", "n_loss", "loss_mfe05", "loss_mfe10", "loss_mfe15"]].copy()
    for c in ("05", "10", "15"):
        t["pct" + c] = 100 * t["loss_mfe" + c] / t["n_loss"]
    t["strategy"] = t["strategy"].str.replace(r"^UT_a([^_]+)_c(\d+)_", r"\1/\2 ", regex=True).str.replace("_gross", "", regex=False)
    t["timeframe"] = pd.Categorical(t["timeframe"], ["5m", "15m", "1h", "4h"], ordered=True)
    t = t.sort_values(["strategy", "timeframe"]).rename(columns={
        "n_trades": "n", "n_loss": "rugi", "loss_mfe05": "m05", "loss_mfe10": "m10", "loss_mfe15": "m15"})
    return ("\n### Trade rugi yang sempat untung dulu (gross, trail). m05/m10/m15 = jumlah rugi yang pernah >= 0.5R/1R/1.5R,"
            " pct = % dari rugi. Batas atas manfaat BEP, belum menghitung pemenang yang terpotong\n```\n"
            + t.round(1).to_string(index=False) + "\n```")


def capital_report(path="capital_sim.csv"):
    # Modal kecil (net, periode full), median antar pair. eqN = equity akhir USDT dgn risiko N%/trade (posisi dinaikkan ke min $5);
    # ddN = max DD % median; skipN = % trade yang dilewati kalau posisi <$5 tidak dinaikkan; ruin10 = jumlah pair yang tak bisa buka posisi lagi (risiko 10%)
    if not os.path.exists(path):
        return ""
    r = pd.read_csv(path)
    key = ["strategy", "timeframe"]
    m5 = r[r["policy"] == "min5"]
    t = m5.pivot_table(index=key, columns="risk_pct", values="final_eq", aggfunc="median")
    t.columns = [f"eq{int(c)}" for c in t.columns]
    dd = m5[m5["risk_pct"].isin([5, 10])].pivot_table(index=key, columns="risk_pct", values="max_dd_pct", aggfunc="median")
    dd.columns = [f"dd{int(c)}" for c in dd.columns]
    sk = r[(r["policy"] == "skip") & r["risk_pct"].isin([1, 2])].copy()
    sk["skip_pct"] = 100 * sk["n_skip"] / sk["n_trades"].replace(0, np.nan)
    sk = sk.pivot_table(index=key, columns="risk_pct", values="skip_pct", aggfunc="median")
    sk.columns = [f"skip{int(c)}" for c in sk.columns]
    base = m5[m5["risk_pct"] == 1].groupby(key).agg(pairs=("symbol", "nunique"), n=("n_trades", "median"))
    ruin = m5[m5["risk_pct"] == 10].groupby(key)["ruin"].sum().rename("ruin10")
    t = base.join(t).join(dd).join(sk).join(ruin).reset_index()
    t["strategy"] = t["strategy"].str.replace(r"^UT_a([^_]+)_c(\d+)_", r"\1/\2 ", regex=True)
    t["timeframe"] = pd.Categorical(t["timeframe"], ["5m", "15m", "1h", "4h"], ordered=True)
    t = t.sort_values(["timeframe", "strategy"])
    return ("\n### Simulasi modal $10 (net, full). eqN = equity akhir USDT di risiko N%/trade, posisi dinaikkan ke min $5;"
            " ddN = max DD % ; skipN = % trade dilewati kalau tidak dinaikkan; ruin10 = pair yang mati di risiko 10%. Median antar pair\n```\n"
            + t.round(1).to_string(index=False) + "\n```")



def evaluate_eth(summary_path="eth_momentum_summary.json",
                 trades_path="eth_momentum_trades.csv"):
    """Evaluate ETH Momentum Breakout and publish a GitHub Actions summary."""
    with open(summary_path) as f:
        meta = json.load(f)

    df = pd.read_csv(trades_path)
    if not df.empty:
        df = df.sort_values("entry_ts").reset_index(drop=True)
        df["return_pct"] = pd.to_numeric(df["return_pct"], errors="coerce")
        df = df.dropna(subset=["return_pct"])

    n = len(df)
    wins = df[df["return_pct"] > 0]["return_pct"] if n else pd.Series(dtype=float)
    losses = df[df["return_pct"] <= 0]["return_pct"] if n else pd.Series(dtype=float)

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    pf = gross_profit / gross_loss if gross_loss else float("inf")
    wr = len(wins) / n * 100 if n else 0
    total_return = ((1 + df["return_pct"] / 100).prod() - 1) * 100 if n else 0

    if n:
        equity = (1 + df["return_pct"] / 100).cumprod()
        max_dd = ((equity / equity.cummax()) - 1).min() * 100
    else:
        max_dd = 0

    avg_win = wins.mean() if len(wins) else 0
    avg_loss = losses.mean() if len(losses) else 0
    payoff = avg_win / abs(avg_loss) if avg_loss else float("inf")

    def fmt(value, digits=2):
        if value == float("inf"):
            return "∞"
        return f"{value:.{digits}f}"

    lines = [
        "# ETH Momentum Breakout — Evaluasi Backtest",
        "",
        f"**Pair:** {meta.get('symbol', 'ETH/USDT:USDT')}  ",
        f"**Timeframe:** {meta.get('timeframe', '2h')}  ",
        f"**Data candle:** {meta.get('period_start', '?')} → {meta.get('period_end', '?')}  ",
        f"**Jumlah candle:** {meta.get('candles', '?')}  ",
        "",
        "## Ringkasan performa",
        "",
        "| Metrik | Hasil |",
        "|---|---:|",
        f"| Total trade | {n} |",
        f"| Win rate | {fmt(wr)}% |",
        f"| Profit factor | {fmt(pf)} |",
        f"| Compounded return | {fmt(total_return)}% |",
        f"| Max drawdown (equity per trade) | {fmt(max_dd)}% |",
        f"| Average win | {fmt(avg_win)}% |",
        f"| Average loss | {fmt(avg_loss)}% |",
        f"| Average win/loss ratio | {fmt(payoff)} |",
        "",
        "## Exit reason",
        "",
        "| Exit | Jumlah |",
        "|---|---:|",
    ]

    if n:
        for name, count in df["exit_reason"].value_counts().items():
            lines.append(f"| {name} | {count} |")
    else:
        lines.append("| Tidak ada trade | 0 |")

    lines += [
        "",
        "## Hasil per tahun",
        "",
        "| Tahun | Trade | Win rate | PF | Return compounded |",
        "|---:|---:|---:|---:|---:|",
    ]

    if n:
        df["year"] = pd.to_datetime(df["entry_ts"], utc=True).dt.year
        for year, group in df.groupby("year"):
            w = group[group["return_pct"] > 0]["return_pct"]
            l = group[group["return_pct"] <= 0]["return_pct"]
            gp, gl = w.sum(), abs(l.sum())
            year_pf = gp / gl if gl else float("inf")
            year_return = ((1 + group["return_pct"] / 100).prod() - 1) * 100
            year_wr = len(w) / len(group) * 100
            lines.append(
                f"| {year} | {len(group)} | {fmt(year_wr)}% | "
                f"{fmt(year_pf)} | {fmt(year_return)}% |"
            )
    else:
        lines.append("| - | 0 | - | - | - |")

    lines += [
        "",
        "## Hasil berdasarkan regime",
        "",
        "| Regime | Trade | Win rate | PF | Avg return/trade |",
        "|---:|---:|---:|---:|---:|",
    ]

    if n and "regime" in df.columns:
        for regime, group in df.groupby("regime"):
            w = group[group["return_pct"] > 0]["return_pct"]
            l = group[group["return_pct"] <= 0]["return_pct"]
            gp, gl = w.sum(), abs(l.sum())
            regime_pf = gp / gl if gl else float("inf")
            regime_wr = len(w) / len(group) * 100
            avg_ret = group["return_pct"].mean()
            label = {1: "HOT", 0: "NORMAL", -1: "COLD"}.get(
                int(regime), str(regime)
            )
            lines.append(
                f"| {label} | {len(group)} | {fmt(regime_wr)}% | "
                f"{fmt(regime_pf)} | {fmt(avg_ret)}% |"
            )
    else:
        lines.append("| - | 0 | - | - | - |")

    lines += [
        "",
        "## Pembanding README",
        "",
        "| Metrik | README | Hasil backtest ini |",
        "|---|---:|---:|",
        f"| Return | +197% | {fmt(total_return)}% |",
        f"| Profit factor | 3.08 | {fmt(pf)} |",
        f"| Win rate | 56% | {fmt(wr)}% |",
        f"| Max drawdown | sekitar 14% | {fmt(abs(max_dd))}% |",
        "",
        "> Angka README adalah angka referensi, bukan kriteria kelulusan otomatis. "
        "Perbandingan harus memperhatikan kesamaan data, periode, dan asumsi eksekusi.",
        "",
        "## Catatan validitas",
        "",
        "- Backtest ini menggunakan pendekatan OHLC, bukan tick replay.",
        "- Belum ada pemisahan IS/OOS pada laporan ini.",
        "- Max drawdown dihitung dari equity berurutan per trade.",
        f"- Data gap lebih dari 3 candle: {meta.get('data_gaps_over_3_candles', 'tidak tersedia')}.",
        "- Hasil ini belum membuktikan robustness out-of-sample.",
        "",
        f"**Catatan eksekusi:** {meta.get('note', 'Tidak ada catatan tambahan.')}",
    ]

    report = "\n".join(lines) + "\n"
    Path("eth_momentum_evaluation.md").write_text(report)
    print(report)

    github_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if github_summary:
        with open(github_summary, "a") as f:
            f.write(report)


def main_legacy():
    path = sys.argv[1] if len(sys.argv) > 1 else CSV
    df = load(path)
    text = render(evaluate(df)) + ut_report(df) + mfe_report(df) + mtf_report(df) + regime_report() + excursion_report() + capital_report()
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(text + "\n")



def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--eth":
        evaluate_eth()
    else:
        main_legacy()


if __name__ == "__main__":
    main()
