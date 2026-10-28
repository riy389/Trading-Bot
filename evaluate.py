#!/usr/bin/env python3
"""Evaluasi results_summary.csv dengan kriteria terkunci (tidak diubah per run).

Satu pair LULUS untuk sebuah kombinasi (strategi, mode, timeframe, SL) kalau:
  OOS PF > OOS_PF_MIN, OOS trade >= OOS_MIN_TRADES,
  dan IS PF > 1.
Kombinasi LULUS kalau mayoritas pair (> 50%) lulus.
Output dicetak ke stdout dan ditambahkan ke $GITHUB_STEP_SUMMARY kalau ada.
"""
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
    d = df[(df["period"] == "OOS") & df["strategy"].str.startswith("MTF_") & (df["n_trades"] > 0)].copy()
    if d.empty:
        return ""
    d["w"] = d["win_rate_pct"] * d["n_trades"]
    t = d.groupby(["strategy", "timeframe", "mode", "take_profit"]).agg(
        pairs=("symbol", "count"), n_oos=("n_trades", "sum"),
        pairs_profit=("total_return_pct", lambda x: int((x > 0).sum())),
        med_pf=("profit_factor", "median"), avg_ret=("total_return_pct", "mean"),
        med_dd=("max_drawdown_pct", "median"), w=("w", "sum"),
    ).reset_index()
    t["wr_info"] = t["w"] / t["n_oos"]
    t = t.drop(columns="w")
    return "\n### MTF per kondisi pasar (OOS, WR hanya info)\n```\n" + t.round(2).to_string(index=False) + "\n```"


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


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else CSV
    df = load(path)
    text = render(evaluate(df)) + mtf_report(df) + regime_report()
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
