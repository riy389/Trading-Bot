#!/usr/bin/env python3
"""Evaluasi results_summary.csv dengan kriteria terkunci (tidak diubah per run).

Satu pair LULUS untuk sebuah kombinasi (strategi, mode, timeframe, SL) kalau:
  OOS PF > OOS_PF_MIN, OOS win rate >= OOS_WR_MIN, OOS trade >= OOS_MIN_TRADES,
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
OOS_WR_MIN = 55.0
OOS_MIN_TRADES = 30
TOP_N = 15
KEY = ["strategy", "mode", "timeframe", "stop_loss"]


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
        & (j["win_rate_pct"] >= OOS_WR_MIN)
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
        f"## Evaluasi (OOS PF>{OOS_PF_MIN}, WR>={OOS_WR_MIN:.0f}%, trade>={OOS_MIN_TRADES}, IS PF>1; lulus = mayoritas pair)",
        f"Kombinasi diuji: {len(out)} | LULUS: {int(out['lulus'].sum())}",
        "",
        "### Per strategi (jumlah kombinasi lulus / total)",
    ]
    per = out.groupby("strategy")["lulus"].agg(["sum", "count"])
    for name, r in per.iterrows():
        lines.append(f"- {name}: {int(r['sum'])}/{int(r['count'])}")
    lines += ["", f"### Top {TOP_N} (urut jumlah pair lulus)", "```"]
    cols = ["strategy", "mode", "timeframe", "stop_loss", "pass_pairs", "pairs",
            "med_pf_is", "med_pf_oos", "med_wr_oos", "med_ret_oos", "med_dd_oos", "med_n_oos"]
    top = out.head(TOP_N)[cols].copy()
    top["strategy"] = top["strategy"].str.replace("RegimeSwitch", "RS").str.replace("TrendFollowing_EMA21_55", "Trend").str.replace("MeanReversion_RSI14", "MeanRev")
    lines.append(top.round(2).to_string(index=False))
    lines.append("```")
    return "\n".join(lines)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else CSV
    text = render(evaluate(load(path)))
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
