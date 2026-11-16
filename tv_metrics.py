#!/usr/bin/env python3
"""
Modul laporan statistik lengkap bergaya TradingView / TradeSearcher.
Dapat di-import oleh supertrend.py, zerolag2.py, meanrev.py, dll.
"""
import numpy as np
import pandas as pd


def generate_tv_report(
    trades_list,
    df: pd.DataFrame,
    tf_name: str,
    variant_name: str,
    initial_capital: float = 10000.0,
    fee_pct: float = 0.0012,       # 0.12% fee round-trip
    slippage_pct: float = 0.0010,  # 0.10% slippage round-trip
    compounding: bool = True,      # True = 100% equity (seperti TV), False = fixed lot
) -> str:
    """
    Format elemen trades_list yang didukung:
      - 6 tuple: (je, pos, ep, px, rd, reason)
      - 7 tuple: (je, jx, pos, ep, px, rd, reason) -> jika menyertakan index exit jx
    """
    if not trades_list:
        return f"### TV Report: {tf_name} | {variant_name}\nTidak ada transaksi.\n"

    ts = pd.to_datetime(df["ts"]).reset_index(drop=True)
    c = df["close"].to_numpy(float)
    n_bars = len(c)
    round_trip_cost = fee_pct + slippage_pct

    # Hitung durasi per bar dalam menit
    if len(ts) > 1:
        bar_minutes = (ts.iloc[1] - ts.iloc[0]).total_seconds() / 60.0
    else:
        bar_minutes = 60.0

    records = []
    equity = initial_capital
    peak_equity = initial_capital
    max_dd_pct = 0.0
    total_commission_usd = 0.0
    equity_returns = []

    for idx_t, tr in enumerate(trades_list):
        if len(tr) == 7:
            je, jx, pos, ep, px, rd, reason = tr
        else:
            je, pos, ep, px, rd, reason = tr
            # Estimasi jx dari entry trade berikutnya jika tidak dicatat eksplisit
            jx = trades_list[idx_t + 1][0] if idx_t + 1 < len(trades_list) else n_bars - 1

        je = int(np.clip(je, 0, n_bars - 1))
        jx = int(np.clip(max(jx, je + 1), 0, n_bars - 1))
        bars_held = max(jx - je, 1)

        gross_ret = ((px - ep) / ep) if pos == 1 else ((ep - px) / ep)
        net_ret = gross_ret - round_trip_cost

        pos_size_usd = equity if compounding else initial_capital
        comm_usd = pos_size_usd * round_trip_cost
        pnl_usd = pos_size_usd * net_ret

        equity += pnl_usd
        total_commission_usd += comm_usd
        equity_returns.append(net_ret)

        if equity > peak_equity:
            peak_equity = equity
        dd_pct = ((peak_equity - equity) / peak_equity) * 100.0 if peak_equity > 0 else 0.0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

        records.append({
            "pos": pos,
            "net_pct": net_ret * 100.0,
            "pnl_usd": pnl_usd,
            "bars": bars_held,
            "reason": reason,
        })

    res = pd.DataFrame(records)
    wins = res[res["net_pct"] > 0]
    losses = res[res["net_pct"] <= 0]
    longs = res[res["pos"] == 1]
    shorts = res[res["pos"] == -1]

    long_wins = longs[longs["net_pct"] > 0]
    long_losses = longs[longs["net_pct"] <= 0]
    short_wins = shorts[shorts["net_pct"] > 0]
    short_losses = shorts[shorts["net_pct"] <= 0]

    total_trades = len(res)
    win_rate = (len(wins) / total_trades) * 100.0
    win_rate_long = (len(long_wins) / len(longs) * 100.0) if len(longs) > 0 else 0.0
    win_rate_short = (len(short_wins) / len(shorts) * 100.0) if len(shorts) > 0 else 0.0

    # Profit Factor (dilabeli Risk/Reward Ratio di TradeSearcher) & Payoff Ratio murni
    gross_profit_pct = wins["net_pct"].sum()
    gross_loss_pct = abs(losses["net_pct"].sum())
    pf_all = (gross_profit_pct / gross_loss_pct) if gross_loss_pct > 0 else 999.0

    pf_long = (long_wins["net_pct"].sum() / abs(long_losses["net_pct"].sum())) if len(long_losses) > 0 else 999.0
    pf_short = (short_wins["net_pct"].sum() / abs(short_losses["net_pct"].sum())) if len(short_losses) > 0 else 999.0

    avg_win_pct = wins["net_pct"].mean() if len(wins) > 0 else 0.0
    avg_loss_pct = abs(losses["net_pct"].mean()) if len(losses) > 0 else 0.0
    payoff_ratio = (avg_win_pct / avg_loss_pct) if avg_loss_pct > 0 else 999.0

    # Durasi rata-rata
    avg_bars = res["bars"].mean()
    total_mins = int(round(avg_bars * bar_minutes))
    dur_h, dur_m = divmod(total_mins, 60)
    dur_str = f"{dur_h}h {dur_m}m"

    # Periode & Buy and Hold
    start_dt = ts.iloc[0]
    end_dt = ts.iloc[-1]
    days_span = max((end_dt - start_dt).days, 1)
    years_span = days_span / 365.25
    buy_hold_pct = ((c[-1] - c[0]) / c[0]) * 100.0

    roi_pct = ((equity - initial_capital) / initial_capital) * 100.0
    net_profit_usd = equity - initial_capital
    comm_pct_of_cap = (total_commission_usd / initial_capital) * 100.0

    # Sharpe & Sortino Ratio (berbasis trade return disetahunkan)
    rets = np.array(equity_returns)
    trades_per_year = total_trades / max(years_span, 0.1)
    mean_r = np.mean(rets)
    std_r = np.std(rets, ddof=1) if len(rets) > 1 else 0.0
    downside = rets[rets < 0]
    down_std = np.std(downside, ddof=1) if len(downside) > 1 else 0.0

    sharpe = (mean_r / std_r) * np.sqrt(trades_per_year) if std_r > 0 else 0.0
    sortino = (mean_r / down_std) * np.sqrt(trades_per_year) if down_std > 0 else 0.0

    open_pnl = 0.0 if res.iloc[-1]["reason"] != "end" else res.iloc[-1]["net_pct"]

    metrics = [
        ("Profit Factor (TV R/R)", f"{pf_all:.2f} (Long {pf_long:.2f} · Short {pf_short:.2f})"),
        ("Payoff Ratio (Avg Win/Loss)", f"{payoff_ratio:.2f}"),
        ("Win Rate %", f"{win_rate:.2f} % (Long {win_rate_long:.2f}% · Short {win_rate_short:.2f}%)"),
        ("Total Trades", f"{total_trades} ({len(wins)}W · {len(losses)}L | Long {len(longs)} · Short {len(shorts)})"),
        ("ROI", f"{roi_pct:.2f} % (vs Buy & Hold {buy_hold_pct:.2f} %)"),
        ("Max Drawdown", f"{max_dd_pct:.2f} %"),
        ("Avg Trade % (Combined)", f"{res['net_pct'].mean():.2f} %"),
        ("Avg Winning Trade %", f"{avg_win_pct:.2f} %"),
        ("Avg Losing Trade %", f"{avg_loss_pct:.2f} %"),
        ("Largest Win / Loss %", f"+{res['net_pct'].max():.2f} % / {res['net_pct'].min():.2f} %"),
        ("Avg Trade Duration", f"{dur_str} ({avg_bars:.1f} bars)"),
        ("Sharpe / Sortino Ratio", f"{sharpe:.3f} / {sortino:.3f}"),
        ("Initial -> End Capital", f"{initial_capital:,.2f} USD -> {equity:,.2f} USD"),
        ("Net Profit", f"{net_profit_usd:,.2f} USD"),
        ("Total Commission + Slippage", f"{comm_pct_of_cap:.2f} % ({total_commission_usd:,.2f} USD)"),
        ("Open P/L", f"{open_pnl:.2f} %"),
        ("Backtest Period", f"{days_span} days ({start_dt.strftime('%Y-%m-%d')} - {end_dt.strftime('%Y-%m-%d')})"),
    ]

    report_df = pd.DataFrame(metrics, columns=["Metric", "Value"])
    fence = "`" * 3
    return f"\n#### Detail Performa: {tf_name} — {variant_name}\n{fence}\n{report_df.to_string(index=False)}\n{fence}\n"
