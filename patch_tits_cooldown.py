import re

path = "backtest.py"
with open(path) as f:
    src = f.read()

# 1) SL geser ke candle SEBELUM candle sinyal (cara penulis), bukan candle sinyal itu sendiri
anchor1 = "TT_SL_LAG = 0                # B: SL = open candle HA sinyal (candle terakhir yang sudah close saat entry); 1 = open candle HA sebelumnya"
assert anchor1 in src, "anchor1 (TT_SL_LAG) tidak ketemu, cek backtest.py"
src = src.replace(
    anchor1,
    "TT_SL_LAG = 1                # B: SL = open candle HA sinyal (candle terakhir yang sudah close saat entry); 1 = open candle HA sebelumnya\n"
    "TT_COOLDOWN = 3               # B: jumlah candle jeda setelah exit kena SL sebelum boleh entry baru lagi (0 = tanpa cooldown)",
)

# 2) simulate_ut: tambah parameter cooldown opsional (default None -> perilaku lama tidak berubah).
#    Setelah exit SL (exit_reason stop_loss/be_stop/lock_stop), blokir entry baru sampai `cooldown` candle berlalu.
old_sig = (
    "def simulate_ut(df, ut, session, allow_short, trail_on=True, tp_on=False, be_r=None, bias=None,\n"
    "                be_cost=BE_COST, lock=None, close_end=False, sl_fixed=None, tp_r=TP_R, reverse=True):"
)
assert old_sig in src, "signature simulate_ut tidak ketemu, cek backtest.py"
new_sig = (
    "def simulate_ut(df, ut, session, allow_short, trail_on=True, tp_on=False, be_r=None, bias=None,\n"
    "                be_cost=BE_COST, lock=None, close_end=False, sl_fixed=None, tp_r=TP_R, reverse=True,\n"
    "                cooldown=None):"
)
src = src.replace(old_sig, new_sig)

old_init = '    trades, pos, pending = [], None, None\n    for i in range(1, len(df)):'
assert old_init in src, "init trades/pos/pending tidak ketemu, cek backtest.py"
new_init = (
    '    trades, pos, pending = [], None, None\n'
    '    cooldown_until = -1   # index candle: entry baru diblokir selama i <= cooldown_until\n'
    '    for i in range(1, len(df)):'
)
src = src.replace(old_init, new_init)

old_ok = (
    '                ok = (nl > 0 and nl / e >= MIN_SL_PCT and not math.isnan(tr_sig)\n'
    '                      and (bias is None or bias[i - 1] == sign)  # filter arah EMA200 4h\n'
    '                      and sign * (e - tr_sig) > 0                    # open belum melewati garis UT\n'
    '                      and (session == "free" or tod[i] <= SESSION_LAST_ENTRY_MIN))'
)
assert old_ok in src, "blok ok= (syarat entry) tidak ketemu, cek backtest.py"
new_ok = (
    '                ok = (nl > 0 and nl / e >= MIN_SL_PCT and not math.isnan(tr_sig)\n'
    '                      and (bias is None or bias[i - 1] == sign)  # filter arah EMA200 4h\n'
    '                      and sign * (e - tr_sig) > 0                    # open belum melewati garis UT\n'
    '                      and (session == "free" or tod[i] <= SESSION_LAST_ENTRY_MIN)\n'
    '                      and i > cooldown_until)                        # belum lewat masa cooldown setelah SL'
)
src = src.replace(old_ok, new_ok)

old_close = (
    '            if reason is not None:\n'
    '                trades.append(_close(pos, tss[i], fill, reason))\n'
    '                pos = None'
)
assert old_close in src, "blok trades.append(_close(...)) exit tidak ketemu, cek backtest.py"
new_close = (
    '            if reason is not None:\n'
    '                trades.append(_close(pos, tss[i], fill, reason))\n'
    '                pos = None\n'
    '                if cooldown and reason in ("stop_loss", "be_stop", "lock_stop"):\n'
    '                    cooldown_until = i + cooldown'
)
src = src.replace(old_close, new_close)

# 3) run_tits: pakai cooldown di jalur macd_ha (bukan di jalur random pembanding, supaya pembanding tetap murni acak)
old_sim = (
    '        for sname, ut in (("macd_ha", (lvl, upb, dnb, nl)), ("random", (lvl, hit & side, hit & ~side, nl))):\n'
    '            trades = simulate_ut(df, ut, "free", True, False, False, None, None, be_cost=cost_net,\n'
    '                                 close_end=True, reverse=True)'
)
assert old_sim in src, "loop sname/simulate_ut di run_tits tidak ketemu, cek backtest.py"
new_sim = (
    '        for sname, ut in (("macd_ha", (lvl, upb, dnb, nl)), ("random", (lvl, hit & side, hit & ~side, nl))):\n'
    '            cd = TT_COOLDOWN if sname == "macd_ha" else 0\n'
    '            trades = simulate_ut(df, ut, "free", True, False, False, None, None, be_cost=cost_net,\n'
    '                                 close_end=True, reverse=True, cooldown=cd)'
)
src = src.replace(old_sim, new_sim)

with open(path, "w") as f:
    f.write(src)

print("Patch OK: TT_SL_LAG=1, TT_COOLDOWN=3 ditambahkan, simulate_ut dukung cooldown, run_tits pakai cooldown di jalur macd_ha.")
