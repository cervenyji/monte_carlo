import pandas as pd
import json
import os
import numpy as np
from datetime import datetime
from collections import defaultdict
import urllib.request

df = pd.read_csv('in/tables/dbs_branch_network_epb.csv')

# 1. Vyčištění dat
df['branch_type'] = df['branch_type'].astype(str).str.strip()
df['branch_code_clean'] = df['branch_code'].astype(int)

# 2. Definice seznamů
branch_types_y = ['BP', 'BZ', 'PP', 'UP', 'ZP']
special_codes = [902, 904, 905, 906, 909, 919]

# 3. Podmínka
cond1 = (df['branch_code_clean'] >= 1) & (df['branch_code_clean'] <= 675) & (df['branch_type'].isin(branch_types_y))
cond2 = (df['branch_code_clean'].isin(special_codes)) & (df['branch_type'] == 'PR')
df['bns_flag'] = np.where(cond1 | cond2, 'Y', 'N')

# === Příprava dat ===
df["effective_date"] = pd.to_datetime(df["effective_date"])
df_bns = df[df["bns_flag"] == "Y"].copy()
df_bns = df_bns.sort_values(["branch_code", "effective_date"])

# =========================================================
# ČÁST A — Měsíční snapshoty + aktuální stav
# =========================================================
df_bns["year_month"] = df_bns["effective_date"].dt.to_period("M")
all_months = sorted(df_bns["year_month"].unique())

def build_snapshot(cutoff_date, label, is_current=False):
    mask = df_bns["effective_date"] <= cutoff_date
    if not mask.any():
        return None
    snap = (
        df_bns[mask]
        .sort_values("effective_date")
        .groupby("branch_code")
        .last()
        .reset_index()
    )
    total = len(snap)
    opened = snap[~snap["branch_closed"]].shape[0]
    closed = snap[snap["branch_closed"]].shape[0]
    cashless = snap[(~snap["branch_closed"]) & (snap["cashless"] == True)].shape[0]
    non_cashless = opened - cashless
    open_branches = snap[~snap["branch_closed"]]
    new_format = open_branches[
        (open_branches["format"].notna()) & (open_branches["format"] != "")
    ].shape[0]
    old_format = opened - new_format
    return {
        "timestamp": cutoff_date.strftime("%Y-%m-%d"),
        "label": label,
        "total": int(total),
        "opened": int(opened),
        "closed": int(closed),
        "cashless": int(cashless),
        "non_cashless": int(non_cashless),
        "new_format": int(new_format),
        "old_format": int(old_format),
        "is_current": is_current,
    }

history = []
last_data_date = df_bns["effective_date"].max().normalize()
last_data_month = last_data_date.to_period("M")

for month in all_months:
    month_end = month.to_timestamp(how="end").normalize()
    if month == last_data_month and month_end > last_data_date:
        continue
    if month_end > last_data_date:
        continue
    snap = build_snapshot(month_end, str(month))
    if snap:
        history.append(snap)

month_end_of_last = last_data_month.to_timestamp(how="end").normalize()
if last_data_date < month_end_of_last:
    current_label = f"{last_data_month} (aktuální {last_data_date.strftime('%d.%m.')})"
    snap = build_snapshot(last_data_date, current_label, is_current=True)
    if snap:
        history.append(snap)

print(f"📊 Zpracováno {len(history)} snapshotů (včetně aktuálního)")

# =========================================================
# ČÁST A.2 — Agregace na kvartály a roky
# =========================================================
def aggregate_history(history_rows, period):
    if not history_rows:
        return []
    by_period = {}
    for row in history_rows:
        ts = pd.to_datetime(row["timestamp"])
        if period == "Q":
            key = f"{ts.year} Q{(ts.month - 1) // 3 + 1}"
            sort_key = (ts.year, (ts.month - 1) // 3)
        elif period == "Y":
            key = str(ts.year)
            sort_key = (ts.year,)
        else:
            key = row["label"]
            sort_key = (ts,)
        if key not in by_period or row["timestamp"] > by_period[key]["timestamp"]:
            by_period[key] = {**row, "label": key, "_sort": sort_key}
    out = sorted(by_period.values(), key=lambda r: r["_sort"])
    for r in out:
        r.pop("_sort", None)
    return out

history_monthly = history
history_quarterly = aggregate_history(history, "Q")
history_yearly = aggregate_history(history, "Y")

print(f"📊 Agregace: M={len(history_monthly)}, Q={len(history_quarterly)}, Y={len(history_yearly)}")

# =========================================================
# ČÁST B — Detail poboček (timeline změn)
# =========================================================
tracked_cols = [
    "branch_name", "branch_type", "branch_closed", "cashless",
    "format", "address", "city", "region"
]
tracked_cols = [c for c in tracked_cols if c in df_bns.columns]

branches_data = {}
for code, grp in df_bns.groupby("branch_code"):
    grp = grp.sort_values("effective_date").reset_index(drop=True)
    branch_name = str(grp["branch_name"].iloc[-1]) if "branch_name" in grp.columns else f"Pobočka {code}"
    events = []
    prev_row = None
    for i, row in grp.iterrows():
        date_str = row["effective_date"].strftime("%Y-%m-%d")
        if prev_row is None:
            state = {}
            for col in tracked_cols:
                val = row[col]
                if pd.isna(val): val = None
                elif isinstance(val, (np.bool_, bool)): val = bool(val)
                elif isinstance(val, (np.integer,)): val = int(val)
                elif isinstance(val, (np.floating,)): val = float(val)
                else: val = str(val)
                state[col] = val
            events.append({"date": date_str, "type": "initial", "label": "Počáteční stav", "state": state, "changes": []})
        else:
            changes = []
            state = {}
            for col in tracked_cols:
                old_val = prev_row[col]
                new_val = row[col]
                if pd.isna(old_val): old_val = None
                if pd.isna(new_val): new_val = None
                if isinstance(old_val, (np.bool_, bool)): old_val = bool(old_val)
                if isinstance(new_val, (np.bool_, bool)): new_val = bool(new_val)
                if isinstance(old_val, (np.integer,)): old_val = int(old_val)
                if isinstance(new_val, (np.integer,)): new_val = int(new_val)
                if isinstance(old_val, (np.floating,)): old_val = float(old_val)
                if isinstance(new_val, (np.floating,)): new_val = float(new_val)
                if old_val is not None and not isinstance(old_val, (bool, int, float)): old_val = str(old_val)
                if new_val is not None and not isinstance(new_val, (bool, int, float)): new_val = str(new_val)
                state[col] = new_val
                if old_val != new_val:
                    changes.append({"field": col, "old": old_val, "new": new_val})
            if changes:
                evt_type = "change"; label = "Změna"
                for ch in changes:
                    if ch["field"] == "branch_closed":
                        if ch["new"] == True: evt_type = "closed"; label = "Pobočka zavřena"
                        elif ch["new"] == False: evt_type = "reopened"; label = "Pobočka znovu otevřena"
                    elif ch["field"] == "cashless":
                        if ch["new"] == True: evt_type = "cashless"; label = "Přechod na cashless"
                        elif ch["new"] == False: evt_type = "cash_added"; label = "Vrácena hotovost"
                    elif ch["field"] == "format" and ch["new"] and ch["new"] != "":
                        evt_type = "format_change"; label = "Nový formát: " + str(ch["new"])
                events.append({"date": date_str, "type": evt_type, "label": label, "state": state, "changes": changes})
        prev_row = row
    branches_data[int(code)] = {"code": int(code), "name": branch_name, "events": events, "total_changes": len(events) - 1}

sorted_branches = sorted(branches_data.values(), key=lambda b: b["code"])

print(f"✅ Připraveno {len(sorted_branches)} poboček pro timeline")

# =========================================================
# ČÁST B.2 — Posledních N změn napříč sítí
# =========================================================
def categorize_change(field, old_val, new_val):
    if field == "cashless":
        if new_val is True:
            return ("cashless", "Přechod na cashless")
        elif new_val is False:
            return ("cashless", "Vrácena hotovost")
    elif field == "format":
        if new_val and str(new_val) not in ("", "nan", "None"):
            return ("format", f"Nový formát: {new_val}")
    elif field == "branch_closed":
        if new_val is True:
            return ("closed", "Pobočka zavřena")
        elif new_val is False:
            return ("closed", "Pobočka znovu otevřena")
    return None

all_changes = []
for code, bdata in branches_data.items():
    for evt in bdata["events"]:
        if evt["type"] == "initial":
            continue
        for ch in evt["changes"]:
            cat = categorize_change(ch["field"], ch["old"], ch["new"])
            if cat is None:
                continue
            category, label = cat
            all_changes.append({
                "date": evt["date"],
                "branch_code": int(code),
                "branch_name": bdata["name"],
                "category": category,
                "label": label,
                "field": ch["field"],
                "old": ch["old"],
                "new": ch["new"],
            })

all_changes.sort(key=lambda x: (x["date"], -x["branch_code"]), reverse=True)

recent_changes_by_cat = {
    "all": all_changes[:10],
    "cashless": [c for c in all_changes if c["category"] == "cashless"][:10],
    "format": [c for c in all_changes if c["category"] == "format"][:10],
    "closed": [c for c in all_changes if c["category"] == "closed"][:10],
}

print(f"🔔 Změny připraveny: "
      f"all={len(recent_changes_by_cat['all'])}, "
      f"cashless={len(recent_changes_by_cat['cashless'])}, "
      f"format={len(recent_changes_by_cat['format'])}, "
      f"closed={len(recent_changes_by_cat['closed'])}")

# =========================================================
# ČÁST C — Měsíční přírůstky poboček s novým formátem
# =========================================================
format_transitions = []
for code, bdata in branches_data.items():
    for evt in bdata["events"]:
        if evt["type"] == "initial":
            continue
        for ch in evt["changes"]:
            if ch["field"] == "format":
                old_empty = ch["old"] is None or str(ch["old"]).strip() in ("", "nan", "None")
                new_nonempty = ch["new"] is not None and str(ch["new"]).strip() not in ("", "nan", "None")
                if old_empty and new_nonempty:
                    format_transitions.append({
                        "date": evt["date"],
                        "month": evt["date"][:7],
                        "branch_code": int(code),
                        "branch_name": bdata["name"],
                        "format_new": str(ch["new"]),
                    })

format_transitions.sort(key=lambda x: (x["date"], x["branch_code"]))

fmt_by_month = defaultdict(list)
for ft in format_transitions:
    fmt_by_month[ft["month"]].append(ft)

all_fmt_months = sorted(fmt_by_month.keys())
format_monthly_summary = []
for m in all_fmt_months:
    branches_list = fmt_by_month[m]
    format_monthly_summary.append({
        "month": m,
        "count": len(branches_list),
        "branches": branches_list,
    })

print(f"🎨 Přechody na nový formát: {len(format_transitions)} celkem, {len(format_monthly_summary)} měsíců")

# =========================================================
# STAŽENÍ APEXCHARTS — embed do HTML (funguje offline/file://)
# =========================================================
_apex_url = "https://cdn.jsdelivr.net/npm/apexcharts/dist/apexcharts.min.js"
try:
    with urllib.request.urlopen(_apex_url, timeout=15) as _r:
        _apex_js = _r.read().decode("utf-8")
    print(f"📦 ApexCharts stažen ({len(_apex_js)//1024} KB) — bude embedded inline")
except Exception as _e:
    _apex_js = None
    print(f"⚠️  ApexCharts se nepodařilo stáhnout ({_e}) — použije se CDN odkaz")

# =========================================================
# GENEROVÁNÍ JEDNOHO HTML
# =========================================================
history_json = json.dumps(history, ensure_ascii=False)
history_all_json = json.dumps({
    "M": history_monthly,
    "Q": history_quarterly,
    "Y": history_yearly,
}, ensure_ascii=False)
branches_json = json.dumps(sorted_branches, ensure_ascii=False)
recent_changes_json = json.dumps(recent_changes_by_cat, ensure_ascii=False)
format_monthly_json = json.dumps(format_monthly_summary, ensure_ascii=False)

REPORT_FILE = "branch_timeline_report.html"
apex_script_tag = (
    "<script>" + _apex_js + "</script>"
    if _apex_js
    else '<script src="https://cdn.jsdelivr.net/npm/apexcharts/dist/apexcharts.min.js"></script>'
)

html = f"""<!DOCTYPE html>
<html lang="cs">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pobočková síť ČS — Report</title>
{apex_script_tag}
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,700&family=JetBrains+Mono:wght@400;600&display=swap');

  :root {{
    --bg: #f3f5f9;
    --card: #ffffff;
    --text: #1e2330;
    --muted: #64748b;
    --dim: #9ca3b0;
    --border: #dfe2ea;
    --border-lt: #eceef4;
    --accent: #0057b8;
    --accent-lt: #e8f0fe;
    --green: #059669;
    --green-bg: #ecfdf5;
    --red: #dc2626;
    --red-bg: #fef2f2;
    --orange: #d97706;
    --orange-bg: #fffbeb;
    --blue: #2563eb;
    --purple: #7c3aed;
    --purple-bg: #f5f3ff;
    --teal: #0d9488;
    --yellow: #b45309;
    --yellow-bg: #fefce8;
    --cyan: #0891b2;
    --cyan-bg: #ecfeff;
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }}

  .tab-nav {{
    display: flex;
    align-items: center;
    gap: 0;
    background: var(--card);
    border-bottom: 1px solid var(--border);
    padding: 0 32px;
    position: sticky;
    top: 0;
    z-index: 100;
  }}

  .tab-nav .brand {{
    font-weight: 700;
    font-size: 0.95rem;
    color: var(--accent);
    padding: 14px 24px 14px 0;
    border-right: 1px solid var(--border);
    margin-right: 4px;
    white-space: nowrap;
  }}

  .tab-btn {{
    padding: 14px 22px;
    background: none;
    border: none;
    border-bottom: 2px solid transparent;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--muted);
    cursor: pointer;
    transition: all 0.2s;
  }}

  .tab-btn:hover {{ color: var(--text); }}

  .tab-btn.active {{
    color: var(--accent);
    border-bottom-color: var(--accent);
    font-weight: 700;
  }}

  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}

  #tabOverview {{
    padding: 28px 32px;
    max-width: 1200px;
    margin: 0 auto;
  }}

  .ov-header {{
    text-align: center;
    margin-bottom: 28px;
  }}

  .ov-header h1 {{
    font-size: 1.4rem;
    color: var(--accent);
    margin-bottom: 4px;
  }}

  .ov-header p {{
    color: var(--muted);
    font-size: 0.85rem;
  }}

  .ov-header .range {{
    display: inline-block;
    margin-top: 8px;
    background: var(--accent-lt);
    color: var(--accent);
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
  }}

  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 10px;
    margin-bottom: 24px;
  }}

  .kpi {{
    background: var(--card);
    border-radius: 10px;
    padding: 14px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    border-top: 3px solid transparent;
  }}

  .kpi .value {{ font-size: 1.6rem; font-weight: 700; line-height: 1.2; }}
  .kpi .label {{ font-size: 0.75rem; color: var(--muted); margin-top: 3px; }}
  .kpi .delta {{ font-size: 0.72rem; margin-top: 2px; }}
  .delta.up {{ color: var(--green); }}
  .delta.down {{ color: var(--red); }}
  .delta.neutral {{ color: var(--muted); }}

  .chart-card {{
    background: var(--card);
    border-radius: 10px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}

  .chart-card h2 {{ font-size: 0.95rem; margin-bottom: 2px; }}
  .chart-card .sub {{ font-size: 0.78rem; color: var(--muted); margin-bottom: 10px; }}

  .rc-tabs {{
    display: flex;
    gap: 4px;
    background: var(--bg);
    padding: 3px;
    border-radius: 8px;
    border: 1px solid var(--border);
  }}
  .rc-tab {{
    padding: 5px 12px;
    background: transparent;
    border: none;
    border-radius: 5px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--muted);
    cursor: pointer;
    transition: all 0.15s;
  }}
  .rc-tab:hover {{ color: var(--text); }}
  .rc-tab.active {{
    background: var(--card);
    color: var(--accent);
    font-weight: 700;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06);
  }}

  .rc-item {{
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid var(--border-lt);
  }}
  .rc-item:last-child {{ border-bottom: none; }}
  .rc-dot {{
    width: 10px; height: 10px; border-radius: 50%;
    margin-top: 6px; flex-shrink: 0;
    border: 2px solid var(--border);
    background: var(--card);
  }}
  .rc-dot.closed {{ background: var(--red); border-color: var(--red); }}
  .rc-dot.reopened {{ background: var(--green); border-color: var(--green); }}
  .rc-dot.cashless {{ background: var(--orange); border-color: var(--orange); }}
  .rc-dot.cash_added {{ background: var(--cyan); border-color: var(--cyan); }}
  .rc-dot.format {{ background: var(--purple); border-color: var(--purple); }}
  .rc-dot.format_change {{ background: var(--purple); border-color: var(--purple); }}
  .rc-dot.change {{ background: var(--yellow); border-color: var(--yellow); }}
  .rc-body {{ flex: 1; }}
  .rc-top {{
    display: flex; gap: 10px; align-items: baseline;
    font-size: 0.8rem; margin-bottom: 4px; flex-wrap: wrap;
  }}
  .rc-date {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; color: var(--dim);
  }}
  .rc-code {{
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent); font-weight: 600; font-size: 0.72rem;
    background: var(--accent-lt); padding: 1px 7px; border-radius: 4px;
  }}
  .rc-bname {{ font-weight: 600; font-size: 0.82rem; }}
  .rc-evlabel {{
    font-size: 0.7rem; color: var(--muted);
    font-style: italic;
  }}
  .rc-change {{ font-size: 0.75rem; color: var(--muted); }}
  .rc-change .fld {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; color: var(--muted); margin-right: 6px;
  }}
  .rc-change .old {{
    color: var(--red); background: var(--red-bg);
    padding: 1px 6px; border-radius: 3px; text-decoration: line-through;
    font-size: 0.7rem;
  }}
  .rc-change .arr {{ color: var(--dim); margin: 0 4px; }}
  .rc-change .new {{
    color: var(--green); background: var(--green-bg);
    padding: 1px 6px; border-radius: 3px; font-weight: 600;
    font-size: 0.7rem;
  }}

  .table-wrap {{
    background: var(--card);
    border-radius: 10px;
    padding: 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    overflow-x: auto;
    margin-top: 6px;
  }}

  .table-wrap h2 {{ font-size: 0.95rem; margin-bottom: 10px; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
  th, td {{ padding: 7px 10px; text-align: right; border-bottom: 1px solid var(--border-lt); }}
  th {{ background: #f8f9fc; color: var(--muted); font-weight: 600; position: sticky; top: 0; }}
  th:first-child, td:first-child {{ text-align: left; }}
  tr:hover td {{ background: #f5f7fb; }}
  tr.is-current td {{ background: #fffbeb; font-weight: 600; }}
  tr.is-current:hover td {{ background: #fef3c7; }}

  #tabDetail {{ display: none; }}

  #tabDetail.active {{
    display: grid;
    grid-template-columns: 310px 1fr;
    min-height: calc(100vh - 49px);
  }}

  .side {{
    background: var(--card);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    height: calc(100vh - 49px);
    position: sticky;
    top: 49px;
  }}

  .side-hdr {{ padding: 18px 16px 12px; border-bottom: 1px solid var(--border); }}
  .side-hdr h2 {{ font-size: 0.95rem; font-weight: 700; color: var(--accent); margin-bottom: 2px; }}
  .side-hdr p {{ font-size: 0.72rem; color: var(--muted); }}

  .search-box {{ padding: 10px 16px; border-bottom: 1px solid var(--border); }}
  .search-box input {{
    width: 100%; padding: 8px 12px;
    background: var(--bg); border: 1px solid var(--border); border-radius: 7px;
    color: var(--text); font-family: 'DM Sans', sans-serif; font-size: 0.82rem;
    outline: none; transition: border-color 0.2s;
  }}
  .search-box input::placeholder {{ color: var(--dim); }}
  .search-box input:focus {{ border-color: var(--accent); }}

  .flt-bar {{ padding: 8px 16px; border-bottom: 1px solid var(--border); display: flex; gap: 5px; flex-wrap: wrap; }}
  .flt-btn {{
    padding: 3px 10px; border-radius: 14px; border: 1px solid var(--border);
    background: transparent; color: var(--muted); font-family: 'DM Sans', sans-serif;
    font-size: 0.69rem; font-weight: 500; cursor: pointer; transition: all 0.15s;
  }}
  .flt-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
  .flt-btn.active {{ background: var(--accent-lt); border-color: var(--accent); color: var(--accent); font-weight: 600; }}

  .list-cnt {{ padding: 5px 16px; font-size: 0.69rem; color: var(--dim); border-bottom: 1px solid var(--border-lt); background: var(--bg); }}
  .b-list {{ flex: 1; overflow-y: auto; }}
  .b-list::-webkit-scrollbar {{ width: 4px; }}
  .b-list::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}

  .b-item {{
    padding: 9px 16px; cursor: pointer; transition: all 0.1s;
    border-left: 3px solid transparent; border-bottom: 1px solid var(--border-lt);
  }}
  .b-item:hover {{ background: var(--bg); }}
  .b-item.active {{ background: var(--accent-lt); border-left-color: var(--accent); }}
  .b-item .code {{ font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--accent); font-weight: 600; }}
  .b-item .name {{ font-size: 0.8rem; margin-top: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .b-item .meta {{ font-size: 0.67rem; color: var(--dim); margin-top: 2px; display: flex; gap: 7px; align-items: center; }}
  .b-item .badge {{ background: var(--bg); padding: 1px 6px; border-radius: 8px; font-weight: 600; font-size: 0.65rem; border: 1px solid var(--border); }}
  .sdot {{ width: 6px; height: 6px; border-radius: 50%; display: inline-block; }}
  .sdot.open {{ background: var(--green); }}
  .sdot.closed {{ background: var(--red); }}

  .detail {{ padding: 28px 32px; overflow-y: auto; height: calc(100vh - 49px); }}
  .empty-st {{ display: flex; flex-direction: column; align-items: center; justify-content: center; height: 50vh; color: var(--dim); }}
  .empty-st .arr {{ font-size: 2rem; margin-bottom: 10px; opacity: 0.3; }}

  .br-hdr {{ margin-bottom: 24px; padding-bottom: 18px; border-bottom: 1px solid var(--border); }}
  .br-hdr .top {{ display: flex; align-items: baseline; gap: 12px; margin-bottom: 6px; }}
  .br-hdr .bcode {{ font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: var(--accent); background: var(--accent-lt); padding: 3px 10px; border-radius: 5px; font-weight: 600; }}
  .br-hdr h2 {{ font-size: 1.3rem; font-weight: 700; letter-spacing: -0.02em; }}
  .chips {{ display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; }}
  .chip {{ padding: 3px 11px; border-radius: 14px; font-size: 0.7rem; font-weight: 600; }}
  .chip.open {{ background: var(--green-bg); color: var(--green); }}
  .chip.closed {{ background: var(--red-bg); color: var(--red); }}
  .chip.cashless {{ background: var(--orange-bg); color: var(--orange); }}
  .chip.cash {{ background: var(--cyan-bg); color: var(--cyan); }}
  .chip.nfmt {{ background: var(--purple-bg); color: var(--purple); }}
  .chip.ofmt {{ background: var(--yellow-bg); color: var(--yellow); }}
  .sum-txt {{ margin-top: 6px; font-size: 0.78rem; color: var(--muted); }}

  .tl {{ position: relative; padding-left: 30px; }}
  .tl::before {{
    content: ''; position: absolute; left: 11px; top: 5px; bottom: 5px; width: 2px;
    background: linear-gradient(to bottom, var(--accent), var(--border) 20%, var(--border) 85%, transparent);
    border-radius: 2px;
  }}
  .tl-ev {{ position: relative; margin-bottom: 18px; animation: fadeUp 0.22s ease forwards; opacity: 0; }}
  @keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(5px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}
  .tl-ev .dot {{ position: absolute; left: -24px; top: 13px; width: 9px; height: 9px; border-radius: 50%; border: 2px solid var(--border); background: var(--card); z-index: 1; }}
  .tl-ev.initial .dot {{ background: var(--accent); border-color: var(--accent); }}
  .tl-ev.closed .dot {{ background: var(--red); border-color: var(--red); }}
  .tl-ev.reopened .dot {{ background: var(--green); border-color: var(--green); }}
  .tl-ev.cashless .dot {{ background: var(--orange); border-color: var(--orange); }}
  .tl-ev.cash_added .dot {{ background: var(--cyan); border-color: var(--cyan); }}
  .tl-ev.format_change .dot {{ background: var(--purple); border-color: var(--purple); }}
  .tl-ev.change .dot {{ background: var(--yellow); border-color: var(--yellow); }}

  .ev-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; transition: box-shadow 0.2s; }}
  .ev-card:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.05); }}
  .ev-date {{ font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--dim); margin-bottom: 3px; }}
  .ev-lbl {{ font-size: 0.88rem; font-weight: 700; margin-bottom: 6px; }}
  .tl-ev.initial .ev-lbl {{ color: var(--accent); }}
  .tl-ev.closed .ev-lbl {{ color: var(--red); }}
  .tl-ev.reopened .ev-lbl {{ color: var(--green); }}
  .tl-ev.cashless .ev-lbl {{ color: var(--orange); }}
  .tl-ev.cash_added .ev-lbl {{ color: var(--cyan); }}
  .tl-ev.format_change .ev-lbl {{ color: var(--purple); }}
  .tl-ev.change .ev-lbl {{ color: var(--yellow); }}

  .ch-row {{ display: flex; align-items: center; gap: 7px; padding: 4px 0; font-size: 0.76rem; border-bottom: 1px solid var(--border-lt); }}
  .ch-row:last-child {{ border-bottom: none; }}
  .ch-f {{ font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--muted); min-width: 100px; font-weight: 600; }}
  .ch-old {{ color: var(--red); background: var(--red-bg); padding: 1px 6px; border-radius: 3px; font-size: 0.7rem; text-decoration: line-through; }}
  .ch-arr {{ color: var(--dim); font-size: 0.7rem; }}
  .ch-new {{ color: var(--green); background: var(--green-bg); padding: 1px 6px; border-radius: 3px; font-size: 0.7rem; font-weight: 600; }}
  .st-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 3px 14px; margin-top: 4px; }}
  .st-item {{ display: flex; justify-content: space-between; padding: 2px 0; font-size: 0.72rem; }}
  .st-k {{ color: var(--dim); font-family: 'JetBrains Mono', monospace; font-size: 0.66rem; }}
  .st-v {{ font-weight: 500; }}

  #tabFormats {{ padding: 28px 32px; max-width: 1200px; margin: 0 auto; }}

  .fmt-branch-list {{ display: flex; flex-wrap: wrap; gap: 5px; margin-top: 4px; }}
  .fmt-branch-pill {{
    display: inline-flex; align-items: center; gap: 5px;
    background: var(--purple-bg); border: 1px solid #ddd6fe;
    border-radius: 14px; padding: 3px 10px; font-size: 0.7rem; color: var(--purple);
  }}
  .fmt-branch-pill .pcode {{ font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.68rem; }}
  .fmt-branch-pill .pfmt {{ background: var(--purple); color: #fff; border-radius: 8px; padding: 1px 6px; font-size: 0.63rem; font-weight: 600; }}
  .fmt-row-toggle {{ cursor: pointer; user-select: none; }}
  .fmt-row-toggle:hover td {{ background: #f0ebff !important; }}
  .fmt-detail-row td {{ padding: 0 !important; border-bottom: 1px solid var(--border-lt); }}
  .fmt-detail-inner {{ padding: 10px 12px 14px 28px; background: #faf8ff; }}
  .expand-icon {{ display: inline-block; transition: transform 0.18s; font-size: 0.65rem; margin-left: 6px; color: var(--dim); }}
  .expand-icon.open {{ transform: rotate(90deg); }}

  .foot {{ text-align: center; padding: 16px; font-size: 0.7rem; color: var(--dim); border-top: 1px solid var(--border-lt); margin-top: 20px; }}

  @media (max-width: 860px) {{
    #tabDetail.active {{ grid-template-columns: 1fr; }}
    .side {{ position: relative; height: auto; max-height: 38vh; }}
    .detail {{ height: auto; }}
    #tabOverview, #tabFormats {{ padding: 20px 16px; }}
  }}
</style>
</head>
<body>

<div class="tab-nav">
  <div class="brand">Pobočková síť ČS</div>
  <button class="tab-btn active" data-tab="tabOverview">Přehled sítě</button>
  <button class="tab-btn" data-tab="tabDetail">Detail pobočky</button>
  <button class="tab-btn" data-tab="tabFormats">Nové formáty</button>
</div>

<div id="tabOverview" class="tab-content active">
  <div class="ov-header">
    <h1>Měsíční snapshoty pobočkové sítě</h1>
    <p>Stav sítě v čase — pouze pobočky s bns_flag = "Y"</p>
    <div class="range" id="rangeLabel"></div>
  </div>
  <div class="kpi-row" id="kpiRow"></div>
  <div class="chart-card">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;flex-wrap:wrap;gap:10px;">
      <div><h2 style="margin-bottom:2px;">Poslední změny v síti</h2><div class="sub" style="margin-bottom:0;">Klíčové události napříč všemi pobočkami</div></div>
      <div class="rc-tabs" id="rcTabs">
        <button class="rc-tab active" data-cat="all">Vše</button>
        <button class="rc-tab" data-cat="cashless">Cashless</button>
        <button class="rc-tab" data-cat="format">Nový formát</button>
        <button class="rc-tab" data-cat="closed">Uzavření</button>
      </div>
    </div>
    <div id="recentChangesList"></div>
  </div>
  <div class="chart-card">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;flex-wrap:wrap;gap:10px;">
      <div><h2 style="margin-bottom:2px;">Vývoj pobočkové sítě</h2><div class="sub" style="margin-bottom:0;">Starý formát (modrá) + nový formát (zelená) · zrušené pod osou (šedá) · cashless čárkovaná linie</div></div>
      <div class="rc-tabs" id="granTabs">
        <button class="rc-tab" data-gran="M">Měsíce</button>
        <button class="rc-tab active" data-gran="Q">Kvartály</button>
        <button class="rc-tab" data-gran="Y">Roky</button>
      </div>
    </div>
    <div id="chartMain"></div>
  </div>
  <div class="chart-card"><h2>Cashless vs. s hotovostí</h2><div class="sub">Z otevřených poboček</div><div id="chartCashless"></div></div>
  <div class="chart-card"><h2>Nový vs. starý formát</h2><div class="sub">Z otevřených poboček</div><div id="chartFormat"></div></div>
  <div class="chart-card"><h2>Struktura sítě v čase</h2><div class="sub">Stacked columns — rozpad otevřených poboček</div><div id="chartStacked"></div></div>
  <div class="table-wrap">
    <h2>Všechny měsíční snapshoty</h2>
    <table id="historyTable">
      <thead><tr><th>Měsíc</th><th>Celkem</th><th>Otevřené</th><th>Zavřené</th><th>Cashless</th><th>S hotovostí</th><th>Nový formát</th><th>Starý formát</th><th>Δ Otevřené</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>
  <div class="foot">Vygenerováno automaticky — filtr: bns_flag = "Y"</div>
</div>

<div id="tabDetail" class="tab-content">
  <div class="side">
    <div class="side-hdr"><h2>Timeline poboček</h2><p>Historie změn stavů jednotlivých poboček</p></div>
    <div class="search-box"><input type="text" id="searchInput" placeholder="Hledat kód nebo název…"></div>
    <div class="flt-bar">
      <button class="flt-btn active" data-filter="all">Vše</button>
      <button class="flt-btn" data-filter="has-changes">Se změnami</button>
      <button class="flt-btn" data-filter="closed">Zavřené</button>
      <button class="flt-btn" data-filter="cashless">Cashless</button>
    </div>
    <div class="list-cnt" id="listCount"></div>
    <div class="b-list" id="branchList"></div>
  </div>
  <div class="detail" id="detailContent">
    <div class="empty-st"><div class="arr">◀</div><p>Vyberte pobočku ze seznamu vlevo</p></div>
  </div>
</div>

<div id="tabFormats" class="tab-content">
  <div class="ov-header">
    <h1>Adopce nových formátů</h1>
    <p>Pobočky, které přešly z prázdného formátu na nenulový — měsíční pohled</p>
  </div>
  <div class="kpi-row" id="fmtKpiRow"></div>
  <div class="chart-card">
    <h2>Měsíční přírůstek poboček s novým formátem</h2>
    <div class="sub">Počet poboček, které v daném měsíci poprvé dostaly formát (sloupce) · kumulativní součet (linie)</div>
    <div id="chartFmtAdoption"></div>
  </div>
  <div class="chart-card">
    <h2>Meziměsíční rozdíl (Δ přírůstku)</h2>
    <div class="sub">Zelená = více než minulý měsíc · Červená = méně</div>
    <div id="chartFmtDelta"></div>
  </div>
  <div class="table-wrap">
    <h2>Detail po měsících</h2>
    <table id="fmtTable">
      <thead><tr>
        <th>Měsíc</th>
        <th style="text-align:right;">Přírůstek</th>
        <th style="text-align:right;">Δ od min. měsíce</th>
        <th style="text-align:right;">Kumulativní součet</th>
        <th style="text-align:left;">Pobočky <span style="font-weight:400;color:var(--dim);">(klikněte pro rozbalení)</span></th>
      </tr></thead>
      <tbody id="fmtTableBody"></tbody>
    </table>
  </div>
  <div class="foot">Filtr: bns_flag = "Y" · Pouze přechody formát: prázdný → nenulový</div>
</div>

<script>
let chartsRendered = false;
let fmtChartsRendered = false;

document.querySelectorAll('.tab-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
    if (btn.dataset.tab === 'tabOverview' && !chartsRendered) renderCharts();
    if (btn.dataset.tab === 'tabFormats' && !fmtChartsRendered) renderFmtCharts();
  }});
}});

const historyByGran = {history_all_json};
const history = historyByGran['M'];
const hLast = history[history.length - 1];
const hPrev = history.length > 1 ? history[history.length - 2] : null;
const hFirst = history[0];

document.getElementById('rangeLabel').textContent = hFirst.label + ' → ' + hLast.label + ' (' + history.length + ' záznamů)';

function mkDelta(curr, old) {{
  if (old === null || old === undefined) return '<span class="delta neutral">—</span>';
  const d = curr - old;
  if (d === 0) return '<span class="delta neutral">beze změny</span>';
  return '<span class="delta ' + (d > 0 ? 'up' : 'down') + '">' + (d > 0 ? '+' : '') + d + ' oproti min.</span>';
}}

document.getElementById('kpiRow').innerHTML = [
  {{ label: 'Celkem', value: hLast.total, color: 'var(--accent)', key: 'total' }},
  {{ label: 'Otevřené', value: hLast.opened, color: 'var(--green)', key: 'opened' }},
  {{ label: 'Zavřené', value: hLast.closed, color: 'var(--red)', key: 'closed' }},
  {{ label: 'Cashless', value: hLast.cashless, color: 'var(--orange)', key: 'cashless' }},
  {{ label: 'S hotovostí', value: hLast.non_cashless, color: 'var(--blue)', key: 'non_cashless' }},
  {{ label: 'Nový formát', value: hLast.new_format, color: 'var(--purple)', key: 'new_format' }},
  {{ label: 'Starý formát', value: hLast.old_format, color: 'var(--teal)', key: 'old_format' }},
].map(k => '<div class="kpi" style="border-top-color:' + k.color + '"><div class="value" style="color:' + k.color + '">' + k.value + '</div><div class="label">' + k.label + '</div>' + mkDelta(k.value, hPrev ? hPrev[k.key] : null) + '</div>').join('');

const recentChangesByCat = {recent_changes_json};
let activeRcCat = 'all';
const fldLabelsRC = {{ branch_name:'Název', branch_type:'Typ', branch_closed:'Zavřeno', cashless:'Cashless', format:'Formát', address:'Adresa', city:'Město', region:'Region' }};
function fvRC(v) {{ if(v===null||v===undefined) return '—'; if(v===true) return 'Ano'; if(v===false) return 'Ne'; if(v===''||v==='nan') return '—'; return String(v); }}
function escRC(s) {{ const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }}

function renderRecentChanges() {{
  const shown = (recentChangesByCat[activeRcCat]||[]).slice(0,5);
  const el = document.getElementById('recentChangesList');
  if(!shown.length) {{ el.innerHTML='<div style="color:var(--dim);font-size:0.8rem;padding:12px 0;text-align:center;">Žádné změny.</div>'; return; }}
  el.innerHTML = shown.map(c =>
    '<div class="rc-item"><div class="rc-dot '+c.category+'"></div><div class="rc-body">'+
    '<div class="rc-top"><span class="rc-date">'+c.date+'</span><span class="rc-code">'+c.branch_code+'</span>'+
    '<span class="rc-bname">'+escRC(c.branch_name)+'</span><span class="rc-evlabel">'+escRC(c.label)+'</span></div>'+
    '<div class="rc-change"><span class="fld">'+(fldLabelsRC[c.field]||c.field)+'</span>'+
    '<span class="old">'+escRC(fvRC(c.old))+'</span><span class="arr">→</span><span class="new">'+escRC(fvRC(c.new))+'</span></div>'+
    '</div></div>'
  ).join('');
}}
document.querySelectorAll('#rcTabs .rc-tab').forEach(b => b.addEventListener('click', () => {{ document.querySelectorAll('#rcTabs .rc-tab').forEach(x=>x.classList.remove('active')); b.classList.add('active'); activeRcCat=b.dataset.cat; renderRecentChanges(); }}));
renderRecentChanges();

function renderCharts() {{
  chartsRendered = true;
  function makeChart(el, series, colors, stacked) {{
    new ApexCharts(document.querySelector(el), {{
      chart: {{ type:'bar', height:280, stacked:stacked||false, fontFamily:'DM Sans,sans-serif', toolbar:{{show:true}}, animations:{{enabled:true,easing:'easeinout',speed:400}} }},
      series, colors,
      xaxis: {{ categories: history.map(h=>h.label), labels:{{rotate:-45,style:{{fontSize:'10px'}}}} }},
      yaxis: {{ labels:{{style:{{fontSize:'11px'}}}}, min:0 }},
      plotOptions: {{ bar:{{columnWidth:'70%',borderRadius:2,borderRadiusApplication:'end',borderRadiusWhenStacked:'last'}} }},
      dataLabels: {{ enabled: history.length<=12 }},
      tooltip: {{ shared:true, intersect:false }},
      legend: {{ position:'top', fontSize:'12px' }},
      grid: {{ borderColor:'#e8eaf0', strokeDashArray:3 }},
    }}).render();
  }}

  window.buildMainChartOptions = function(hist) {{
    return {{
      chart: {{ type:'bar', height:380, stacked:true, fontFamily:'DM Sans,sans-serif', toolbar:{{show:true}}, animations:{{enabled:true,easing:'easeinout',speed:400}} }},
      series: [
        {{ name:'Starý formát', type:'bar', data:hist.map(h=>h.old_format) }},
        {{ name:'Nový formát', type:'bar', data:hist.map(h=>h.new_format) }},
        {{ name:'Zrušené', type:'bar', data:hist.map(h=>-h.closed) }},
        {{ name:'Cashless', type:'line', data:hist.map(h=>h.cashless) }},
      ],
      colors: ['#3b82f6','#10b981','#d1d5db','#1e293b'],
      stroke: {{ width:[0,0,0,2.5], dashArray:[0,0,0,6], curve:'smooth' }},
      plotOptions: {{ bar:{{columnWidth:'65%',borderRadius:2,borderRadiusApplication:'end',borderRadiusWhenStacked:'last'}} }},
      xaxis: {{ categories:hist.map(h=>h.label), labels:{{rotate:-45,style:{{fontSize:'10px'}}}}, axisBorder:{{show:true,color:'#94a3b8'}} }},
      yaxis: {{ labels:{{style:{{fontSize:'11px'}},formatter:v=>Math.abs(Math.round(v))}} }},
      dataLabels: {{ enabled:hist.length<=14, formatter:val=>{{const v=Math.abs(Math.round(val)); return v===0?'':(val<0?'-'+v:v);}}, style:{{fontSize:'10px',fontWeight:700,colors:['#fff','#fff','#64748b','#1e293b']}}, background:{{enabled:true,foreColor:'#fff',borderRadius:2,padding:3,opacity:0.85,borderWidth:0}} }},
      tooltip: {{ shared:true, intersect:false, y:{{formatter:val=>Math.abs(Math.round(val))}} }},
      legend: {{ position:'top', fontSize:'12px', markers:{{width:10,height:10,radius:2}} }},
      grid: {{ borderColor:'#e8eaf0', strokeDashArray:3 }},
      annotations: {{
        yaxis: [{{ y:0, borderColor:'#94a3b8', strokeDashArray:0, borderWidth:1 }}],
        points: hist.map(h=>({{ x:h.label, y:h.opened, seriesIndex:1, marker:{{size:0}}, label:{{ text:String(h.opened), borderColor:'transparent', borderWidth:0, borderRadius:3, style:{{background:'transparent',color:'#1e293b',fontSize:'11px',fontWeight:700,padding:{{left:4,right:4,top:2,bottom:2}}}}, offsetY:-8 }} }})),
      }},
    }};
  }};

  window.mainChart = new ApexCharts(document.querySelector('#chartMain'), window.buildMainChartOptions(historyByGran['Q']));
  window.mainChart.render();

  makeChart('#chartCashless', [{{name:'Cashless',data:history.map(h=>h.cashless)}},{{name:'S hotovostí',data:history.map(h=>h.non_cashless)}}], ['#d97706','#2563eb'], true);
  makeChart('#chartFormat', [{{name:'Nový formát',data:history.map(h=>h.new_format)}},{{name:'Starý formát',data:history.map(h=>h.old_format)}}], ['#7c3aed','#0d9488'], true);
  makeChart('#chartStacked', [
    {{name:'Cashless+nový',data:history.map(h=>Math.min(h.cashless,h.new_format))}},
    {{name:'Cashless+starý',data:history.map(h=>Math.max(0,h.cashless-h.new_format))}},
    {{name:'Hotovost+nový',data:history.map(h=>Math.max(0,h.new_format-h.cashless))}},
    {{name:'Hotovost+starý',data:history.map(h=>h.old_format)}},
  ], ['#f59e0b','#ef4444','#8b5cf6','#6b7280'], true);
}}

const tbody = document.querySelector('#historyTable tbody');
history.slice().reverse().forEach((h,i,arr) => {{
  const prevH = i < arr.length-1 ? arr[i+1] : null;
  const dOpen = prevH ? h.opened-prevH.opened : 0;
  const tr = document.createElement('tr');
  if(h.is_current) tr.className='is-current';
  tr.innerHTML = '<td>'+h.label+'</td><td>'+h.total+'</td><td><strong>'+h.opened+'</strong></td><td>'+h.closed+'</td><td>'+h.cashless+'</td><td>'+h.non_cashless+'</td><td>'+h.new_format+'</td><td>'+h.old_format+'</td><td><span class="delta '+(dOpen>0?'up':dOpen<0?'down':'neutral')+'">'+(prevH?(dOpen>0?'+':'')+dOpen:'—')+'</span></td>';
  tbody.appendChild(tr);
}});

renderCharts();

document.querySelectorAll('#granTabs .rc-tab').forEach(b => b.addEventListener('click', () => {{
  document.querySelectorAll('#granTabs .rc-tab').forEach(x=>x.classList.remove('active'));
  b.classList.add('active');
  const hist = historyByGran[b.dataset.gran]||[];
  if(window.mainChart && hist.length) window.mainChart.updateOptions(window.buildMainChartOptions(hist),true,true);
}}));

const branches = {branches_json};
let activeFilter='all', activeCode=null;
const searchInput=document.getElementById('searchInput');
const branchList=document.getElementById('branchList');
const detailContent=document.getElementById('detailContent');
const listCount=document.getElementById('listCount');
const fldLabels={{ branch_name:'Název',branch_type:'Typ',branch_closed:'Zavřeno',cashless:'Cashless',format:'Formát',address:'Adresa',city:'Město',region:'Region' }};
function fv(v){{ if(v===null||v===undefined)return'—'; if(v===true)return'Ano'; if(v===false)return'Ne'; if(v===''||v==='nan')return'—'; return String(v); }}
function esc(s){{ const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }}

function filterBranches() {{
  const q=searchInput.value.toLowerCase().trim();
  return branches.filter(b=>{{
    if(q && !String(b.code).includes(q) && !b.name.toLowerCase().includes(q)) return false;
    const last=b.events[b.events.length-1];
    if(activeFilter==='has-changes' && b.total_changes===0) return false;
    if(activeFilter==='closed' && !last.state.branch_closed) return false;
    if(activeFilter==='cashless' && !last.state.cashless) return false;
    return true;
  }});
}}

function renderList() {{
  const filtered=filterBranches();
  listCount.textContent=filtered.length+' poboček';
  branchList.innerHTML=filtered.map(b=>{{
    const last=b.events[b.events.length-1];
    const cls=activeCode===b.code?' active':'';
    const dc=last.state.branch_closed?'closed':'open';
    return '<div class="b-item'+cls+'" data-code="'+b.code+'"><div class="code">'+b.code+'</div><div class="name">'+esc(b.name)+'</div><div class="meta"><span class="badge">'+b.total_changes+' změn</span><span class="sdot '+dc+'"></span> '+(last.state.branch_closed?'zavřena':'otevřena')+(last.state.cashless?' · cashless':'')+'</div></div>';
  }}).join('');
  branchList.querySelectorAll('.b-item').forEach(el=>el.addEventListener('click',()=>{{activeCode=parseInt(el.dataset.code);renderList();renderTimeline();}}));
}}

function renderTimeline() {{
  const br=branches.find(b=>b.code===activeCode); if(!br) return;
  const ls=br.events[br.events.length-1].state;
  const hf=ls.format&&ls.format!==''&&ls.format!=='nan';
  let h='<div class="br-hdr"><div class="top"><span class="bcode">'+br.code+'</span><h2>'+esc(br.name)+'</h2></div><div class="chips">'+(ls.branch_closed?'<span class="chip closed">● Zavřena</span>':'<span class="chip open">● Otevřena</span>')+(ls.cashless?'<span class="chip cashless">Cashless</span>':'<span class="chip cash">S hotovostí</span>')+(hf?'<span class="chip nfmt">Formát: '+esc(ls.format)+'</span>':'<span class="chip ofmt">Bez formátu</span>')+'</div><div class="sum-txt">'+br.events.length+' záznamů · '+br.total_changes+' změn · '+br.events[0].date+' → '+br.events[br.events.length-1].date+'</div></div>';
  h+='<div class="tl">';
  br.events.forEach((evt,idx)=>{{
    const dl=Math.min(idx*0.04,0.5);
    h+='<div class="tl-ev '+evt.type+'" style="animation-delay:'+dl+'s"><div class="dot"></div><div class="ev-card"><div class="ev-date">'+evt.date+'</div><div class="ev-lbl">'+esc(evt.label)+'</div>';
    if(evt.type==='initial'){{
      h+='<div class="st-grid">';
      for(const [k,v] of Object.entries(evt.state)) h+='<div class="st-item"><span class="st-k">'+(fldLabels[k]||k)+'</span><span class="st-v">'+esc(fv(v))+'</span></div>';
      h+='</div>';
    }} else if(evt.changes.length) {{
      evt.changes.forEach(ch=>{{ h+='<div class="ch-row"><span class="ch-f">'+(fldLabels[ch.field]||ch.field)+'</span><span class="ch-old">'+esc(fv(ch.old))+'</span><span class="ch-arr">→</span><span class="ch-new">'+esc(fv(ch.new))+'</span></div>'; }});
    }}
    h+='</div></div>';
  }});
  h+='</div><div class="foot">filtr: bns_flag = "Y"</div>';
  detailContent.innerHTML=h;
}}

searchInput.addEventListener('input',renderList);
document.querySelectorAll('.flt-btn').forEach(b=>b.addEventListener('click',()=>{{document.querySelectorAll('.flt-btn').forEach(x=>x.classList.remove('active'));b.classList.add('active');activeFilter=b.dataset.filter;renderList();}}));
renderList();

const fmtData = {format_monthly_json};

function renderFmtCharts() {{
  fmtChartsRendered=true;
  if(!fmtData.length) {{ document.getElementById('fmtKpiRow').innerHTML='<div style="color:var(--dim);padding:20px;text-align:center;">Žádné přechody na nový formát.</div>'; return; }}
  const total=fmtData.reduce((s,m)=>s+m.count,0);
  const maxM=fmtData.reduce((a,b)=>b.count>a.count?b:a);
  const avg=(total/fmtData.length).toFixed(1);
  document.getElementById('fmtKpiRow').innerHTML=[
    {{label:'Celkem přechodů',value:total,color:'var(--purple)'}},
    {{label:'Měsíců s přechody',value:fmtData.length,color:'var(--accent)'}},
    {{label:'Ø za měsíc',value:avg,color:'var(--teal)'}},
    {{label:'Nejakt. měsíc',value:maxM.month,sub:maxM.count+' poboček',color:'var(--orange)'}},
  ].map(k=>'<div class="kpi" style="border-top-color:'+k.color+'"><div class="value" style="color:'+k.color+';font-size:'+(String(k.value).length>6?'1.1rem':'1.6rem')+'">'+k.value+'</div><div class="label">'+k.label+'</div>'+(k.sub?'<div class="delta neutral">'+k.sub+'</div>':'')+'</div>').join('');

  const labels=fmtData.map(m=>m.month);
  const counts=fmtData.map(m=>m.count);
  const cumulative=counts.reduce((acc,v)=>{{acc.push((acc.length?acc[acc.length-1]:0)+v);return acc;}},[]);
  const deltas=counts.map((v,i)=>i===0?0:v-counts[i-1]);

  new ApexCharts(document.querySelector('#chartFmtAdoption'),{{
    chart:{{type:'bar',height:300,fontFamily:'DM Sans,sans-serif',toolbar:{{show:true}},animations:{{enabled:true,easing:'easeinout',speed:400}}}},
    series:[{{name:'Přírůstek',type:'bar',data:counts}},{{name:'Kumulativní',type:'line',data:cumulative}}],
    colors:['#7c3aed','#0891b2'],
    stroke:{{width:[0,2.5],curve:'smooth'}},
    plotOptions:{{bar:{{columnWidth:'60%',borderRadius:3}}}},
    xaxis:{{categories:labels,labels:{{rotate:-45,style:{{fontSize:'10px'}}}}}},
    yaxis:[{{title:{{text:'Přírůstek',style:{{fontSize:'11px'}}}},min:0}},{{opposite:true,title:{{text:'Kumulativně',style:{{fontSize:'11px'}}}},min:0}}],
    dataLabels:{{enabled:fmtData.length<=18,enabledOnSeries:[0],style:{{fontSize:'10px',fontWeight:700}},background:{{enabled:true,borderRadius:2,padding:2,opacity:0.85,borderWidth:0}}}},
    tooltip:{{shared:true,intersect:false}},
    legend:{{position:'top',fontSize:'12px'}},
    grid:{{borderColor:'#e8eaf0',strokeDashArray:3}},
  }}).render();

  new ApexCharts(document.querySelector('#chartFmtDelta'),{{
    chart:{{type:'bar',height:220,fontFamily:'DM Sans,sans-serif',toolbar:{{show:false}},animations:{{enabled:true,easing:'easeinout',speed:400}}}},
    series:[{{name:'Δ přírůstek',data:deltas}}],
    plotOptions:{{bar:{{columnWidth:'60%',borderRadius:2,colors:{{ranges:[{{from:-9999,to:-1,color:'#dc2626'}},{{from:0,to:0,color:'#94a3b8'}},{{from:1,to:9999,color:'#059669'}}]}}}}}},
    xaxis:{{categories:labels,labels:{{rotate:-45,style:{{fontSize:'10px'}}}}}},
    yaxis:{{labels:{{style:{{fontSize:'11px'}},formatter:v=>(v>0?'+':'')+v}}}},
    dataLabels:{{enabled:fmtData.length<=18,formatter:v=>(v>0?'+':'')+v,style:{{fontSize:'10px',fontWeight:700}}}},
    annotations:{{yaxis:[{{y:0,borderColor:'#94a3b8',strokeDashArray:0,borderWidth:1}}]}},
    tooltip:{{y:{{formatter:v=>(v>0?'+':'')+v+' poboček'}}}},
    grid:{{borderColor:'#e8eaf0',strokeDashArray:3}},
  }}).render();

  const fmtTbody=document.getElementById('fmtTableBody');
  fmtData.slice().reverse().forEach((m,i)=>{{
    const origIdx=fmtData.length-1-i;
    const delta=origIdx===0?null:m.count-fmtData[origIdx-1].count;
    const cumul=cumulative[origIdx];
    const dText=delta===null?'—':(delta>0?'+'+delta:String(delta));
    const dClass=delta===null?'neutral':delta>0?'up':delta<0?'down':'neutral';
    const pillsHtml=m.branches.map(b=>'<span class="fmt-branch-pill"><span class="pcode">'+b.branch_code+'</span>'+esc(b.branch_name)+'<span class="pfmt">'+esc(b.format_new)+'</span></span>').join('');
    const tr=document.createElement('tr');
    tr.className='fmt-row-toggle';
    tr.innerHTML='<td style="font-family:monospace;font-size:0.8rem;">'+m.month+'</td><td style="font-weight:700;color:var(--purple);">'+m.count+'</td><td><span class="delta '+dClass+'">'+dText+'</span></td><td>'+cumul+'</td><td style="text-align:left;color:var(--muted);font-size:0.75rem;">'+m.branches.slice(0,3).map(b=>'<span style="font-family:monospace;font-size:0.7rem;color:var(--accent);">'+b.branch_code+'</span>').join(' ')+(m.count>3?' <span style="color:var(--dim);">+'+( m.count-3)+' dalších</span>':'')+' <span class="expand-icon" id="icon-'+i+'">▶</span></td>';
    const trD=document.createElement('tr');
    trD.className='fmt-detail-row'; trD.style.display='none';
    trD.innerHTML='<td colspan="5"><div class="fmt-detail-inner"><div class="fmt-branch-list">'+pillsHtml+'</div></div></td>';
    tr.addEventListener('click',()=>{{
      const open=trD.style.display!=='none';
      trD.style.display=open?'none':'table-row';
      const ic=document.getElementById('icon-'+i);
      if(ic) ic.classList.toggle('open',!open);
    }});
    fmtTbody.appendChild(tr); fmtTbody.appendChild(trD);
  }});
}}

if(document.getElementById('tabFormats').classList.contains('active')) renderFmtCharts();
</script>
</body>
</html>"""

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write(html)

print(f"📄 Kombinovaný report uložen: {REPORT_FILE}")
