import pandas as pd
import json
import os
import numpy as np
from datetime import datetime
from collections import defaultdict
import urllib.request

df = pd.read_csv('in/tables/dbs_branch_network_epb.csv')

df['branch_type'] = df['branch_type'].astype(str).str.strip()
df['branch_code_clean'] = df['branch_code'].astype(int)

branch_types_y = ['BP', 'BZ', 'PP', 'UP', 'ZP']
special_codes = [902, 904, 905, 906, 909, 919]

cond1 = (df['branch_code_clean'] >= 1) & (df['branch_code_clean'] <= 675) & (df['branch_type'].isin(branch_types_y))
cond2 = (df['branch_code_clean'].isin(special_codes)) & (df['branch_type'] == 'PR')
df['bns_flag'] = np.where(cond1 | cond2, 'Y', 'N')

df["effective_date"] = pd.to_datetime(df["effective_date"])
df_bns = df[df["bns_flag"] == "Y"].copy()
df_bns = df_bns.sort_values(["branch_code", "effective_date"])

has_nf_cols = ("branch_building_nf_sf" in df_bns.columns and "nf_number" in df_bns.columns)

df_bns["year_month"] = df_bns["effective_date"].dt.to_period("M")
all_months = sorted(df_bns["year_month"].unique())

def is_new_format_mask(df_sub):
    if not has_nf_cols:
        return df_sub["format"].notna() & (df_sub["format"].astype(str).str.strip() != "")
    nf_sf_ok = df_sub["branch_building_nf_sf"].astype(str).str.strip() == "NF"
    fmt_ok = (
        df_sub["format"].notna() &
        (df_sub["format"].astype(str).str.strip() != "") &
        (df_sub["format"].astype(str).str.strip() != "nan")
    )
    nf_num_ok = pd.to_numeric(df_sub["nf_number"], errors="coerce").notna()
    return nf_sf_ok & fmt_ok & nf_num_ok

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
    new_format = int(is_new_format_mask(open_branches).sum())
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

print(f"\U0001f4ca Zpracováno {len(history)} snapshotů (včetně aktuálního)")

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

print(f"\U0001f4ca Agregace: M={len(history_monthly)}, Q={len(history_quarterly)}, Y={len(history_yearly)}")

tracked_cols = [
    "branch_name", "branch_type", "branch_closed", "cashless",
    "format", "branch_building_nf_sf", "nf_number", "address", "city", "region"
]
tracked_cols = [c for c in tracked_cols if c in df_bns.columns]

branches_data = {}
for code, grp in df_bns.groupby("branch_code"):
    grp = grp.sort_values("effective_date").reset_index(drop=True)
    branch_name = str(grp["branch_name"].iloc[-1]) if "branch_name" in grp.columns else f"Pobčka {code}"
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
                        if ch["new"] == True: evt_type = "closed"; label = "Pobčka zavřena"
                        elif ch["new"] == False: evt_type = "reopened"; label = "Pobčka znovu otevřena"
                    elif ch["field"] == "cashless":
                        if ch["new"] == True: evt_type = "cashless"; label = "Přechod na cashless"
                        elif ch["new"] == False: evt_type = "cash_added"; label = "Vrácena hotovost"
                    elif ch["field"] == "format" and ch["new"] and ch["new"] != "":
                        evt_type = "format_change"; label = "Nový formát: " + str(ch["new"])
                events.append({"date": date_str, "type": evt_type, "label": label, "state": state, "changes": changes})
        prev_row = row
    branches_data[int(code)] = {"code": int(code), "name": branch_name, "events": events, "total_changes": len(events) - 1}

sorted_branches = sorted(branches_data.values(), key=lambda b: b["code"])
print(f"✅ Připraveno {len(sorted_branches)} pobček pro timeline")

def categorize_change(field, old_val, new_val):
    if field == "cashless":
        if new_val is True: return ("cashless", "Přechod na cashless")
        elif new_val is False: return ("cashless", "Vrácena hotovost")
    elif field == "format":
        if new_val and str(new_val) not in ("", "nan", "None"):
            return ("format", f"Nový formát: {new_val}")
    elif field == "branch_closed":
        if new_val is True: return ("closed", "Pobčka zavřena")
        elif new_val is False: return ("closed", "Pobčka znovu otevřena")
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

print(f"\U0001f514 Změny připraveny: all={len(recent_changes_by_cat['all'])}, cashless={len(recent_changes_by_cat['cashless'])}, format={len(recent_changes_by_cat['format'])}, closed={len(recent_changes_by_cat['closed'])}")

# Kraj mapping
_KRAJ_MAP = {
    "Praha": ["Praha","Praha 1","Praha 2","Praha 3","Praha 4","Praha 5","Praha 6","Praha 7","Praha 8","Praha 9","Praha 10","Praha 11","Praha 12","Praha 13","Praha 14","Praha 15","Praha 16","Praha 17","Praha 18","Praha 19","Praha 20","Praha 21"],
    "Středočeský": ["Benešov","Beroun","Brandýs nad Labem","Kladno","Kolín","Kutná Hora","Mělník","Mladá Boleslav","Nymburk","Příbram","Rakovník","Říčany","Slaný","Čáslav","Dobříš","Hořovice","Kralupy nad Vltavou","Lysá nad Labem","Mnichovo Hradiště","Neratovice","Poděbrady","Vlašim"],
    "Jihočeský": ["České Budějovice","Písek","Tábor","Strakonice","Jindřichův Hradec","Český Krumlov","Prachatice","Blatná","Dačice","Milevsko","Písek","Soběslav","Třeboň","Vimperk","Vodňany"],
    "Plzeňský": ["Plzeň","Klatovy","Rokycany","Domažlice","Tachov","Blovice","Horažďovice","Horšovský Týn","Nepomuk","Přeštice","Stříbro","Sušice"],
    "Karlovarský": ["Karlovy Vary","Sokolov","Cheb","Mariánské Lázně","Aš","Františkovy Lázně","Kraslice","Ostrov","Jáchymov"],
    "Ústecký": ["Ústí nad Labem","Most","Teplice","Chomutov","Děčín","Litoměřice","Louny","Litvínov","Kadaň","Bílina","Duchcov","Jirkov","Klášterec nad Ohří","Lovosice","Roudnice nad Labem","Rumburk","Šluknov","Varnsdorf","Žatec"],
    "Liberecký": ["Liberec","Jablonec nad Nisou","Česká Lípa","Semily","Turnov","Nový Bor","Frýdlant","Jilemnice","Tanvald","Železný Brod"],
    "Královéhradecký": ["Hradec Králové","Jičín","Náchod","Trutnov","Rychnov nad Kněžnou","Dvůr Králové nad Labem","Broumov","Dobruška","Hořice","Jaroměř","Kostelec nad Orlicí","Nová Paka","Nové Město nad Metují","Nový Bydžov","Opočno"],
    "Pardubický": ["Pardubice","Chrudim","Svitavy","Ústí nad Orlicí","Litomyšl","Vysoké Mýto","Česká Třebová","Hlinsko","Holice","Lanškroun","Polička","Přelouč","Skuteč"],
    "Vysočina": ["Jihlava","Havlíčkův Brod","Žďár nad Sázavou","Třebíč","Pelhřimov","Velké Meziříčí","Bystřice nad Pernštejnem","Humpolec","Moravské Budějovice","Náměšť nad Oslavou","Nové Město na Moravě","Pacov","Telč","Tišnov","Velká Bíteš"],
    "Jihomoravský": ["Brno","Hodonín","Znojmo","Břeclav","Vyškov","Blansko","Boskovice","Kuřim","Kyjov","Mikulov","Pohořelice","Rosice","Slavkov u Brna","Strážnice","Tišnov","Veselí nad Moravou"],
    "Olomoucký": ["Olomouc","Přerov","Prostějov","Šumperk","Jeseník","Litovel","Mohelnice","Šternberk","Uničov","Konice","Lipník nad Bečvou","Zábřeh"],
    "Zlínský": ["Zlín","Uherské Hradiště","Vsetín","Kroměříž","Uherský Brod","Otrokovice","Valašské Meziříčí","Holešov","Luhačovice","Napajedla","Rožnov pod Radhoštěm","Valašské Klobouky","Vizovice","Zlín"],
    "Moravskoslezský": ["Ostrava","Opava","Karviná","Frýdek-Místek","Havířov","Orlová","Nový Jičín","Třinec","Kopřivnice","Krnov","Bohumín","Český Těšín","Bruntál","Bílovec","Frenštát pod Radhoštěm","Hlučín","Jablunkov","Rýmařov","Vítkov"],
}
CITY_TO_KRAJ = {}
for _kn, _cs in _KRAJ_MAP.items():
    for _c in _cs:
        CITY_TO_KRAJ[_c.lower().strip()] = _kn

def get_kraj(city, address=""):
    c = city.strip().lower()
    if c in CITY_TO_KRAJ:
        return CITY_TO_KRAJ[c]
    if c.startswith("praha"):
        return "Praha"
    if c.startswith("brno"):
        return "Jihomoravský"
    if c.startswith("ostrava"):
        return "Moravskoslezský"
    for key, kraj in CITY_TO_KRAJ.items():
        if len(key) >= 5 and (key in c or c in key):
            return kraj
    addr = address.strip().lower()
    for key, kraj in CITY_TO_KRAJ.items():
        if len(key) >= 5 and key in addr:
            return kraj
    return "Neznámý"

# Uzavrene pobocky
closed_branches_data = []
for code, bdata in branches_data.items():
    evts = bdata["events"]
    if not evts:
        continue
    last_state = evts[-1]["state"]
    if not last_state.get("branch_closed"):
        continue
    close_date = evts[0]["date"]
    for evt in evts:
        for ch in evt.get("changes", []):
            if ch["field"] == "branch_closed" and ch["new"] is True:
                close_date = evt["date"]
                break
    city_val = str(last_state.get("city") or "")
    addr_val = str(last_state.get("address") or "")
    closed_branches_data.append({
        "branch_code": int(code),
        "branch_name": bdata["name"],
        "close_date": close_date,
        "city": city_val,
        "region": str(last_state.get("region") or ""),
        "branch_type": str(last_state.get("branch_type") or ""),
        "address": addr_val,
        "kraj": get_kraj(city_val, addr_val),
    })
closed_branches_data.sort(key=lambda x: x["close_date"], reverse=True)

# Vsechny zmeny po mesicich pro kalendar
changes_by_month_cal = defaultdict(list)
for ch in all_changes:
    changes_by_month_cal[ch["date"][:7]].append(ch)

print(f"\U0001f534 Uzavřených pobček: {len(closed_branches_data)}")
print(f"\U0001f4c5 Měsíce s kalendářními událostmi: {len(changes_by_month_cal)}")

def is_new_format_state(state):
    nf_sf  = state.get("branch_building_nf_sf")
    fmt    = state.get("format")
    nf_num = state.get("nf_number")
    if not has_nf_cols:
        return fmt is not None and str(fmt).strip() not in ("", "nan", "None")
    nf_sf_ok = nf_sf is not None and str(nf_sf).strip() == "NF"
    fmt_ok   = fmt is not None and str(fmt).strip() not in ("", "nan", "None")
    nf_num_ok = False
    if nf_num is not None:
        try:
            float(str(nf_num).strip())
            nf_num_ok = True
        except (ValueError, TypeError):
            pass
    return nf_sf_ok and fmt_ok and nf_num_ok

format_transitions = []
for code, bdata in branches_data.items():
    events = bdata["events"]
    for i, evt in enumerate(events):
        if i == 0:
            continue
        prev_nf = is_new_format_state(events[i - 1]["state"])
        curr_nf = is_new_format_state(evt["state"])
        if not prev_nf and curr_nf:
            curr_state = evt["state"]
            format_transitions.append({
                "date": evt["date"],
                "month": evt["date"][:7],
                "branch_code": int(code),
                "branch_name": bdata["name"],
                "format_new": str(curr_state.get("format", "")),
                "nf_number": str(curr_state.get("nf_number", "")),
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

print(f"\U0001f3a8 Přechody na nový formát: {len(format_transitions)} celkem, {len(format_monthly_summary)} měsíců")

_apex_url = "https://cdn.jsdelivr.net/npm/apexcharts/dist/apexcharts.min.js"
try:
    with urllib.request.urlopen(_apex_url, timeout=15) as _r:
        _apex_js = _r.read().decode("utf-8")
    print(f"\U0001f4e6 ApexCharts stažen ({len(_apex_js)//1024} KB) — bude embedded inline")
except Exception as _e:
    _apex_js = None
    print(f"⚠️  ApexCharts se nepodařilo stáhnout ({_e}) — použije se CDN odkaz")

history_json = json.dumps(history, ensure_ascii=False)
history_all_json = json.dumps({"M": history_monthly, "Q": history_quarterly, "Y": history_yearly}, ensure_ascii=False)
branches_json = json.dumps(sorted_branches, ensure_ascii=False)
recent_changes_json = json.dumps(recent_changes_by_cat, ensure_ascii=False)
format_monthly_json = json.dumps(format_monthly_summary, ensure_ascii=False)
def _build_closed_ts(items, key_fn):
    by_yk = defaultdict(lambda: defaultdict(int))
    for item in items:
        by_yk[item["close_date"][:4]][key_fn(item)] += 1
    years = sorted(by_yk.keys())
    keys = sorted({k for yr_data in by_yk.values() for k in yr_data})
    series = [{"name": k, "data": [by_yk[yr].get(k, 0) for yr in years]}
              for k in keys if any(by_yk[yr].get(k, 0) > 0 for yr in years)]
    return {"years": years, "series": series}

closed_chart_data = {
    "region": _build_closed_ts(closed_branches_data, lambda b: b.get("region") or "Neznámý"),
    "kraj": _build_closed_ts(closed_branches_data, lambda b: b.get("kraj", "Neznámý")),
}

closed_branches_json = json.dumps(closed_branches_data, ensure_ascii=False)
closed_chart_data_json = json.dumps(closed_chart_data, ensure_ascii=False)
changes_by_month_json = json.dumps({k: v for k, v in changes_by_month_cal.items()}, ensure_ascii=False)

# Cashless transitions with running %
def get_branch_state_at(bdata, date_str):
    state = None
    for evt in bdata["events"]:
        if evt["date"] <= date_str:
            state = evt["state"]
        else:
            break
    return state

cashless_transitions = []
for _code, _bdata in branches_data.items():
    for _evt in _bdata["events"]:
        for _ch in _evt.get("changes", []):
            if _ch["field"] == "cashless" and _ch["new"] is True:
                cashless_transitions.append({
                    "date": _evt["date"],
                    "branch_code": int(_code),
                    "branch_name": _bdata["name"],
                    "city": str(_evt["state"].get("city") or ""),
                    "region": str(_evt["state"].get("region") or ""),
                })
cashless_transitions.sort(key=lambda x: x["date"])

print(f"💳 Cashless přechodů: {len(cashless_transitions)} — výpočet % probíhá...")
for _ct in cashless_transitions:
    _open = 0; _cash = 0
    for _c2, _b2 in branches_data.items():
        _st = get_branch_state_at(_b2, _ct["date"])
        if _st is None: continue
        if not _st.get("branch_closed", False):
            _open += 1
            if _st.get("cashless") is True: _cash += 1
    _ct["open_count"] = _open
    _ct["cashless_count"] = _cash
    _ct["pct"] = round(100 * _cash / _open, 1) if _open > 0 else 0

# Timeline events 2024-2026
timeline_data = {}
for _ch in all_changes:
    _yr = _ch["date"][:4]
    if _yr not in ("2024", "2025", "2026"): continue
    if _ch["category"] not in ("cashless", "closed", "format"): continue
    _mk = _ch["date"][:7]
    if _yr not in timeline_data: timeline_data[_yr] = {}
    if _mk not in timeline_data[_yr]: timeline_data[_yr][_mk] = []
    timeline_data[_yr][_mk].append(_ch)

cashless_transitions_json = json.dumps(cashless_transitions, ensure_ascii=False)
timeline_data_json = json.dumps(timeline_data, ensure_ascii=False)
print(f"📅 Timeline: {sum(len(v) for v in timeline_data.values())} měsíců s událostmi")

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
<title>Pobčková síť ČS — Report</title>
{apex_script_tag}
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,700&family=JetBrains+Mono:wght@400;600&display=swap');
  :root {{
    --bg:#f3f5f9; --card:#fff; --text:#1e2330; --muted:#64748b; --dim:#9ca3b0;
    --border:#dfe2ea; --border-lt:#eceef4; --accent:#0057b8; --accent-lt:#e8f0fe;
    --green:#059669; --green-bg:#ecfdf5; --red:#dc2626; --red-bg:#fef2f2;
    --orange:#d97706; --orange-bg:#fffbeb; --blue:#2563eb; --purple:#7c3aed;
    --purple-bg:#f5f3ff; --teal:#0d9488; --yellow:#b45309; --yellow-bg:#fefce8;
    --cyan:#0891b2; --cyan-bg:#ecfeff;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'DM Sans',sans-serif; background:var(--bg); color:var(--text); min-height:100vh; }}

  .tab-nav {{ display:flex; align-items:center; background:var(--card); border-bottom:1px solid var(--border); padding:0 32px; position:sticky; top:0; z-index:100; }}
  .tab-nav .brand {{ font-weight:700; font-size:0.95rem; color:var(--accent); padding:14px 24px 14px 0; border-right:1px solid var(--border); margin-right:4px; white-space:nowrap; }}
  .tab-btn {{ padding:14px 22px; background:none; border:none; border-bottom:2px solid transparent; font-family:'DM Sans',sans-serif; font-size:0.85rem; font-weight:500; color:var(--muted); cursor:pointer; transition:all 0.2s; }}
  .tab-btn:hover {{ color:var(--text); }}
  .tab-btn.active {{ color:var(--accent); border-bottom-color:var(--accent); font-weight:700; }}
  .tab-content {{ display:none; }}
  .tab-content.active {{ display:block; }}

  #tabOverview {{ padding:28px 32px; max-width:1200px; margin:0 auto; }}
  .ov-header {{ text-align:center; margin-bottom:28px; }}
  .ov-header h1 {{ font-size:1.4rem; color:var(--accent); margin-bottom:4px; }}
  .ov-header p {{ color:var(--muted); font-size:0.85rem; }}
  .ov-header .range {{ display:inline-block; margin-top:8px; background:var(--accent-lt); color:var(--accent); padding:4px 14px; border-radius:20px; font-size:0.78rem; font-weight:600; }}

  .kpi-row {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:10px; margin-bottom:24px; }}
  .kpi {{ background:var(--card); border-radius:10px; padding:14px; text-align:center; box-shadow:0 1px 3px rgba(0,0,0,.05); border-top:3px solid transparent; }}
  .kpi .value {{ font-size:1.6rem; font-weight:700; line-height:1.2; }}
  .kpi .label {{ font-size:0.75rem; color:var(--muted); margin-top:3px; }}
  .kpi .delta {{ font-size:0.72rem; margin-top:2px; }}
  .delta.up {{ color:var(--green); }}
  .delta.down {{ color:var(--red); }}
  .delta.neutral {{ color:var(--muted); }}

  .chart-card {{ background:var(--card); border-radius:10px; padding:18px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,.05); }}
  .chart-card h2 {{ font-size:0.95rem; margin-bottom:2px; }}
  .chart-card .sub {{ font-size:0.78rem; color:var(--muted); margin-bottom:10px; }}

  .rc-tabs {{ display:flex; gap:4px; background:var(--bg); padding:3px; border-radius:8px; border:1px solid var(--border); }}
  .rc-tab {{ padding:5px 12px; background:transparent; border:none; border-radius:5px; font-family:'DM Sans',sans-serif; font-size:0.72rem; font-weight:500; color:var(--muted); cursor:pointer; transition:all .15s; }}
  .rc-tab:hover {{ color:var(--text); }}
  .rc-tab.active {{ background:var(--card); color:var(--accent); font-weight:700; box-shadow:0 1px 2px rgba(0,0,0,.06); }}

  .rc-item {{ display:flex; align-items:flex-start; gap:12px; padding:10px 0; border-bottom:1px solid var(--border-lt); }}
  .rc-item:last-child {{ border-bottom:none; }}
  .rc-dot {{ width:10px; height:10px; border-radius:50%; margin-top:6px; flex-shrink:0; border:2px solid var(--border); background:var(--card); }}
  .rc-dot.closed {{ background:var(--red); border-color:var(--red); }}
  .rc-dot.reopened {{ background:var(--green); border-color:var(--green); }}
  .rc-dot.cashless {{ background:var(--orange); border-color:var(--orange); }}
  .rc-dot.cash_added {{ background:var(--cyan); border-color:var(--cyan); }}
  .rc-dot.format,.rc-dot.format_change {{ background:var(--purple); border-color:var(--purple); }}
  .rc-dot.change {{ background:var(--yellow); border-color:var(--yellow); }}
  .rc-body {{ flex:1; }}
  .rc-top {{ display:flex; gap:10px; align-items:baseline; font-size:0.8rem; margin-bottom:4px; flex-wrap:wrap; }}
  .rc-date {{ font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:var(--dim); }}
  .rc-code {{ font-family:'JetBrains Mono',monospace; color:var(--accent); font-weight:600; font-size:0.72rem; background:var(--accent-lt); padding:1px 7px; border-radius:4px; }}
  .rc-bname {{ font-weight:600; font-size:0.82rem; }}
  .rc-evlabel {{ font-size:0.7rem; color:var(--muted); font-style:italic; }}
  .rc-change {{ font-size:0.75rem; color:var(--muted); }}
  .rc-change .fld {{ font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:var(--muted); margin-right:6px; }}
  .rc-change .old {{ color:var(--red); background:var(--red-bg); padding:1px 6px; border-radius:3px; text-decoration:line-through; font-size:0.7rem; }}
  .rc-change .arr {{ color:var(--dim); margin:0 4px; }}
  .rc-change .new {{ color:var(--green); background:var(--green-bg); padding:1px 6px; border-radius:3px; font-weight:600; font-size:0.7rem; }}

  .table-wrap {{ background:var(--card); border-radius:10px; padding:18px; box-shadow:0 1px 3px rgba(0,0,0,.05); overflow-x:auto; margin-top:6px; }}
  .table-wrap h2 {{ font-size:0.95rem; margin-bottom:10px; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.8rem; }}
  th,td {{ padding:7px 10px; text-align:right; border-bottom:1px solid var(--border-lt); }}
  th {{ background:#f8f9fc; color:var(--muted); font-weight:600; position:sticky; top:0; }}
  th:first-child,td:first-child {{ text-align:left; }}
  tr:hover td {{ background:#f5f7fb; }}
  tr.is-current td {{ background:#fffbeb; font-weight:600; }}
  tr.is-current:hover td {{ background:#fef3c7; }}

  #tabDetail {{ display:none; }}
  #tabDetail.active {{ display:grid; grid-template-columns:310px 1fr; min-height:calc(100vh - 49px); }}
  .side {{ background:var(--card); border-right:1px solid var(--border); display:flex; flex-direction:column; height:calc(100vh - 49px); position:sticky; top:49px; }}
  .side-hdr {{ padding:18px 16px 12px; border-bottom:1px solid var(--border); }}
  .side-hdr h2 {{ font-size:0.95rem; font-weight:700; color:var(--accent); margin-bottom:2px; }}
  .side-hdr p {{ font-size:0.72rem; color:var(--muted); }}
  .search-box {{ padding:10px 16px; border-bottom:1px solid var(--border); }}
  .search-box input {{ width:100%; padding:8px 12px; background:var(--bg); border:1px solid var(--border); border-radius:7px; color:var(--text); font-family:'DM Sans',sans-serif; font-size:0.82rem; outline:none; transition:border-color .2s; }}
  .search-box input::placeholder {{ color:var(--dim); }}
  .search-box input:focus {{ border-color:var(--accent); }}
  .flt-bar {{ padding:8px 16px; border-bottom:1px solid var(--border); display:flex; gap:5px; flex-wrap:wrap; }}
  .flt-btn {{ padding:3px 10px; border-radius:14px; border:1px solid var(--border); background:transparent; color:var(--muted); font-family:'DM Sans',sans-serif; font-size:0.69rem; font-weight:500; cursor:pointer; transition:all .15s; }}
  .flt-btn:hover {{ border-color:var(--accent); color:var(--accent); }}
  .flt-btn.active {{ background:var(--accent-lt); border-color:var(--accent); color:var(--accent); font-weight:600; }}
  .list-cnt {{ padding:5px 16px; font-size:0.69rem; color:var(--dim); border-bottom:1px solid var(--border-lt); background:var(--bg); }}
  .b-list {{ flex:1; overflow-y:auto; }}
  .b-list::-webkit-scrollbar {{ width:4px; }}
  .b-list::-webkit-scrollbar-thumb {{ background:var(--border); border-radius:3px; }}
  .b-item {{ padding:9px 16px; cursor:pointer; transition:all .1s; border-left:3px solid transparent; border-bottom:1px solid var(--border-lt); }}
  .b-item:hover {{ background:var(--bg); }}
  .b-item.active {{ background:var(--accent-lt); border-left-color:var(--accent); }}
  .b-item .code {{ font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:var(--accent); font-weight:600; }}
  .b-item .name {{ font-size:0.8rem; margin-top:1px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .b-item .meta {{ font-size:0.67rem; color:var(--dim); margin-top:2px; display:flex; gap:7px; align-items:center; }}
  .b-item .badge {{ background:var(--bg); padding:1px 6px; border-radius:8px; font-weight:600; font-size:0.65rem; border:1px solid var(--border); }}
  .sdot {{ width:6px; height:6px; border-radius:50%; display:inline-block; }}
  .sdot.open {{ background:var(--green); }}
  .sdot.closed {{ background:var(--red); }}
  .detail {{ padding:28px 32px; overflow-y:auto; height:calc(100vh - 49px); }}
  .empty-st {{ display:flex; flex-direction:column; align-items:center; justify-content:center; height:50vh; color:var(--dim); }}
  .empty-st .arr {{ font-size:2rem; margin-bottom:10px; opacity:.3; }}

  .br-hdr {{ margin-bottom:24px; padding-bottom:18px; border-bottom:1px solid var(--border); }}
  .br-hdr .top {{ display:flex; align-items:baseline; gap:12px; margin-bottom:6px; }}
  .br-hdr .bcode {{ font-family:'JetBrains Mono',monospace; font-size:0.8rem; color:var(--accent); background:var(--accent-lt); padding:3px 10px; border-radius:5px; font-weight:600; }}
  .br-hdr h2 {{ font-size:1.3rem; font-weight:700; letter-spacing:-.02em; }}
  .chips {{ display:flex; gap:6px; margin-top:8px; flex-wrap:wrap; }}
  .chip {{ padding:3px 11px; border-radius:14px; font-size:0.7rem; font-weight:600; }}
  .chip.open {{ background:var(--green-bg); color:var(--green); }}
  .chip.closed {{ background:var(--red-bg); color:var(--red); }}
  .chip.cashless {{ background:var(--orange-bg); color:var(--orange); }}
  .chip.cash {{ background:var(--cyan-bg); color:var(--cyan); }}
  .chip.nfmt {{ background:var(--purple-bg); color:var(--purple); }}
  .chip.ofmt {{ background:var(--yellow-bg); color:var(--yellow); }}
  .sum-txt {{ margin-top:6px; font-size:0.78rem; color:var(--muted); }}

  .tl {{ position:relative; padding-left:30px; }}
  .tl::before {{ content:''; position:absolute; left:11px; top:5px; bottom:5px; width:2px; background:linear-gradient(to bottom,var(--accent),var(--border) 20%,var(--border) 85%,transparent); border-radius:2px; }}
  .tl-ev {{ position:relative; margin-bottom:18px; animation:fadeUp .22s ease forwards; opacity:0; }}
  @keyframes fadeUp {{ from {{ opacity:0; transform:translateY(5px); }} to {{ opacity:1; transform:translateY(0); }} }}
  .tl-ev .dot {{ position:absolute; left:-24px; top:13px; width:9px; height:9px; border-radius:50%; border:2px solid var(--border); background:var(--card); z-index:1; }}
  .tl-ev.initial .dot {{ background:var(--accent); border-color:var(--accent); }}
  .tl-ev.closed .dot {{ background:var(--red); border-color:var(--red); }}
  .tl-ev.reopened .dot {{ background:var(--green); border-color:var(--green); }}
  .tl-ev.cashless .dot {{ background:var(--orange); border-color:var(--orange); }}
  .tl-ev.cash_added .dot {{ background:var(--cyan); border-color:var(--cyan); }}
  .tl-ev.format_change .dot {{ background:var(--purple); border-color:var(--purple); }}
  .tl-ev.change .dot {{ background:var(--yellow); border-color:var(--yellow); }}
  .ev-card {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:12px 16px; transition:box-shadow .2s; }}
  .ev-card:hover {{ box-shadow:0 2px 8px rgba(0,0,0,.05); }}
  .ev-date {{ font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:var(--dim); margin-bottom:3px; }}
  .ev-lbl {{ font-size:0.88rem; font-weight:700; margin-bottom:6px; }}
  .tl-ev.initial .ev-lbl {{ color:var(--accent); }}
  .tl-ev.closed .ev-lbl {{ color:var(--red); }}
  .tl-ev.reopened .ev-lbl {{ color:var(--green); }}
  .tl-ev.cashless .ev-lbl {{ color:var(--orange); }}
  .tl-ev.cash_added .ev-lbl {{ color:var(--cyan); }}
  .tl-ev.format_change .ev-lbl {{ color:var(--purple); }}
  .tl-ev.change .ev-lbl {{ color:var(--yellow); }}
  .ch-row {{ display:flex; align-items:center; gap:7px; padding:4px 0; font-size:0.76rem; border-bottom:1px solid var(--border-lt); }}
  .ch-row:last-child {{ border-bottom:none; }}
  .ch-f {{ font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:var(--muted); min-width:100px; font-weight:600; }}
  .ch-old {{ color:var(--red); background:var(--red-bg); padding:1px 6px; border-radius:3px; font-size:0.7rem; text-decoration:line-through; }}
  .ch-arr {{ color:var(--dim); font-size:0.7rem; }}
  .ch-new {{ color:var(--green); background:var(--green-bg); padding:1px 6px; border-radius:3px; font-size:0.7rem; font-weight:600; }}
  .st-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); gap:3px 14px; margin-top:4px; }}
  .st-item {{ display:flex; justify-content:space-between; padding:2px 0; font-size:0.72rem; }}
  .st-k {{ color:var(--dim); font-family:'JetBrains Mono',monospace; font-size:0.66rem; }}
  .st-v {{ font-weight:500; }}

  #tabFormats {{ padding:28px 32px; max-width:1200px; margin:0 auto; }}
  .fmt-branch-list {{ display:flex; flex-wrap:wrap; gap:5px; margin-top:4px; }}
  .fmt-branch-pill {{ display:inline-flex; align-items:center; gap:5px; background:var(--purple-bg); border:1px solid #ddd6fe; border-radius:14px; padding:3px 10px; font-size:0.7rem; color:var(--purple); }}
  .fmt-branch-pill .pcode {{ font-family:'JetBrains Mono',monospace; font-weight:700; font-size:0.68rem; }}
  .fmt-branch-pill .pfmt {{ background:var(--purple); color:#fff; border-radius:8px; padding:1px 6px; font-size:0.63rem; font-weight:600; }}
  .fmt-branch-pill .pnum {{ background:var(--accent); color:#fff; border-radius:8px; padding:1px 6px; font-size:0.63rem; font-weight:600; }}
  .fmt-row-toggle {{ cursor:pointer; user-select:none; }}
  .fmt-row-toggle:hover td {{ background:#f0ebff !important; }}
  .fmt-detail-row td {{ padding:0 !important; border-bottom:1px solid var(--border-lt); }}
  .fmt-detail-inner {{ padding:10px 12px 14px 28px; background:#faf8ff; }}
  .expand-icon {{ display:inline-block; transition:transform .18s; font-size:0.65rem; margin-left:6px; color:var(--dim); }}
  .expand-icon.open {{ transform:rotate(90deg); }}

  /* Kalendar */
  .cal-wrap {{ background:var(--card); border-radius:10px; padding:18px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,.05); }}
  .cal-wrap h2 {{ font-size:0.95rem; margin-bottom:4px; }}
  .cal-wrap .sub {{ font-size:0.75rem; color:var(--muted); margin-bottom:14px; }}
  .cal-legend {{ display:flex; gap:14px; margin-bottom:12px; font-size:0.7rem; color:var(--muted); align-items:center; flex-wrap:wrap; }}
  .cal-legend-dot {{ width:10px; height:10px; border-radius:3px; display:inline-block; margin-right:4px; }}
  .cal-year {{ margin-bottom:10px; }}
  .cal-year-label {{ font-size:0.78rem; font-weight:700; color:var(--muted); margin-bottom:5px; letter-spacing:.02em; }}
  .cal-months {{ display:grid; grid-template-columns:repeat(12,1fr); gap:4px; }}
  .cal-cell {{ border-radius:6px; padding:5px 4px; min-height:50px; border:1.5px solid var(--border-lt); background:var(--bg); transition:all .12s; position:relative; overflow:hidden; }}
  .cal-cell.clickable {{ cursor:pointer; }}
  .cal-cell.clickable:hover {{ border-color:var(--accent); transform:translateY(-1px); box-shadow:0 2px 8px rgba(0,0,0,.1); }}
  .cal-cell.has-closed {{ background:var(--red-bg); border-color:#fca5a5; }}
  .cal-cell.has-nf {{ background:var(--purple-bg); border-color:#c4b5fd; }}
  .cal-cell.has-both {{ background:linear-gradient(145deg,#fef2f2 55%,#f5f3ff 55%); border-color:#fca5a5; }}
  .cal-month-name {{ font-size:0.62rem; font-weight:700; color:var(--dim); margin-bottom:2px; }}
  .cal-cell.has-closed .cal-month-name {{ color:var(--red); }}
  .cal-cell.has-nf .cal-month-name {{ color:var(--purple); }}
  .cal-cell.has-both .cal-month-name {{ color:var(--red); }}
  .cal-codes {{ font-size:0.52rem; line-height:1.5; font-family:monospace; }}
  .cal-codes .cc-red {{ color:#991b1b; }}
  .cal-codes .cc-purple {{ color:#5b21b6; }}

  /* Modal */
  .modal-overlay {{ position:fixed; inset:0; background:rgba(0,0,0,.5); z-index:500; display:flex; align-items:center; justify-content:center; padding:20px; }}
  .modal-box {{ background:var(--card); border-radius:12px; padding:24px; max-width:580px; width:100%; max-height:82vh; overflow-y:auto; box-shadow:0 20px 60px rgba(0,0,0,.25); animation:modalIn .16s ease; }}
  @keyframes modalIn {{ from {{ opacity:0; transform:scale(.97) translateY(6px); }} to {{ opacity:1; transform:none; }} }}
  .modal-hdr {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; padding-bottom:12px; border-bottom:1px solid var(--border); }}
  .modal-hdr h3 {{ font-size:1.05rem; font-weight:700; }}
  .modal-close {{ background:none; border:none; font-size:1.1rem; color:var(--muted); cursor:pointer; padding:3px 9px; border-radius:6px; line-height:1; transition:all .15s; }}
  .modal-close:hover {{ background:var(--bg); color:var(--text); }}
  .modal-section {{ margin-bottom:14px; }}
  .modal-stitle {{ font-size:0.68rem; font-weight:700; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; margin-bottom:6px; display:flex; align-items:center; gap:6px; }}
  .modal-stitle .ms-dot {{ width:9px; height:9px; border-radius:50%; flex-shrink:0; }}
  .modal-item {{ display:flex; gap:8px; align-items:center; padding:5px 0; border-bottom:1px solid var(--border-lt); font-size:0.78rem; }}
  .modal-item:last-child {{ border-bottom:none; }}
  .modal-code {{ font-family:monospace; font-size:0.7rem; font-weight:700; color:var(--accent); background:var(--accent-lt); padding:1px 6px; border-radius:4px; flex-shrink:0; }}
  .modal-bname {{ font-weight:600; flex:1; }}
  .modal-detail {{ color:var(--muted); font-size:0.7rem; }}

  /* Uzavrene pobocky tab */
  #tabClosed {{ padding:28px 32px; max-width:1100px; margin:0 auto; }}
  .closed-date {{ font-family:'JetBrains Mono',monospace; font-size:0.78rem; font-weight:600; }}

  /* Cashless tab */
  #tabCashless {{ padding:28px 32px; max-width:1200px; margin:0 auto; }}
  .cl-pct-bar {{ height:6px; background:var(--border-lt); border-radius:3px; margin-top:3px; }}
  .cl-pct-fill {{ height:100%; border-radius:3px; background:var(--orange); transition:width .4s; }}
  .cl-pct-val {{ font-size:0.75rem; font-weight:700; color:var(--orange); }}

  /* Timeline tab */
  #tabTimeline {{ padding:28px 32px; max-width:1400px; margin:0 auto; }}
  .tl-year-block {{ margin-bottom:40px; }}
  .tl-year-title {{ font-size:1.1rem; font-weight:800; color:var(--accent); margin-bottom:18px; letter-spacing:-.02em; }}
  .tl-track {{ position:relative; overflow-x:auto; padding-bottom:4px; }}
  .tl-spine {{ display:flex; align-items:flex-start; gap:0; min-width:max-content; }}
  .tl-month-col {{ display:flex; flex-direction:column; align-items:center; min-width:90px; position:relative; }}
  .tl-month-col::before {{ content:''; position:absolute; top:16px; left:0; right:0; height:2px; background:var(--border); z-index:0; }}
  .tl-month-col:first-child::before {{ left:50%; }}
  .tl-month-col:last-child::before {{ right:50%; }}
  .tl-dot-wrap {{ position:relative; z-index:1; display:flex; flex-direction:column; align-items:center; gap:4px; margin-bottom:8px; }}
  .tl-dot {{ width:14px; height:14px; border-radius:50%; border:2.5px solid var(--border); background:var(--card); flex-shrink:0; }}
  .tl-dot.has-events {{ border-color:var(--accent); background:var(--accent); box-shadow:0 0 0 3px var(--accent-lt); cursor:pointer; }}
  .tl-dot.has-closed {{ background:var(--red); border-color:var(--red); box-shadow:0 0 0 3px var(--red-bg); }}
  .tl-dot.has-cashless {{ background:var(--orange); border-color:var(--orange); box-shadow:0 0 0 3px var(--orange-bg); }}
  .tl-dot.has-multi {{ background:linear-gradient(135deg,var(--red) 50%,var(--orange) 50%); border-color:var(--red); }}
  .tl-dot.has-nf {{ background:var(--purple); border-color:var(--purple); box-shadow:0 0 0 3px var(--purple-bg); }}
  .tl-month-lbl {{ font-size:0.62rem; font-weight:700; color:var(--dim); text-align:center; white-space:nowrap; }}
  .tl-month-events {{ padding:6px 4px 0; width:88px; }}
  .tl-chip {{ display:inline-flex; align-items:center; gap:3px; font-size:0.58rem; padding:1px 5px; border-radius:8px; margin:1px 0; white-space:nowrap; max-width:84px; overflow:hidden; text-overflow:ellipsis; font-weight:600; }}
  .tl-chip.cashless {{ background:var(--orange-bg); color:var(--orange); }}
  .tl-chip.closed {{ background:var(--red-bg); color:var(--red); }}
  .tl-chip.format {{ background:var(--purple-bg); color:var(--purple); }}
  .tl-chip-code {{ font-family:monospace; font-weight:700; }}
  .tl-more {{ font-size:0.58rem; color:var(--dim); margin-top:1px; text-align:center; }}
  .tl-legend {{ display:flex; gap:14px; margin-bottom:16px; flex-wrap:wrap; font-size:0.72rem; color:var(--muted); align-items:center; }}
  .tl-leg-dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:4px; }}

  /* Porovnani tab */
  #tabCompare {{ padding:28px 32px; max-width:1300px; margin:0 auto; }}
  .cmp-controls {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:20px 24px; margin-bottom:20px; display:flex; flex-wrap:wrap; gap:16px; align-items:flex-end; }}
  .cmp-field {{ display:flex; flex-direction:column; gap:5px; }}
  .cmp-field label {{ font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }}
  .cmp-field input[type=date] {{ font-family:'DM Sans',sans-serif; font-size:0.85rem; padding:7px 11px; border:1.5px solid var(--border); border-radius:7px; background:var(--bg); color:var(--text); outline:none; cursor:pointer; }}
  .cmp-field input[type=date]:focus {{ border-color:var(--accent); }}
  .cmp-vs {{ font-size:0.75rem; font-weight:700; color:var(--dim); padding-bottom:9px; }}
  .cmp-btn {{ padding:8px 22px; background:var(--accent); color:#fff; border:none; border-radius:7px; font-family:'DM Sans',sans-serif; font-size:0.85rem; font-weight:700; cursor:pointer; transition:opacity .15s; }}
  .cmp-btn:hover {{ opacity:.88; }}
  .cmp-summary {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:16px; }}
  .cmp-stat {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:10px 18px; font-size:0.78rem; }}
  .cmp-stat strong {{ display:block; font-size:1.1rem; font-weight:800; }}
  .cmp-stat.added strong {{ color:var(--green); }}
  .cmp-stat.removed strong {{ color:var(--red); }}
  .cmp-stat.changed strong {{ color:var(--orange); }}
  .cmp-stat.same strong {{ color:var(--dim); }}
  .cmp-filters {{ display:flex; gap:10px; margin-bottom:12px; flex-wrap:wrap; }}
  .cmp-filter-btn {{ padding:4px 14px; font-size:0.75rem; font-weight:600; border-radius:14px; border:1.5px solid var(--border); background:var(--bg); color:var(--muted); cursor:pointer; transition:all .14s; }}
  .cmp-filter-btn.active {{ background:var(--accent); border-color:var(--accent); color:#fff; }}
  .cmp-table-wrap {{ overflow-x:auto; border-radius:10px; border:1px solid var(--border); }}
  .cmp-table {{ width:100%; border-collapse:collapse; font-size:0.78rem; }}
  .cmp-table th {{ background:var(--bg); padding:9px 12px; text-align:left; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); white-space:nowrap; border-bottom:1.5px solid var(--border); position:sticky; top:0; z-index:2; }}
  .cmp-table td {{ padding:7px 12px; border-bottom:1px solid var(--border-lt); vertical-align:middle; white-space:nowrap; }}
  .cmp-table tr:last-child td {{ border-bottom:none; }}
  .cmp-table tr:hover td {{ background:#f8f9fc; }}
  .cmp-row-added td {{ background:#f0fdf4 !important; }}
  .cmp-row-removed td {{ background:#fff5f5 !important; }}
  .cmp-row-changed td {{ }}
  .cmp-cell-changed {{ background:var(--orange-bg) !important; border-radius:3px; }}
  .cmp-badge {{ display:inline-block; font-size:0.6rem; font-weight:700; padding:1px 7px; border-radius:9px; margin-right:5px; text-transform:uppercase; letter-spacing:.04em; }}
  .cmp-badge.new {{ background:#dcfce7; color:#166534; }}
  .cmp-badge.del {{ background:#fee2e2; color:#991b1b; }}
  .cmp-badge.chg {{ background:#ffedd5; color:#92400e; }}
  .cmp-cell-val {{ display:flex; flex-direction:column; gap:1px; }}
  .cmp-val-old {{ font-size:0.67rem; color:var(--red); text-decoration:line-through; }}
  .cmp-val-new {{ font-size:0.78rem; color:var(--green); font-weight:600; }}
  .cmp-val-same {{ color:var(--text); }}
  .bool-true {{ color:var(--green); font-weight:700; }}
  .bool-false {{ color:var(--dim); }}
  .cmp-code {{ font-family:'JetBrains Mono',monospace; font-size:0.7rem; font-weight:700; color:var(--accent); }}
  .cmp-placeholder {{ text-align:center; padding:60px 20px; color:var(--dim); font-size:0.88rem; }}

  .foot {{ text-align:center; padding:16px; font-size:0.7rem; color:var(--dim); border-top:1px solid var(--border-lt); margin-top:20px; }}
  @media (max-width:860px) {{
    #tabDetail.active {{ grid-template-columns:1fr; }}
    .side {{ position:relative; height:auto; max-height:38vh; }}
    .detail {{ height:auto; }}
    #tabOverview,#tabFormats,#tabClosed,#tabCompare,#tabCashless,#tabTimeline {{ padding:20px 16px; }}
    .cal-months {{ grid-template-columns:repeat(6,1fr); }}
  }}
</style>
</head>
<body>

<div class="tab-nav">
  <div class="brand">Pobčková síť ČS</div>
  <button class="tab-btn active" data-tab="tabOverview">Přehled sítě</button>
  <button class="tab-btn" data-tab="tabDetail">Detail pobčky</button>
  <button class="tab-btn" data-tab="tabFormats">Nové formáty</button>
  <button class="tab-btn" data-tab="tabClosed">Uzavřené pobčky</button>
  <button class="tab-btn" data-tab="tabCashless">Přechod na cashless</button>
  <button class="tab-btn" data-tab="tabTimeline">Timeline 2024–2026</button>
  <button class="tab-btn" data-tab="tabCompare">Porovnání</button>
</div>

<div id="tabOverview" class="tab-content active">
  <div class="ov-header">
    <h1>Měsíční snapshoty pobčkové sítě</h1>
    <p>Stav sítě v čase — pouze pobčky s bns_flag = &quot;Y&quot;</p>
    <div class="range" id="rangeLabel"></div>
  </div>
  <div id="calendarSection"></div>
  <div class="kpi-row" id="kpiRow"></div>
  <div class="chart-card">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;flex-wrap:wrap;gap:10px;">
      <div><h2 style="margin-bottom:2px;">Poslední změny v sítě</h2><div class="sub" style="margin-bottom:0;">Klíčové události napříč všemi pobčkami</div></div>
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
      <div><h2 style="margin-bottom:2px;">Vývoj pobčkové sítě</h2><div class="sub" style="margin-bottom:0;">Starý formát (modrá) + NF (zelená) · zrušené pod osou · cashless linie</div></div>
      <div class="rc-tabs" id="granTabs">
        <button class="rc-tab" data-gran="M">Měsíce</button>
        <button class="rc-tab active" data-gran="Q">Kvarty</button>
        <button class="rc-tab" data-gran="Y">Roky</button>
      </div>
    </div>
    <div id="chartMain"></div>
  </div>
  <div class="chart-card"><h2>Cashless vs. s hotovostí</h2><div class="sub">Z otevřených pobček</div><div id="chartCashless"></div></div>
  <div class="chart-card"><h2>Nový formát (NF) vs. starý</h2><div class="sub">Z otevřených — NF = branch_building_nf_sf==NF &amp; format vyplněný &amp; nf_number je číslo</div><div id="chartFormat"></div></div>
  <div class="chart-card"><h2>Struktura sítě v čase</h2><div class="sub">Stacked columns</div><div id="chartStacked"></div></div>
  <div class="table-wrap">
    <h2>Všechny měsíční snapshoty</h2>
    <table id="historyTable">
      <thead><tr><th>Měsíc</th><th>Celkem</th><th>Otevřené</th><th>Zavřené</th><th>Cashless</th><th>S hotovostí</th><th>NF</th><th>Starý</th><th>Δ Otevřené</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>
  <div class="foot">Vygenerováno automaticky — filtr: bns_flag = &quot;Y&quot; · NF = nf_sf==NF + format + nf_number</div>
</div>

<div id="tabDetail" class="tab-content">
  <div class="side">
    <div class="side-hdr"><h2>Timeline pobček</h2><p>Historie změn stavů</p></div>
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
    <div class="empty-st"><div class="arr">&#9664;</div><p>Vyberte pobčku ze seznamu vlevo</p></div>
  </div>
</div>

<div id="tabFormats" class="tab-content">
  <div class="ov-header">
    <h1>Adopce nových formátů (NF)</h1>
    <p>Pobčky, které přešly na NF: branch_building_nf_sf=NF + format vyplněný + nf_number je číslo</p>
  </div>
  <div class="kpi-row" id="fmtKpiRow"></div>
  <div class="chart-card">
    <h2>Měsíční přírůstek pobček s NF</h2>
    <div class="sub">Počet pobček, které v daném měsíci poprvé splňovaly všechny 3 podmínky NF (sloupce) · kumulativní součet (linie)</div>
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
        <th style="text-align:right;">Kumulativní</th>
        <th style="text-align:left;">Pobčky (klikněte)</th>
      </tr></thead>
      <tbody id="fmtTableBody"></tbody>
    </table>
  </div>
  <div class="foot">NF podmínky: branch_building_nf_sf = NF &amp; format vyplněný &amp; nf_number je číslo</div>
</div>

<div id="tabClosed" class="tab-content">
  <div class="ov-header">
    <h1>Uzavřené pobčky</h1>
    <p>Pobčky s branch_closed = True — seřazeno od nejnovějšího uzavření</p>
    <div class="range" id="closedCountBadge"></div>
  </div>
  <div class="chart-card" style="margin-bottom:16px;">
    <h2>Uzavírání poboček dle regionů ČS</h2>
    <div class="sub">Uzavřené pobočky za rok, dle interních regionů ČS</div>
    <div id="chartClosedRegion"></div>
  </div>
  <div class="chart-card" style="margin-bottom:16px;">
    <h2>Uzavírání poboček dle krajů ČR</h2>
    <div class="sub">Uzavřené pobočky za rok, přiřazení dle města pobočky</div>
    <div id="chartClosedKraj"></div>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th style="text-align:left;">Datum uzavření</th>
        <th style="text-align:left;">Kód</th>
        <th style="text-align:left;">Název pobčky</th>
        <th style="text-align:left;">Město</th>
        <th style="text-align:left;">Region</th>
        <th style="text-align:left;">Typ</th>
      </tr></thead>
      <tbody id="closedTableBody"></tbody>
    </table>
  </div>
</div>

<div id="tabCashless" class="tab-content">
  <div class="ov-header">
    <h1>Přechod poboček na cashless</h1>
    <p>Pobočky seřazené dle data přechodu na bezhotovostní provoz · průběžné % z otevřených poboček</p>
    <div class="range" id="cashlessCountBadge"></div>
  </div>
  <div class="kpi-row" id="cashlessKpiRow"></div>
  <div class="chart-card" style="margin-bottom:16px;">
    <h2>Podíl cashless poboček v čase</h2>
    <div class="sub">% bezhotovostních poboček z celkového počtu otevřených — po každém přechodu</div>
    <div id="chartCashlessPct"></div>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th style="text-align:left;">Datum přechodu</th>
        <th style="text-align:left;">Kód</th>
        <th style="text-align:left;">Název pobočky</th>
        <th style="text-align:left;">Město</th>
        <th style="text-align:left;">Region</th>
        <th style="text-align:right;">Cashless</th>
        <th style="text-align:right;">Otevřených</th>
        <th style="text-align:right;">% cashless</th>
      </tr></thead>
      <tbody id="cashlessTableBody"></tbody>
    </table>
  </div>
</div>

<div id="tabTimeline" class="tab-content">
  <div class="ov-header">
    <h1>Timeline změn 2024–2026</h1>
    <p>Chronologický přehled přechodů na cashless, uzavření poboček a změn formátu NF/SF po měsících</p>
  </div>
  <div class="tl-legend">
    <span><span class="tl-leg-dot" style="background:var(--red);"></span>Uzavření pobočky</span>
    <span><span class="tl-leg-dot" style="background:var(--orange);"></span>Cashless přechod</span>
    <span><span class="tl-leg-dot" style="background:var(--purple);"></span>Změna formátu NF/SF</span>
  </div>
  <div id="timelineContent"></div>
</div>

<div id="tabCompare" class="tab-content">
  <div class="ov-header">
    <h1>Porovnání stavů sítě</h1>
    <p>Porovnej stav pobčkové sítě mezi dvěma daty — zobrازí se všechny pobčky s vyznačenými změnami</p>
  </div>
  <div class="cmp-controls">
    <div class="cmp-field">
      <label>Datum A</label>
      <input type="date" id="cmpDateA">
    </div>
    <span class="cmp-vs">→</span>
    <div class="cmp-field">
      <label>Datum B</label>
      <input type="date" id="cmpDateB">
    </div>
    <button class="cmp-btn" id="cmpRunBtn" onclick="runComparison()">Porovnat</button>
  </div>
  <div id="cmpSummary" style="display:none;">
    <div class="cmp-summary" id="cmpStats"></div>
    <div class="cmp-filters">
      <button class="cmp-filter-btn active" data-f="all" onclick="setCmpFilter(this,'all')">Vše</button>
      <button class="cmp-filter-btn" data-f="changed" onclick="setCmpFilter(this,'changed')">Změněné</button>
      <button class="cmp-filter-btn" data-f="added" onclick="setCmpFilter(this,'added')">Nové</button>
      <button class="cmp-filter-btn" data-f="removed" onclick="setCmpFilter(this,'removed')">Zrušené</button>
      <button class="cmp-filter-btn" data-f="same" onclick="setCmpFilter(this,'same')">Beze změny</button>
    </div>
    <div class="cmp-table-wrap">
      <table class="cmp-table" id="cmpTable">
        <thead id="cmpThead"></thead>
        <tbody id="cmpTbody"></tbody>
      </table>
    </div>
  </div>
  <div id="cmpPlaceholder" class="cmp-placeholder">Vyberte dvě data a klikněte na Porovnat</div>
</div>

<!-- Kalendar modal -->
<div id="calModal" class="modal-overlay" style="display:none;">
  <div class="modal-box">
    <div class="modal-hdr">
      <h3 id="calModalTitle"></h3>
      <button class="modal-close" id="calModalClose">&#x2715;</button>
    </div>
    <div id="calModalBody"></div>
  </div>
</div>

<script>
let chartsRendered=false, fmtChartsRendered=false, closedChartsRendered=false, cashlessChartsRendered=false, timelineRendered=false;

document.querySelectorAll('.tab-btn').forEach(btn => {{
  btn.addEventListener('click',()=>{{
    document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
    if(btn.dataset.tab==='tabOverview'&&!chartsRendered) renderCharts();
    if(btn.dataset.tab==='tabFormats'&&!fmtChartsRendered) renderFmtCharts();
    if(btn.dataset.tab==='tabClosed'&&!closedChartsRendered) renderClosedCharts();
    if(btn.dataset.tab==='tabCashless'&&!cashlessChartsRendered) renderCashlessTab();
    if(btn.dataset.tab==='tabTimeline'&&!timelineRendered) renderTimeline();
  }});
}});

const historyByGran={history_all_json};
const history=historyByGran['M'];
const hLast=history[history.length-1];
const hPrev=history.length>1?history[history.length-2]:null;
const hFirst=history[0];

document.getElementById('rangeLabel').textContent=hFirst.label+' → '+hLast.label+' ('+history.length+' záznamů)';

function mkDelta(curr,old){{
  if(old===null||old===undefined) return '<span class="delta neutral">—</span>';
  const d=curr-old;
  if(d===0) return '<span class="delta neutral">beze změny</span>';
  return '<span class="delta '+(d>0?'up':'down')+'">'+(d>0?'+':'')+d+' oproti min.</span>';
}}

document.getElementById('kpiRow').innerHTML=[
  {{label:'Celkem',value:hLast.total,color:'var(--accent)',key:'total'}},
  {{label:'Otevřené',value:hLast.opened,color:'var(--green)',key:'opened'}},
  {{label:'Zavřené',value:hLast.closed,color:'var(--red)',key:'closed'}},
  {{label:'Cashless',value:hLast.cashless,color:'var(--orange)',key:'cashless'}},
  {{label:'S hotovostí',value:hLast.non_cashless,color:'var(--blue)',key:'non_cashless'}},
  {{label:'NF formát',value:hLast.new_format,color:'var(--purple)',key:'new_format'}},
  {{label:'Starý formát',value:hLast.old_format,color:'var(--teal)',key:'old_format'}},
].map(k=>'<div class="kpi" style="border-top-color:'+k.color+'"><div class="value" style="color:'+k.color+'">'+k.value+'</div><div class="label">'+k.label+'</div>'+mkDelta(k.value,hPrev?hPrev[k.key]:null)+'</div>').join('');

const recentChangesByCat={recent_changes_json};
let activeRcCat='all';
const fldLabelsRC={{branch_name:'Název',branch_type:'Typ',branch_closed:'Zavřeno',cashless:'Cashless',format:'Formát',branch_building_nf_sf:'NF/SF',nf_number:'NF číslo',address:'Adresa',city:'Město',region:'Region'}};
function fvRC(v){{if(v===null||v===undefined)return'—';if(v===true)return'Ano';if(v===false)return'Ne';if(v===''||v==='nan')return'—';return String(v);}}
function esc(s){{const d=document.createElement('div');d.textContent=String(s==null?'':s);return d.innerHTML;}}

function renderRecentChanges(){{
  const shown=(recentChangesByCat[activeRcCat]||[]).slice(0,10);
  const el=document.getElementById('recentChangesList');
  if(!shown.length){{el.innerHTML='<div style="color:var(--dim);font-size:0.8rem;padding:12px 0;text-align:center;">Žádné změny.</div>';return;}}
  el.innerHTML=shown.map(c=>
    '<div class="rc-item"><div class="rc-dot '+c.category+'"></div><div class="rc-body">'+
    '<div class="rc-top"><span class="rc-date">'+c.date+'</span><span class="rc-code">'+c.branch_code+'</span>'+
    '<span class="rc-bname">'+esc(c.branch_name)+'</span><span class="rc-evlabel">'+esc(c.label)+'</span></div>'+
    '<div class="rc-change"><span class="fld">'+(fldLabelsRC[c.field]||c.field)+'</span>'+
    '<span class="old">'+esc(fvRC(c.old))+'</span><span class="arr">→</span><span class="new">'+esc(fvRC(c.new))+'</span></div>'+
    '</div></div>'
  ).join('');
}}
document.querySelectorAll('#rcTabs .rc-tab').forEach(b=>b.addEventListener('click',()=>{{
  document.querySelectorAll('#rcTabs .rc-tab').forEach(x=>x.classList.remove('active'));
  b.classList.add('active');activeRcCat=b.dataset.cat;renderRecentChanges();
}}));
renderRecentChanges();

/* ====== KALENDAR ====== */
const changesByMonth={changes_by_month_json};
const fmtData={format_monthly_json};
const fmtByMonthMap={{}};
fmtData.forEach(m=>{{fmtByMonthMap[m.month]=m.branches;}});

const MONTH_SHORT=['Led','Úno','Bře','Dub','Kvě','Čvn','Čvc','Srp','Zář','Říj','Lis','Pro'];
const MONTH_FULL=['Leden','Únor','Březen','Duben','Květen','Červen','červenec','Srpen','Září','Říjen','Listopad','Prosinec'];

function renderCalendar(){{
  const yearsSet=new Set();
  history.forEach(h=>yearsSet.add(h.timestamp.substring(0,4)));
  Object.keys(changesByMonth).forEach(m=>yearsSet.add(m.substring(0,4)));
  Object.keys(fmtByMonthMap).forEach(m=>yearsSet.add(m.substring(0,4)));
  const years=[...yearsSet].sort();

  let html='<div class="cal-wrap"><h2>Kalendářní přehled sítě</h2>';
  html+='<div class="sub">Kliknutím na měsíc otevřete detail událostí</div>';
  html+='<div class="cal-legend">';
  html+='<span><span class="cal-legend-dot" style="background:var(--red-bg);border:1.5px solid #fca5a5;"></span>Uzavřené pobčky</span>';
  html+='<span><span class="cal-legend-dot" style="background:var(--purple-bg);border:1.5px solid #c4b5fd;"></span>NF formát přechod</span>';
  html+='<span><span class="cal-legend-dot" style="background:linear-gradient(145deg,#fef2f2 55%,#f5f3ff 55%);border:1.5px solid #fca5a5;"></span>Obě události</span>';
  html+='</div>';

  years.forEach(yr=>{{
    html+='<div class="cal-year"><div class="cal-year-label">'+yr+'</div><div class="cal-months">';
    for(let mo=1;mo<=12;mo++){{
      const mKey=yr+'-'+String(mo).padStart(2,'0');
      const mChg=changesByMonth[mKey]||[];
      const closures=mChg.filter(c=>c.category==='closed'&&c.new===true);
      const nfTrans=fmtByMonthMap[mKey]||[];
      const hasClosed=closures.length>0;
      const hasNF=nfTrans.length>0;
      const hasAny=hasClosed||hasNF||mChg.length>0;
      let cls='cal-cell';
      if(hasAny) cls+=' clickable';
      if(hasClosed&&hasNF) cls+=' has-both';
      else if(hasClosed) cls+=' has-closed';
      else if(hasNF) cls+=' has-nf';
      let inner='<div class="cal-month-name">'+MONTH_SHORT[mo-1]+'</div><div class="cal-codes">';
      if(hasClosed){{
        const shown=closures.slice(0,4).map(c=>c.branch_code).join(' ');
        inner+='<div class="cc-red">✕ '+shown+(closures.length>4?' +'+(closures.length-4):'')+'</div>';
      }}
      if(hasNF){{
        const shown=nfTrans.slice(0,3).map(b=>b.branch_code).join(' ');
        inner+='<div class="cc-purple">◆ '+shown+(nfTrans.length>3?' +'+(nfTrans.length-3):'')+'</div>';
      }}
      inner+='</div>';
      html+='<div class="'+cls+'"'+(hasAny?' data-month="'+mKey+'"':'')+'>'  +inner+'</div>';
    }}
    html+='</div></div>';
  }});
  html+='</div>';
  document.getElementById('calendarSection').innerHTML=html;
  document.getElementById('calendarSection').addEventListener('click',e=>{{
    const cell=e.target.closest('[data-month]');
    if(cell) openCalModal(cell.dataset.month);
  }});
}}

function openCalModal(mKey){{
  const yr=mKey.substring(0,4);
  const mo=parseInt(mKey.substring(5,7),10);
  const title=MONTH_FULL[mo-1]+' '+yr;
  const mChg=changesByMonth[mKey]||[];
  const closures=mChg.filter(c=>c.category==='closed');
  const nfTrans=fmtByMonthMap[mKey]||[];
  const cashless=mChg.filter(c=>c.category==='cashless');
  const others=mChg.filter(c=>c.category!=='closed'&&c.category!=='cashless');
  let body='';
  if(closures.length){{
    body+='<div class="modal-section"><div class="modal-stitle"><span class="ms-dot" style="background:var(--red);"></span>Uzavřené pobčky ('+closures.length+')</div>';
    closures.forEach(c=>{{body+='<div class="modal-item"><span class="modal-code">'+c.branch_code+'</span><span class="modal-bname">'+esc(c.branch_name)+'</span><span class="modal-detail">'+esc(c.label)+'</span></div>';}});
    body+='</div>';
  }}
  if(nfTrans.length){{
    body+='<div class="modal-section"><div class="modal-stitle"><span class="ms-dot" style="background:var(--purple);"></span>Nové formáty NF ('+nfTrans.length+')</div>';
    nfTrans.forEach(b=>{{body+='<div class="modal-item"><span class="modal-code">'+b.branch_code+'</span><span class="modal-bname">'+esc(b.branch_name)+'</span><span class="modal-detail">'+esc(b.format_new)+(b.nf_number&&b.nf_number!=='None'?' #'+esc(b.nf_number):'')+'</span></div>';}});
    body+='</div>';
  }}
  if(cashless.length){{
    body+='<div class="modal-section"><div class="modal-stitle"><span class="ms-dot" style="background:var(--orange);"></span>Cashless změny ('+cashless.length+')</div>';
    cashless.forEach(c=>{{body+='<div class="modal-item"><span class="modal-code">'+c.branch_code+'</span><span class="modal-bname">'+esc(c.branch_name)+'</span><span class="modal-detail">'+esc(c.label)+'</span></div>';}});
    body+='</div>';
  }}
  if(others.length){{
    body+='<div class="modal-section"><div class="modal-stitle"><span class="ms-dot" style="background:var(--yellow);"></span>Ostatní změny ('+others.length+')</div>';
    others.forEach(c=>{{body+='<div class="modal-item"><span class="modal-code">'+c.branch_code+'</span><span class="modal-bname">'+esc(c.branch_name)+'</span><span class="modal-detail">'+esc(c.label)+'</span></div>';}});
    body+='</div>';
  }}
  if(!body) body='<div style="color:var(--dim);text-align:center;padding:24px;">Žádné klíčové události.</div>';
  document.getElementById('calModalTitle').textContent=title;
  document.getElementById('calModalBody').innerHTML=body;
  document.getElementById('calModal').style.display='flex';
}}

function closeCalModal(){{
  document.getElementById('calModal').style.display='none';
}}
document.getElementById('calModalClose').addEventListener('click',closeCalModal);
document.getElementById('calModal').addEventListener('click',e=>{{if(e.target===document.getElementById('calModal')) closeCalModal();}});
document.addEventListener('keydown',e=>{{if(e.key==='Escape') closeCalModal();}});

renderCalendar();

function renderCharts(){{
  chartsRendered=true;
  function makeChart(el,series,colors,stacked){{
    new ApexCharts(document.querySelector(el),{{
      chart:{{type:'bar',height:280,stacked:stacked||false,fontFamily:'DM Sans,sans-serif',toolbar:{{show:true}},animations:{{enabled:true,easing:'easeinout',speed:400}}}},
      series,colors,
      xaxis:{{categories:history.map(h=>h.label),labels:{{rotate:-45,style:{{fontSize:'10px'}}}}}},
      yaxis:{{labels:{{style:{{fontSize:'11px'}}}},min:0}},
      plotOptions:{{bar:{{columnWidth:'70%',borderRadius:2,borderRadiusApplication:'end',borderRadiusWhenStacked:'last'}}}},
      dataLabels:{{enabled:history.length<=12}},
      tooltip:{{shared:true,intersect:false}},
      legend:{{position:'top',fontSize:'12px'}},
      grid:{{borderColor:'#e8eaf0',strokeDashArray:3}},
    }}).render();
  }}

  window.buildMainChartOptions=function(hist){{
    return {{
      chart:{{type:'bar',height:380,stacked:true,fontFamily:'DM Sans,sans-serif',toolbar:{{show:true}},animations:{{enabled:true,easing:'easeinout',speed:400}}}},
      series:[
        {{name:'Starý formát',type:'bar',data:hist.map(h=>h.old_format)}},
        {{name:'NF formát',type:'bar',data:hist.map(h=>h.new_format)}},
        {{name:'Zrušené',type:'bar',data:hist.map(h=>-h.closed)}},
        {{name:'Cashless',type:'line',data:hist.map(h=>h.cashless)}},
      ],
      colors:['#3b82f6','#10b981','#d1d5db','#1e293b'],
      stroke:{{width:[0,0,0,2.5],dashArray:[0,0,0,6],curve:'smooth'}},
      plotOptions:{{bar:{{columnWidth:'65%',borderRadius:2,borderRadiusApplication:'end',borderRadiusWhenStacked:'last'}}}},
      xaxis:{{categories:hist.map(h=>h.label),labels:{{rotate:-45,style:{{fontSize:'10px'}}}},axisBorder:{{show:true,color:'#94a3b8'}}}},
      yaxis:{{labels:{{style:{{fontSize:'11px'}},formatter:v=>Math.abs(Math.round(v))}}}},
      dataLabels:{{enabled:hist.length<=14,formatter:val=>{{const v=Math.abs(Math.round(val));return v===0?'':(val<0?'-'+v:v);}},style:{{fontSize:'10px',fontWeight:700,colors:['#fff','#fff','#64748b','#1e293b']}},background:{{enabled:true,foreColor:'#fff',borderRadius:2,padding:3,opacity:0.85,borderWidth:0}}}},
      tooltip:{{shared:true,intersect:false,y:{{formatter:val=>Math.abs(Math.round(val))}}}},
      legend:{{position:'top',fontSize:'12px',markers:{{width:10,height:10,radius:2}}}},
      grid:{{borderColor:'#e8eaf0',strokeDashArray:3}},
      annotations:{{
        yaxis:[{{y:0,borderColor:'#94a3b8',strokeDashArray:0,borderWidth:1}}],
        points:hist.map(h=>({{x:h.label,y:h.opened,seriesIndex:1,marker:{{size:0}},label:{{text:String(h.opened),borderColor:'transparent',borderWidth:0,borderRadius:3,style:{{background:'transparent',color:'#1e293b',fontSize:'11px',fontWeight:700,padding:{{left:4,right:4,top:2,bottom:2}}}},offsetY:-8}}}})),
      }},
    }};
  }};

  window.mainChart=new ApexCharts(document.querySelector('#chartMain'),window.buildMainChartOptions(historyByGran['Q']));
  window.mainChart.render();
  makeChart('#chartCashless',[{{name:'Cashless',data:history.map(h=>h.cashless)}},{{name:'S hotovostí',data:history.map(h=>h.non_cashless)}}],['#d97706','#2563eb'],true);
  makeChart('#chartFormat',[{{name:'NF formát',data:history.map(h=>h.new_format)}},{{name:'Starý formát',data:history.map(h=>h.old_format)}}],['#7c3aed','#0d9488'],true);
  makeChart('#chartStacked',[
    {{name:'Cashless+NF',data:history.map(h=>Math.min(h.cashless,h.new_format))}},
    {{name:'Cashless+starý',data:history.map(h=>Math.max(0,h.cashless-h.new_format))}},
    {{name:'Hotovost+NF',data:history.map(h=>Math.max(0,h.new_format-h.cashless))}},
    {{name:'Hotovost+starý',data:history.map(h=>h.old_format)}},
  ],['#f59e0b','#ef4444','#8b5cf6','#6b7280'],true);
}}

const tbody=document.querySelector('#historyTable tbody');
history.slice().reverse().forEach((h,i,arr)=>{{
  const prevH=i<arr.length-1?arr[i+1]:null;
  const dOpen=prevH?h.opened-prevH.opened:0;
  const tr=document.createElement('tr');
  if(h.is_current) tr.className='is-current';
  tr.innerHTML='<td>'+h.label+'</td><td>'+h.total+'</td><td><strong>'+h.opened+'</strong></td><td>'+h.closed+'</td><td>'+h.cashless+'</td><td>'+h.non_cashless+'</td><td>'+h.new_format+'</td><td>'+h.old_format+'</td><td><span class="delta '+(dOpen>0?'up':dOpen<0?'down':'neutral')+'">'+(prevH?(dOpen>0?'+':'')+dOpen:'—')+'</span></td>';
  tbody.appendChild(tr);
}});

renderCharts();
document.querySelectorAll('#granTabs .rc-tab').forEach(b=>b.addEventListener('click',()=>{{
  document.querySelectorAll('#granTabs .rc-tab').forEach(x=>x.classList.remove('active'));
  b.classList.add('active');
  const hist=historyByGran[b.dataset.gran]||[];
  if(window.mainChart&&hist.length) window.mainChart.updateOptions(window.buildMainChartOptions(hist),true,true);
}}));

/* ====== TAB 2 — DETAIL ====== */
const branches={branches_json};
let activeFilter='all',activeCode=null;
const searchInput=document.getElementById('searchInput');
const branchList=document.getElementById('branchList');
const detailContent=document.getElementById('detailContent');
const listCount=document.getElementById('listCount');
const fldLabels={{branch_name:'Název',branch_type:'Typ',branch_closed:'Zavřeno',cashless:'Cashless',format:'Formát',branch_building_nf_sf:'NF/SF',nf_number:'NF číslo',address:'Adresa',city:'Město',region:'Region'}};

function isNewFormat(st){{
  const nf=st.branch_building_nf_sf;
  const fm=st.format;
  const nn=st.nf_number;
  if(!nf||String(nf).trim()!=='NF') return false;
  if(!fm||['','nan','None'].includes(String(fm).trim())) return false;
  if(nn===null||nn===undefined) return false;
  return !isNaN(parseFloat(String(nn)));
}}
function fv(v){{if(v===null||v===undefined)return'—';if(v===true)return'Ano';if(v===false)return'Ne';if(v===''||v==='nan')return'—';return String(v);}}

function filterBranches(){{
  const q=searchInput.value.toLowerCase().trim();
  return branches.filter(b=>{{
    if(q&&!String(b.code).includes(q)&&!b.name.toLowerCase().includes(q)) return false;
    const last=b.events[b.events.length-1];
    if(activeFilter==='has-changes'&&b.total_changes===0) return false;
    if(activeFilter==='closed'&&!last.state.branch_closed) return false;
    if(activeFilter==='cashless'&&!last.state.cashless) return false;
    return true;
  }});
}}

function renderList(){{
  const filtered=filterBranches();
  listCount.textContent=filtered.length+' pobček';
  branchList.innerHTML=filtered.map(b=>{{
    const last=b.events[b.events.length-1];
    const cls=activeCode===b.code?' active':'';
    const dc=last.state.branch_closed?'closed':'open';
    return '<div class="b-item'+cls+'" data-code="'+b.code+'"><div class="code">'+b.code+'</div><div class="name">'+esc(b.name)+'</div><div class="meta"><span class="badge">'+b.total_changes+' změn</span><span class="sdot '+dc+'"></span>'+(last.state.branch_closed?'zavřena':'otevřena')+(last.state.cashless?' · cashless':'')+(isNewFormat(last.state)?' · NF':'')+'</div></div>';
  }}).join('');
  branchList.querySelectorAll('.b-item').forEach(el=>el.addEventListener('click',()=>{{activeCode=parseInt(el.dataset.code);renderList();renderTimeline();}}));
}}

function renderTimeline(){{
  const br=branches.find(b=>b.code===activeCode); if(!br) return;
  const ls=br.events[br.events.length-1].state;
  const hf=isNewFormat(ls);
  let h='<div class="br-hdr"><div class="top"><span class="bcode">'+br.code+'</span><h2>'+esc(br.name)+'</h2></div><div class="chips">'+(ls.branch_closed?'<span class="chip closed">● Zavřena</span>':'<span class="chip open">● Otevřena</span>')+(ls.cashless?'<span class="chip cashless">Cashless</span>':'<span class="chip cash">S hotovostí</span>')+(hf?'<span class="chip nfmt">'+esc(ls.format)+(ls.nf_number!==null&&ls.nf_number!==undefined?' #'+ls.nf_number:'')+' NF</span>':'<span class="chip ofmt">Starý formát</span>')+'</div><div class="sum-txt">'+br.events.length+' záznamů · '+br.total_changes+' změn · '+br.events[0].date+' → '+br.events[br.events.length-1].date+'</div></div>';
  h+='<div class="tl">';
  br.events.forEach((evt,idx)=>{{
    const dl=Math.min(idx*0.04,0.5);
    h+='<div class="tl-ev '+evt.type+'" style="animation-delay:'+dl+'s"><div class="dot"></div><div class="ev-card"><div class="ev-date">'+evt.date+'</div><div class="ev-lbl">'+esc(evt.label)+'</div>';
    if(evt.type==='initial'){{
      h+='<div class="st-grid">';
      for(const [k,v] of Object.entries(evt.state)) h+='<div class="st-item"><span class="st-k">'+(fldLabels[k]||k)+'</span><span class="st-v">'+esc(fv(v))+'</span></div>';
      h+='</div>';
    }} else if(evt.changes.length){{
      evt.changes.forEach(ch=>{{h+='<div class="ch-row"><span class="ch-f">'+(fldLabels[ch.field]||ch.field)+'</span><span class="ch-old">'+esc(fv(ch.old))+'</span><span class="ch-arr">→</span><span class="ch-new">'+esc(fv(ch.new))+'</span></div>';}});
    }}
    h+='</div></div>';
  }});
  h+='</div><div class="foot">filtr: bns_flag = &quot;Y&quot;</div>';
  detailContent.innerHTML=h;
}}

searchInput.addEventListener('input',renderList);
document.querySelectorAll('.flt-btn').forEach(b=>b.addEventListener('click',()=>{{
  document.querySelectorAll('.flt-btn').forEach(x=>x.classList.remove('active'));
  b.classList.add('active');activeFilter=b.dataset.filter;renderList();
}}));
renderList();

/* ====== TAB 3 — NOVE FORMATY ====== */
function renderFmtCharts(){{
  fmtChartsRendered=true;
  if(!fmtData.length){{document.getElementById('fmtKpiRow').innerHTML='<div style="color:var(--dim);padding:20px;text-align:center;">Žádné přechody na NF.</div>';return;}}
  const total=fmtData.reduce((s,m)=>s+m.count,0);
  const maxM=fmtData.reduce((a,b)=>b.count>a.count?b:a);
  const avg=(total/fmtData.length).toFixed(1);
  document.getElementById('fmtKpiRow').innerHTML=[
    {{label:'Celkem přechodů',value:total,color:'var(--purple)'}},
    {{label:'Měsíců s přechody',value:fmtData.length,color:'var(--accent)'}},
    {{label:'Ø za měsíc',value:avg,color:'var(--teal)'}},
    {{label:'Nejakt. měsíc',value:maxM.month,sub:maxM.count+' pobček',color:'var(--orange)'}},
  ].map(k=>'<div class="kpi" style="border-top-color:'+k.color+'"><div class="value" style="color:'+k.color+';font-size:'+(String(k.value).length>6?'1.1rem':'1.6rem')+'">'+k.value+'</div><div class="label">'+k.label+'</div>'+(k.sub?'<div class="delta neutral">'+k.sub+'</div>':'')+'</div>').join('');

  const labels=fmtData.map(m=>m.month);
  const counts=fmtData.map(m=>m.count);
  const cumulative=counts.reduce((acc,v)=>{{acc.push((acc.length?acc[acc.length-1]:0)+v);return acc;}},[]);
  const deltas=counts.map((v,i)=>i===0?0:v-counts[i-1]);

  new ApexCharts(document.querySelector('#chartFmtAdoption'),{{
    chart:{{type:'bar',height:300,fontFamily:'DM Sans,sans-serif',toolbar:{{show:true}},animations:{{enabled:true,easing:'easeinout',speed:400}}}},
    series:[{{name:'Přírůstek NF',type:'bar',data:counts}},{{name:'Kumulativní',type:'line',data:cumulative}}],
    colors:['#7c3aed','#0891b2'],
    stroke:{{width:[0,2.5],curve:'smooth'}},
    plotOptions:{{bar:{{columnWidth:'60%',borderRadius:3}}}},
    xaxis:{{categories:labels,labels:{{rotate:-45,style:{{fontSize:'10px'}}}}}},
    yaxis:[{{title:{{text:'Přírůstek',style:{{fontSize:'11px'}}}},min:0}},{{opposite:true,title:{{text:'Kumulativně',style:{{fontSize:'11px'}}}},min:0}}],
    dataLabels:{{enabled:fmtData.length<=18,enabledOnSeries:[0],style:{{fontSize:'10px',fontWeight:700}},background:{{enabled:true,borderRadius:2,padding:2,opacity:0.85,borderWidth:0}}}},
    tooltip:{{shared:true,intersect:false}},legend:{{position:'top',fontSize:'12px'}},
    grid:{{borderColor:'#e8eaf0',strokeDashArray:3}},
  }}).render();

  new ApexCharts(document.querySelector('#chartFmtDelta'),{{
    chart:{{type:'bar',height:220,fontFamily:'DM Sans,sans-serif',toolbar:{{show:false}},animations:{{enabled:true,easing:'easeinout',speed:400}}}},
    series:[{{name:'Δ NF přírůstek',data:deltas}}],
    plotOptions:{{bar:{{columnWidth:'60%',borderRadius:2,colors:{{ranges:[{{from:-9999,to:-1,color:'#dc2626'}},{{from:0,to:0,color:'#94a3b8'}},{{from:1,to:9999,color:'#059669'}}]}}}}}},
    xaxis:{{categories:labels,labels:{{rotate:-45,style:{{fontSize:'10px'}}}}}},
    yaxis:{{labels:{{style:{{fontSize:'11px'}},formatter:v=>(v>0?'+':'')+v}}}},
    dataLabels:{{enabled:fmtData.length<=18,formatter:v=>(v>0?'+':'')+v,style:{{fontSize:'10px',fontWeight:700}}}},
    annotations:{{yaxis:[{{y:0,borderColor:'#94a3b8',strokeDashArray:0,borderWidth:1}}]}},
    tooltip:{{y:{{formatter:v=>(v>0?'+':'')+v+' pobček'}}}},
    grid:{{borderColor:'#e8eaf0',strokeDashArray:3}},
  }}).render();

  const fmtTbody=document.getElementById('fmtTableBody');
  fmtData.slice().reverse().forEach((m,i)=>{{
    const origIdx=fmtData.length-1-i;
    const delta=origIdx===0?null:m.count-fmtData[origIdx-1].count;
    const cumul=cumulative[origIdx];
    const dText=delta===null?'—':(delta>0?'+'+delta:String(delta));
    const dClass=delta===null?'neutral':delta>0?'up':delta<0?'down':'neutral';
    const pillsHtml=m.branches.map(b=>
      '<span class="fmt-branch-pill"><span class="pcode">'+b.branch_code+'</span>'+esc(b.branch_name)+
      '<span class="pfmt">'+esc(b.format_new)+'</span>'+(b.nf_number&&b.nf_number!=='None'&&b.nf_number!==''?'<span class="pnum">#'+esc(b.nf_number)+'</span>':'')+'</span>'
    ).join('');
    const tr=document.createElement('tr');
    tr.className='fmt-row-toggle';
    tr.innerHTML='<td style="font-family:monospace;font-size:0.8rem;">'+m.month+'</td><td style="font-weight:700;color:var(--purple);">'+m.count+'</td><td><span class="delta '+dClass+'">'+dText+'</span></td><td>'+cumul+'</td><td style="text-align:left;color:var(--muted);font-size:0.75rem;">'+m.branches.slice(0,3).map(b=>'<span style="font-family:monospace;font-size:0.7rem;color:var(--accent);">'+b.branch_code+'</span>').join(' ')+(m.count>3?' <span style="color:var(--dim);">+'+(m.count-3)+' dalších</span>':'')+' <span class="expand-icon" id="icon-'+i+'">▶</span></td>';
    const trD=document.createElement('tr');
    trD.className='fmt-detail-row';trD.style.display='none';
    trD.innerHTML='<td colspan="5"><div class="fmt-detail-inner"><div class="fmt-branch-list">'+pillsHtml+'</div></div></td>';
    tr.addEventListener('click',()=>{{
      const open=trD.style.display!=='none';
      trD.style.display=open?'none':'table-row';
      const ic=document.getElementById('icon-'+i);
      if(ic) ic.classList.toggle('open',!open);
    }});
    fmtTbody.appendChild(tr);fmtTbody.appendChild(trD);
  }});
}}

if(document.getElementById('tabFormats').classList.contains('active')) renderFmtCharts();

/* ====== TAB 4 — UZAVRENE POBOCKY ====== */
const closedChartData={closed_chart_data_json};
function renderClosedCharts(){{
  closedChartsRendered=true;
  const KRAJ_COLORS={{'Praha':'#C0392B','Středočeský':'#D35400','Jihočeský':'#27AE60','Plzeňský':'#2471A3','Karlovarský':'#7D3C98','Ústecký':'#A93226','Liberecký':'#117A65','Královéhradecký':'#1F618D','Pardubický':'#BA4A00','Vysočina':'#1E8449','Jihomoravský':'#6C3483','Olomoucký':'#154360','Zlínský':'#0E6655','Moravskoslezský':'#744212','Neznámý':'#566573'}};
  const REG_PAL=['#0057b8','#7c3aed','#059669','#d97706','#dc2626','#0891b2','#9c4221','#374151','#065f46','#831843','#1e3a5f','#4d7c0f','#7f1d1d','#134e4a'];
  function mkBar(el,data,useKrajColors){{
    if(!data||!data.years||!data.years.length) {{ document.querySelector(el).innerHTML='<p style="color:var(--muted);padding:20px;">Žádná data</p>'; return; }}
    const cols=data.series.map((s,i)=>useKrajColors?(KRAJ_COLORS[s.name]||REG_PAL[i%REG_PAL.length]):REG_PAL[i%REG_PAL.length]);
    new ApexCharts(document.querySelector(el),{{
      chart:{{type:'bar',height:340,stacked:true,fontFamily:'DM Sans,sans-serif',toolbar:{{show:true}},animations:{{enabled:true,easing:'easeinout',speed:400}}}},
      series:data.series,colors:cols,
      xaxis:{{categories:data.years,labels:{{style:{{fontSize:'11px'}}}}}},
      yaxis:{{labels:{{style:{{fontSize:'11px'}},formatter:v=>Math.round(v)}},min:0,title:{{text:'Uzavřené pobočky',style:{{fontSize:'11px'}}}}}},
      plotOptions:{{bar:{{columnWidth:'55%',borderRadius:2,borderRadiusApplication:'end',borderRadiusWhenStacked:'last'}}}},
      dataLabels:{{enabled:true,style:{{fontSize:'9px',fontWeight:700}},formatter:v=>v>0?v:'',background:{{enabled:false}}}},
      tooltip:{{shared:true,intersect:false}},
      legend:{{position:'top',fontSize:'11px',markers:{{width:9,height:9,radius:2}}}},
      grid:{{borderColor:'#e8eaf0',strokeDashArray:3}},
    }}).render();
  }}
  mkBar('#chartClosedRegion',closedChartData.region,false);
  mkBar('#chartClosedKraj',closedChartData.kraj,true);
}}

const closedBranches={closed_branches_json};
document.getElementById('closedCountBadge').textContent=closedBranches.length+' uzavřených pobček';
const cTbody=document.getElementById('closedTableBody');
closedBranches.forEach(b=>{{
  const tr=document.createElement('tr');
  tr.innerHTML=
    '<td><span class="closed-date">'+b.close_date+'</span></td>'+
    '<td><span class="rc-code">'+b.branch_code+'</span></td>'+
    '<td style="font-weight:600;">'+esc(b.branch_name)+'</td>'+
    '<td>'+esc(b.city)+'</td>'+
    '<td>'+esc(b.region)+'</td>'+
    '<td style="color:var(--muted);font-size:0.75rem;">'+esc(b.branch_type)+'</td>';
  cTbody.appendChild(tr);
}});

/* ====== TAB 5 — CASHLESS ====== */
const cashlessTransitions={cashless_transitions_json};
function renderCashlessTab(){{
  cashlessChartsRendered=true;
  const last=cashlessTransitions[cashlessTransitions.length-1];
  const first=cashlessTransitions[0];
  document.getElementById('cashlessCountBadge').textContent=cashlessTransitions.length+' poboček přešlo na cashless';
  document.getElementById('cashlessKpiRow').innerHTML=[
    {{label:'Cashless pobočky',value:last?last.cashless_count:0,color:'var(--orange)'}},
    {{label:'Otevřených celkem',value:last?last.open_count:0,color:'var(--green)'}},
    {{label:'% cashless (aktuální)',value:last?(last.pct+'%'):'-',color:'var(--orange)'}},
    {{label:'1. cashless přechod',value:first?first.date:'-',color:'var(--dim)'}},
  ].map(k=>'<div class="kpi" style="border-top-color:'+k.color+'"><div class="value" style="color:'+k.color+';font-size:'+(String(k.value).length>6?'1.1rem':'1.6rem')+'">'+k.value+'</div><div class="label">'+k.label+'</div></div>').join('');

  if(cashlessTransitions.length>0){{
    new ApexCharts(document.querySelector('#chartCashlessPct'),{{
      chart:{{type:'area',height:280,fontFamily:'DM Sans,sans-serif',toolbar:{{show:true}},animations:{{enabled:true,easing:'easeinout',speed:500}}}},
      series:[{{name:'% cashless',data:cashlessTransitions.map(c=>c.pct)}}],
      colors:['#d97706'],
      fill:{{type:'gradient',gradient:{{shadeIntensity:1,opacityFrom:0.35,opacityTo:0.05,stops:[0,100]}}}},
      stroke:{{width:2.5,curve:'stepline'}},
      xaxis:{{categories:cashlessTransitions.map(c=>c.date),labels:{{rotate:-45,style:{{fontSize:'9px'}},formatter:v=>v.slice(0,7)}},tickAmount:Math.min(cashlessTransitions.length,24)}},
      yaxis:{{min:0,max:100,labels:{{formatter:v=>v+'%',style:{{fontSize:'11px'}}}},title:{{text:'% cashless',style:{{fontSize:'11px'}}}}}},
      dataLabels:{{enabled:false}},
      tooltip:{{x:{{formatter:(_,{{dataPointIndex}})=>cashlessTransitions[dataPointIndex]?.date}},y:{{formatter:v=>v+'% ('+cashlessTransitions[Math.min(dataPointIndex,cashlessTransitions.length-1)]?.cashless_count+'/'+cashlessTransitions[Math.min(dataPointIndex,cashlessTransitions.length-1)]?.open_count+')'}}}},
      markers:{{size:cashlessTransitions.length<=40?4:0}},
      grid:{{borderColor:'#e8eaf0',strokeDashArray:3}},
      annotations:{{yaxis:[{{y:50,borderColor:'#94a3b8',strokeDashArray:4,label:{{text:'50%',style:{{fontSize:'10px',color:'#94a3b8',background:'transparent'}}}}}}]}},
    }}).render();
  }}

  const tbody=document.getElementById('cashlessTableBody');
  cashlessTransitions.slice().reverse().forEach(c=>{{
    const pctColor=c.pct>=75?'var(--green)':c.pct>=50?'var(--orange)':'var(--red)';
    const tr=document.createElement('tr');
    tr.innerHTML=
      '<td><span class="closed-date">'+c.date+'</span></td>'+
      '<td><span class="rc-code">'+c.branch_code+'</span></td>'+
      '<td style="font-weight:600;">'+esc(c.branch_name)+'</td>'+
      '<td>'+esc(c.city)+'</td>'+
      '<td>'+esc(c.region)+'</td>'+
      '<td style="text-align:right;font-variant-numeric:tabular-nums;">'+c.cashless_count+'</td>'+
      '<td style="text-align:right;font-variant-numeric:tabular-nums;">'+c.open_count+'</td>'+
      '<td style="text-align:right;"><span style="font-weight:700;color:'+pctColor+';">'+c.pct+'%</span>'+
        '<div class="cl-pct-bar"><div class="cl-pct-fill" style="width:'+c.pct+'%;background:'+pctColor+';"></div></div></td>';
    tbody.appendChild(tr);
  }});
}}

/* ====== TAB 6 — TIMELINE ====== */
const timelineData={timeline_data_json};
function renderTimeline(){{
  timelineRendered=true;
  const MONTH_SHORT=['Led','Úno','Bře','Dub','Kvě','Čvn','Čvc','Srp','Zář','Říj','Lis','Pro'];
  const el=document.getElementById('timelineContent');
  const years=['2024','2025','2026'];
  let html='';
  years.forEach(yr=>{{
    const yData=timelineData[yr]||{{}};
    const hasAny=Object.keys(yData).length>0;
    html+='<div class="tl-year-block"><div class="tl-year-title">'+yr+'</div>';
    if(!hasAny){{ html+='<div style="color:var(--dim);font-size:0.8rem;padding:8px 0;">Žádné události</div>'; html+='</div>'; return; }}
    html+='<div class="tl-track"><div class="tl-spine">';
    for(let mo=1;mo<=12;mo++){{
      const mk=yr+'-'+String(mo).padStart(2,'0');
      const evts=yData[mk]||[];
      const hasClosed=evts.some(e=>e.category==='closed');
      const hasCashless=evts.some(e=>e.category==='cashless');
      const hasFormat=evts.some(e=>e.category==='format');
      let dotCls='tl-dot';
      if(evts.length>0){{
        const cats=new Set(evts.map(e=>e.category));
        if(cats.size>1) dotCls+=' has-multi';
        else if(hasClosed) dotCls+=' has-closed';
        else if(hasCashless) dotCls+=' has-cashless';
        else if(hasFormat) dotCls+=' has-nf';
        else dotCls+=' has-events';
      }}
      html+='<div class="tl-month-col"><div class="tl-dot-wrap"><div class="'+dotCls+'"></div></div>';
      html+='<div class="tl-month-lbl">'+MONTH_SHORT[mo-1]+'</div>';
      if(evts.length>0){{
        html+='<div class="tl-month-events">';
        const groups={{'closed':evts.filter(e=>e.category==='closed'),'cashless':evts.filter(e=>e.category==='cashless'),'format':evts.filter(e=>e.category==='format')}};
        Object.entries(groups).forEach(([cat,items])=>{{
          if(!items.length) return;
          const shown=items.slice(0,3);
          shown.forEach(e=>{{
            html+='<div class="tl-chip '+cat+'"><span class="tl-chip-code">'+e.branch_code+'</span></div>';
          }});
          if(items.length>3) html+='<div class="tl-more">+'+(items.length-3)+'</div>';
        }});
        html+='</div>';
      }} else {{ html+='<div class="tl-month-events"></div>'; }}
      html+='</div>';
    }}
    html+='</div></div></div>';
  }});
  el.innerHTML=html;
}}

/* ====== TAB 7 — POROVNANI (compare) ====== */
(function(){{
  /* Init date pickers from event data */
  const allDates=[];
  branches.forEach(b=>b.events.forEach(e=>allDates.push(e.date)));
  allDates.sort();
  const minDate=allDates[0]||'2000-01-01';
  const todayISO=new Date().toISOString().slice(0,10);
  document.getElementById('cmpDateA').min=minDate;
  document.getElementById('cmpDateA').max=todayISO;
  document.getElementById('cmpDateA').value=minDate;
  document.getElementById('cmpDateB').min=minDate;
  document.getElementById('cmpDateB').max=todayISO;
  document.getElementById('cmpDateB').value=todayISO;
}})();

const CMP_COLS=['branch_name','branch_type','branch_closed','cashless','format','branch_building_nf_sf','nf_number','address','city','region'];
const CMP_LABELS={{'branch_name':'Název','branch_type':'Typ','branch_closed':'Uzavřena','cashless':'Cashless','format':'Formát','branch_building_nf_sf':'NF/SF','nf_number':'NF číslo','address':'Adresa','city':'Město','region':'Region'}};

let _cmpRows=[], _cmpFilter='all';

function getStateAt(branch, dateISO){{
  /* Return state of branch at dateISO (last event on or before date) */
  const evts=branch.events.filter(e=>e.date<=dateISO);
  if(!evts.length) return null;
  return evts[evts.length-1].state;
}}

function runComparison(){{
  const dA=document.getElementById('cmpDateA').value;
  const dB=document.getElementById('cmpDateB').value;
  if(!dA||!dB) {{ alert('Vyber obě data'); return; }}
  const rows=[];
  const allCodes=new Set(branches.map(b=>b.code));
  branches.forEach(b=>{{
    const sA=getStateAt(b,dA);
    const sB=getStateAt(b,dB);
    if(!sA&&!sB) return;
    let kind='same';
    if(!sA) kind='added';
    else if(!sB) kind='removed';
    else {{
      const diff=CMP_COLS.some(c=>JSON.stringify(sA[c])!==JSON.stringify(sB[c]));
      if(diff) kind='changed';
    }}
    rows.push({{code:b.code,name:b.name,sA,sB,kind}});
  }});
  _cmpRows=rows;
  _cmpFilter='all';
  document.querySelectorAll('.cmp-filter-btn').forEach(b=>b.classList.toggle('active',b.dataset.f==='all'));
  renderCmpTable();
  /* Summary */
  const cnt={{'added':0,'removed':0,'changed':0,'same':0}};
  rows.forEach(r=>cnt[r.kind]++);
  document.getElementById('cmpStats').innerHTML=
    '<div class="cmp-stat added"><strong>'+cnt.added+'</strong> Nové pobočky</div>'+
    '<div class="cmp-stat removed"><strong>'+cnt.removed+'</strong> Zrušené pobočky</div>'+
    '<div class="cmp-stat changed"><strong>'+cnt.changed+'</strong> Změněné pobočky</div>'+
    '<div class="cmp-stat same"><strong>'+cnt.same+'</strong> Beze změny</div>';
  document.getElementById('cmpSummary').style.display='block';
  document.getElementById('cmpPlaceholder').style.display='none';
}}

function setCmpFilter(btn,f){{
  _cmpFilter=f;
  document.querySelectorAll('.cmp-filter-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  renderCmpTable();
}}

function fmtVal(v){{
  if(v===null||v===undefined) return '<span style="color:var(--dim)">—</span>';
  if(v===true) return '<span class="bool-true">✓</span>';
  if(v===false) return '<span class="bool-false">✗</span>';
  return esc(String(v));
}}

function renderCmpTable(){{
  const filtered=_cmpFilter==='all'?_cmpRows:_cmpRows.filter(r=>r.kind===_cmpFilter);
  const thead=document.getElementById('cmpThead');
  const tbody=document.getElementById('cmpTbody');
  thead.innerHTML='<tr><th>Kód</th><th>Status</th>'+CMP_COLS.map(c=>'<th>'+esc(CMP_LABELS[c]||c)+'</th>').join('')+'</tr>';
  tbody.innerHTML='';
  filtered.forEach(r=>{{
    const tr=document.createElement('tr');
    tr.className='cmp-row-'+r.kind;
    const badge=r.kind==='added'?'<span class="cmp-badge new">Nová</span>':
                 r.kind==='removed'?'<span class="cmp-badge del">Zrušena</span>':
                 r.kind==='changed'?'<span class="cmp-badge chg">Změna</span>':'';
    let cells='<td><span class="cmp-code">'+r.code+'</span></td><td>'+badge+'</td>';
    CMP_COLS.forEach(col=>{{
      const vA=r.sA?r.sA[col]:undefined;
      const vB=r.sB?r.sB[col]:undefined;
      const diff=r.kind!=='added'&&r.kind!=='removed'&&JSON.stringify(vA)!==JSON.stringify(vB);
      if(diff){{
        cells+='<td class="cmp-cell-changed"><div class="cmp-cell-val"><span class="cmp-val-old">'+fmtVal(vA)+'</span><span class="cmp-val-new">'+fmtVal(vB)+'</span></div></td>';
      }} else {{
        const v=r.sB?r.sB[col]:(r.sA?r.sA[col]:undefined);
        cells+='<td><span class="cmp-val-same">'+fmtVal(v)+'</span></td>';
      }}
    }});
    tr.innerHTML=cells;
    tbody.appendChild(tr);
  }});
  if(!filtered.length) tbody.innerHTML='<tr><td colspan="'+(CMP_COLS.length+2)+'" style="text-align:center;padding:40px;color:var(--dim);">Žádné pobočky pro zvolený filtr</td></tr>';
}}
</script>
</body>
</html>"""

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\U0001f4c4 Kombinovaný report uložen: {REPORT_FILE}")
