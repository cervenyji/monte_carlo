"""
Průměrný obchodní den — hodinová návštěvnost & kapacitní simulace
=================================================================
Načte stejné zdroje jako statický report interního ratingu a vygeneruje
HTML report s:
  • Vyhledávačem poboček (název / ID, JS)
  • Průměrným návštěvním dnem Po–Ne (hodinové grafy)
  • Skladbou návštěv (schůzky, walk-in, segmenty klientů)
  • Počtem obchodních pozic a bankéřů
  • Monte Carlo simulací kapacity: zvládají bankéři odbavit klienty?

Spuštění (ze složky internirating_report/):
    python navstevnost_report.py

Výstup: navstevnost_obchodni_den.html
"""

import os
import sys
import math
import warnings
import json

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# KONFIGURACE — upravte cesty pokud se liší
# =============================================================================

VISITS_PATH        = "../in/tables/VISITS_2025.csv"
KPIS_PATH          = "kpis_grouped_2026.pkl"
SPECIALISTE_PATH   = "export_specialiste.pkl"
OD_PATH            = "../vypocet_ir_2026/zdroje/report_od_pobocky_dbs_04_2026.xlsx"
PARTIES_PATH       = "parties_2026.pkl"

OUTPUT_FILE = "navstevnost_obchodni_den.html"

HOUR_FROM = 7
HOUR_TO   = 19   # včetně

# Počet Monte Carlo simulací na hodinu
MC_RUNS = 2000

# Servisní časy dle typu návštěvy (minuty): (průměr, std)
SERVICE_TIME = {
    "schuzka_online":   (45, 10),
    "schuzka_fyzicka":  (45, 10),
    "walkin_bezhot":    (18,  5),
    "walkin_hot":       (7,   3),
    "default":          (20,  6),
}

# Konfigurační konstanty pozic (z internirating_report.py)
OBCHODNI_POZICE = {
    "bankéř klientské péče - junior",
    "bankéř klientské péče - medior",
    "firemní bankéř - master",
    "firemní bankéř - medior",
    "firemní bankéř - senior",
    "hypoteční specialista - medior",
    "hypoteční specialista - senior",
    "hypoteční specialista vcb - medior",
    "hypoteční specialista vcb - senior",
    "investiční specialista - medior",
    "manaž. segm. erste premier - team leader s portfoliem",
    "osobní bankéř - junior",
    "osobní bankéř - master",
    "osobní bankéř - medior",
    "osobní bankéř - senior",
    "pobočkový specialista - hypo",
    "podpora firemních bankéřů",
    "pojišťovací specialista - medior",
    "premier bankéř - master",
    "premier bankéř - medior",
    "premier bankéř - senior",
    "privátní bankéř - medior",
    "privátní bankéř - senior",
    "privátní bankéř - wealth management",
    "remote firemní bankéř - medior",
    "remote premier bankéř - medior",
    "spec. pro firemní pojištění - senior",
}
BANKER_POZICE_EXACT = {"OSOBNI_BANKER_-_JUNIOR", "OSOBNI_BANKER_-_MEDIOR", "OSOBNI_BANKER_-_SENIOR"}

# =============================================================================
# NAČTENÍ DAT
# =============================================================================

def _load(label, path, loader, **kwargs):
    if not os.path.exists(path):
        print(f"  ⚠️  {label}: soubor nenalezen ({path}) — sekce bude vynechána")
        return None
    try:
        result = loader(path, **kwargs)
        print(f"  ✅ {label}: {len(result):,} řádků")
        return result
    except Exception as e:
        print(f"  ❌ {label}: chyba načtení — {e}")
        return None


def load_all():
    print("📂 Načítám datové zdroje...")

    visits = _load("VISITS_2025", VISITS_PATH, pd.read_csv, low_memory=False)

    kpis = _load("kpis_grouped_2026", KPIS_PATH,
                 lambda p: pd.read_pickle(p))

    spec = _load("export_specialiste", SPECIALISTE_PATH,
                 lambda p: pd.read_pickle(p))

    od = _load("oteviraci_doba", OD_PATH,
               lambda p: pd.read_excel(p, dtype=str))

    parties = _load("parties_2026", PARTIES_PATH,
                    lambda p: pd.read_pickle(p))

    return visits, kpis, spec, od, parties


# =============================================================================
# PŘÍPRAVA DAT
# =============================================================================

def prepare_visits(visits: pd.DataFrame) -> pd.DataFrame:
    v = visits.copy()
    v.columns = [c.strip().upper() for c in v.columns]

    id_col = next((c for c in ["BRANCH_ID", "BRANCH_CODE", "POBOCKA"] if c in v.columns), None)
    if id_col is None:
        raise ValueError(f"Visits: nenalezen sloupec s ID pobočky. Dostupné: {list(v.columns)}")
    v["BRANCH_CODE"] = pd.to_numeric(v[id_col], errors="coerce")

    if "VISIT_DATE" in v.columns:
        v["_DT"]      = pd.to_datetime(v["VISIT_DATE"], errors="coerce")
        v["_WEEKDAY"] = v["_DT"].dt.weekday   # 0=Po … 6=Ne
        v["_DATE"]    = v["_DT"].dt.date
        v["_MONTH"]   = v["_DT"].dt.month
    else:
        raise ValueError("Visits: nenalezen sloupec VISIT_DATE")

    if "VISIT_TIME" in v.columns:
        v["_HOUR"] = pd.to_numeric(
            v["VISIT_TIME"].astype(str).str.split(":").str[0], errors="coerce"
        )
    elif "VISIT_HOUR" in v.columns:
        v["_HOUR"] = pd.to_numeric(v["VISIT_HOUR"], errors="coerce")
    else:
        v["_HOUR"] = None

    return v.dropna(subset=["BRANCH_CODE"])


def prepare_kpis(kpis: pd.DataFrame) -> pd.DataFrame:
    k = kpis.copy()
    k.columns = [c.strip().upper() for c in k.columns]
    id_col = next((c for c in ["POBOCKA_ID", "BRANCH_CODE", "BRANCH_ID"] if c in k.columns), None)
    if id_col:
        k["BRANCH_CODE"] = pd.to_numeric(k[id_col], errors="coerce")
    return k


def prepare_specialiste(spec: pd.DataFrame):
    s = spec.copy()
    _renames = {
        "BRANCH_ID": "branch_id", "BRANCH_NAME": "branch_name",
        "GPS_X": "gps_x", "GPS_Y": "gps_y", "EVIDENCNI_STAV": "evidencni_stav",
        "Evidenční stav": "evidencni_stav",
    }
    s.rename(columns={k: v for k, v in _renames.items() if k in s.columns}, inplace=True)
    s["branch_id"] = pd.to_numeric(s.get("branch_id", pd.Series(dtype=float)), errors="coerce")

    id_cols = {"branch_id", "branch_name", "gps_x", "gps_y", "evidencni_stav"}
    pozice_cols = [c for c in s.columns if c not in id_cols]

    for c in pozice_cols:
        s[c] = pd.to_numeric(s[c], errors="coerce").fillna(0)

    # Celkové FTE
    s["_total_spec"] = s[pozice_cols].sum(axis=1)

    # Bankéři (osobní bankéř dle BANKER_POZICE_EXACT)
    bnk_cols = [c for c in pozice_cols if c.upper() in BANKER_POZICE_EXACT]
    s["BANKERS_COUNT"] = s[bnk_cols].sum(axis=1) if bnk_cols else 0

    # Obchodní FTE
    obch_cols = [c for c in pozice_cols if c.lower().strip() in OBCHODNI_POZICE]
    s["OBCHODNI_FTE"] = s[obch_cols].sum(axis=1) if obch_cols else 0

    # Detailní pozice pro zobrazení (top 10 obsazených)
    s["_POZICE_DETAIL"] = s.apply(
        lambda r: {c: r[c] for c in pozice_cols if r[c] > 0}, axis=1
    )

    return s, pozice_cols


def prepare_od(od: pd.DataFrame) -> dict:
    """Vrátí dict branch_code → dict s detailem otevírací doby."""
    od.columns = [c.strip().upper() for c in od.columns]
    id_col = next((c for c in ["KOD_POBOCKY", "BRANCH_CODE", "POBOCKA_ID"] if c in od.columns), None)
    if id_col is None:
        return {}
    od["_BC"] = pd.to_numeric(od[id_col], errors="coerce")
    result = {}
    for _, row in od.dropna(subset=["_BC"]).iterrows():
        result[int(row["_BC"])] = row.to_dict()
    return result


def prepare_parties(parties: pd.DataFrame) -> pd.DataFrame:
    p = parties.copy()
    p.columns = [c.strip().upper() for c in p.columns]
    id_col = next((c for c in ["DBS_HOME_BRANCH_CODE", "BRANCH_CODE", "BRANCH_ID"] if c in p.columns), None)
    if id_col:
        p["BRANCH_CODE"] = pd.to_numeric(p[id_col], errors="coerce")
    return p


# =============================================================================
# VÝPOČTY
# =============================================================================

def avg_by_hour(df_branch: pd.DataFrame, weekdays, hours) -> pd.Series:
    sub = df_branch[df_branch["_WEEKDAY"].isin(weekdays)].copy()
    if sub.empty or sub["_HOUR"].isna().all():
        return pd.Series(0.0, index=hours)
    n_days = max(sub["_DATE"].nunique(), 1)
    return sub.groupby("_HOUR").size().reindex(hours, fill_value=0) / n_days


def visit_type_mix(df_branch: pd.DataFrame, kpi_row: pd.Series | None) -> dict:
    """
    Odhadne podíly typů návštěv.
    Priorita: 1) ATTENDANCE_TYPE ze visits  2) KPI sloupce  3) fallback síťové průměry.
    """
    result = {
        "schuzka_online":  0.0,
        "schuzka_fyzicka": 0.0,
        "walkin_bezhot":   0.0,
        "walkin_hot":      0.0,
    }

    # 1) ATTENDANCE_TYPE přímo z visits
    if "ATTENDANCE_TYPE" in df_branch.columns and df_branch["ATTENDANCE_TYPE"].notna().any():
        counts = df_branch["ATTENDANCE_TYPE"].value_counts()
        total = counts.sum()
        for typ, val in counts.items():
            t = str(typ).lower()
            if "online" in t or "digi" in t:
                result["schuzka_online"] += val
            elif "schůzka" in t or "schuzka" in t or "meeting" in t:
                result["schuzka_fyzicka"] += val
            elif "hot" in t or "cash" in t or "pokladna" in t:
                result["walkin_hot"] += val
            else:
                result["walkin_bezhot"] += val
        s = sum(result.values())
        if s > 0:
            return {k: v / s for k, v in result.items()}

    # 2) KPI sloupce
    if kpi_row is not None:
        def _kpi(col):
            return float(kpi_row.get(col, 0) or 0)
        total_kpi = (
            _kpi("POCET_SCHUZEK_ONLINE") + _kpi("POCET_SCHUZEK_FYZICKY") +
            _kpi("POCET_BEZHOT_WALK_IN") + _kpi("POCET_HOT_WALK_IN")
        )
        if total_kpi > 0:
            return {
                "schuzka_online":  _kpi("POCET_SCHUZEK_ONLINE")  / total_kpi,
                "schuzka_fyzicka": _kpi("POCET_SCHUZEK_FYZICKY") / total_kpi,
                "walkin_bezhot":   _kpi("POCET_BEZHOT_WALK_IN")  / total_kpi,
                "walkin_hot":      _kpi("POCET_HOT_WALK_IN")     / total_kpi,
            }

    # 3) Síťové průměry (fallback)
    return {"schuzka_online": 0.12, "schuzka_fyzicka": 0.23,
            "walkin_bezhot": 0.47, "walkin_hot": 0.18}


def avg_service_time_min(mix: dict) -> float:
    """Vážený průměr servisního času v minutách."""
    return sum(mix[k] * SERVICE_TIME[k][0] for k in mix)


# =============================================================================
# MONTE CARLO SIMULACE
# =============================================================================

def monte_carlo_hour(avg_arrivals: float, n_bankers: int, mix: dict, n_runs: int = MC_RUNS) -> dict:
    """
    Pro jednu hodinu simuluje M/M/c frontu (Poisson arrivals, log-normal service).
    Vrátí statistiky přes n_runs simulací.

    Metoda:
    - Příchody: Poisson(λ = avg_arrivals)
    - Servisní čas každého příchozího: mix dle visit_type_mix → log-normal(μ, σ)
    - c = n_bankers (paralelní servery)
    - Sledujeme: čekající ve frontě, čekací dobu, využití bankéřů
    """
    if avg_arrivals <= 0 or n_bankers <= 0:
        return {
            "p_overload": 0.0, "p_wait_15": 0.0, "avg_queue": 0.0,
            "avg_wait_min": 0.0, "avg_util_pct": 0.0, "avg_arrivals": avg_arrivals,
        }

    rng = np.random.default_rng(seed=42)

    type_keys  = list(mix.keys())
    type_probs = np.array([mix[k] for k in type_keys])
    svc_mean   = np.array([SERVICE_TIME[k][0] for k in type_keys])
    svc_std    = np.array([SERVICE_TIME[k][1] for k in type_keys])

    # Log-normal params (μ, σ) z (mean, std) normálního rozdělení
    def _lognorm_params(mean, std):
        var = std ** 2
        mu  = np.log(mean ** 2 / np.sqrt(mean ** 2 + var))
        sig = np.sqrt(np.log(1 + var / mean ** 2))
        return mu, sig

    lnorm_params = [_lognorm_params(m, s) for m, s in zip(svc_mean, svc_std)]

    overload_count = 0
    wait_15_count  = 0
    total_queue    = 0.0
    total_wait     = 0.0
    total_util     = 0.0

    HOUR_MIN = 60.0  # délka hodiny v minutách

    for _ in range(n_runs):
        n_arr = rng.poisson(avg_arrivals)
        if n_arr == 0:
            continue

        # Příchody: rovnoměrně v průběhu hodiny
        arrival_times = np.sort(rng.uniform(0, HOUR_MIN, n_arr))

        # Typ a servisní čas každého klienta
        types = rng.choice(len(type_keys), size=n_arr, p=type_probs)
        svc_times = np.array([
            max(1.0, rng.lognormal(lnorm_params[t][0], lnorm_params[t][1]))
            for t in types
        ])

        # Simulace c-serverové fronty (discrete event)
        # free_at[i] = čas kdy i-tý bankéř bude volný
        free_at = np.zeros(n_bankers)
        waits = np.zeros(n_arr)

        for j in range(n_arr):
            t_arr = arrival_times[j]
            # Nejdříve volný bankéř
            best = int(np.argmin(free_at))
            start = max(t_arr, free_at[best])
            waits[j] = start - t_arr
            free_at[best] = start + svc_times[j]

        # Statistiky pro tento run
        # Fronta: klienti čekající déle než 0 min
        queue_len = np.sum(waits > 0)
        avg_wait  = float(np.mean(waits))
        util      = min(1.0, float(np.mean(svc_times) * n_arr / (HOUR_MIN * n_bankers)))

        if queue_len > n_bankers:  # více čekajících než bankéřů = přetížení
            overload_count += 1
        if np.any(waits > 15):
            wait_15_count += 1

        total_queue += queue_len
        total_wait  += avg_wait
        total_util  += util

    denom = max(n_runs, 1)
    return {
        "p_overload":    overload_count / denom,
        "p_wait_15":     wait_15_count  / denom,
        "avg_queue":     total_queue    / denom,
        "avg_wait_min":  total_wait     / denom,
        "avg_util_pct":  total_util     / denom * 100,
        "avg_arrivals":  avg_arrivals,
    }


def run_monte_carlo(avg_day: pd.Series, n_bankers: int, mix: dict) -> dict:
    """Spustí MC pro každou hodinu. Vrátí dict hour → stats."""
    results = {}
    for h, lam in avg_day.items():
        results[int(h)] = monte_carlo_hour(float(lam), n_bankers, mix)
    return results


# =============================================================================
# HTML KOMPONENTY
# =============================================================================

WEEKDAY_NAMES  = ["Po", "Út", "St", "Čt", "Pá", "So", "Ne"]
MONTH_NAMES    = ["Led","Úno","Bře","Dub","Kvě","Čvn","Čvc","Srp","Zář","Říj","Lis","Pro"]
VISIT_TYPE_LABELS = {
    "schuzka_online":  "Schůzky online",
    "schuzka_fyzicka": "Schůzky fyzické",
    "walkin_bezhot":   "Walk-in bezhotovostní",
    "walkin_hot":      "Walk-in hotovostní",
}
VISIT_TYPE_COLORS = {
    "schuzka_online":  "#2770f0",
    "schuzka_fyzicka": "#45b065",
    "walkin_bezhot":   "#e07020",
    "walkin_hot":      "#9b6bbf",
}


def _bar(values: pd.Series, labels, colors=None, height=70, fmt="{:.1f}") -> str:
    vmax = max(float(values.max()), 0.001)
    default_color = "#2770f0"
    bars = ""
    for i, (label, val) in enumerate(zip(labels, values)):
        pct  = float(val) / vmax * 100
        col  = (colors[i] if colors else None) or default_color
        peak = (float(val) == float(values.max()))
        fw   = "700" if peak else "400"
        bars += f"""<div style="display:flex;flex-direction:column;align-items:center;gap:2px;flex:1;min-width:0;">
  <div style="font-size:0.6rem;color:#555;font-weight:{fw};">{fmt.format(float(val))}</div>
  <div style="width:100%;background:#eef0f4;border-radius:3px 3px 0 0;height:{height}px;display:flex;align-items:flex-end;">
    <div style="width:100%;height:{pct:.1f}%;background:{col};border-radius:3px 3px 0 0;"></div>
  </div>
  <div style="font-size:0.6rem;color:#888;white-space:nowrap;">{label}</div>
</div>"""
    return f'<div style="display:flex;gap:2px;">{bars}</div>'


def _heatmap(df_branch: pd.DataFrame, hours, weekdays=range(5)) -> str:
    work = df_branch[df_branch["_WEEKDAY"].isin(weekdays)]
    if work.empty or work["_HOUR"].isna().all():
        return "<span style='color:#ccc;font-size:0.75rem;'>Chybí data o čase</span>"
    hm = work.groupby(["_WEEKDAY", "_HOUR"]).size().unstack(fill_value=0)
    hm = hm.reindex(index=list(weekdays), columns=hours, fill_value=0)
    day_counts = work.groupby("_WEEKDAY")["_DATE"].nunique()
    for wd in weekdays:
        cnt = day_counts.get(wd, 1)
        if cnt > 0 and wd in hm.index:
            hm.loc[wd] = hm.loc[wd] / cnt
    vmax = max(float(hm.values.max()), 1)

    header = "<tr><th></th>" + "".join(
        f"<th style='font-size:0.58rem;color:#aaa;padding:1px 2px;text-align:center;'>{h}</th>"
        for h in hours
    ) + "</tr>"
    rows = ""
    for wd in weekdays:
        row = f"<tr><td style='font-size:0.62rem;color:#666;font-weight:600;padding:1px 4px;'>{WEEKDAY_NAMES[wd]}</td>"
        for h in hours:
            val = hm.at[wd, h] if (wd in hm.index and h in hm.columns) else 0
            alpha = max(0.06, val / vmax)
            row += f"<td style='background:rgba(39,112,240,{alpha:.2f});width:17px;height:13px;border-radius:2px;'></td>"
        row += "</tr>"
        rows += row
    return f"<table style='border-collapse:separate;border-spacing:2px;'>{header}{rows}</table>"


def _donut(mix: dict) -> str:
    """SVG koláčový graf skladby návštěv."""
    items = [(VISIT_TYPE_LABELS[k], mix[k], VISIT_TYPE_COLORS[k]) for k in mix]
    items.sort(key=lambda x: -x[1])
    cx, cy, r = 55, 55, 42
    start = -math.pi / 2
    paths = ""
    for label, pct, color in items:
        if pct <= 0:
            continue
        angle = pct * 2 * math.pi
        end   = start + angle
        large = 1 if angle > math.pi else 0
        x1, y1 = cx + r * math.cos(start), cy + r * math.sin(start)
        x2, y2 = cx + r * math.cos(end),   cy + r * math.sin(end)
        ri = 22
        ix1, iy1 = cx + ri * math.cos(start), cy + ri * math.sin(start)
        ix2, iy2 = cx + ri * math.cos(end),   cy + ri * math.sin(end)
        paths += (
            f'<path d="M{ix1:.1f},{iy1:.1f} L{x1:.1f},{y1:.1f} '
            f'A{r},{r} 0 {large},1 {x2:.1f},{y2:.1f} '
            f'L{ix2:.1f},{iy2:.1f} A{ri},{ri} 0 {large},0 {ix1:.1f},{iy1:.1f}Z" '
            f'fill="{color}" />'
        )
        start = end

    legend = "".join(
        f'<div style="display:flex;align-items:center;gap:5px;margin-bottom:3px;">'
        f'<div style="width:10px;height:10px;border-radius:2px;background:{color};flex-shrink:0;"></div>'
        f'<span style="font-size:0.65rem;color:#555;">{label} <b>{pct*100:.0f}%</b></span></div>'
        for label, pct, color in items if pct > 0
    )
    return (
        f'<div style="display:flex;align-items:center;gap:12px;">'
        f'<svg width="110" height="110" viewBox="0 0 110 110">{paths}</svg>'
        f'<div>{legend}</div></div>'
    )


def _mc_capacity_chart(mc_results: dict, hours) -> str:
    """Bar chart kapacitní simulace — barva dle využití."""
    bars = ""
    for h in hours:
        if h not in mc_results:
            continue
        r = mc_results[h]
        util = r["avg_util_pct"]
        p_ov = r["p_overload"]
        # Barva: zelená → žlutá → červená dle využití
        if util < 50:
            col = "#45b065"
        elif util < 75:
            col = "#e8a020"
        elif util < 90:
            col = "#e07020"
        else:
            col = "#d62728"
        pct_h = max(util, 2)
        ov_badge = (
            f'<div style="font-size:0.55rem;color:#d62728;font-weight:700;">{p_ov*100:.0f}%⚠</div>'
            if p_ov > 0.15 else ""
        )
        wait  = r["avg_wait_min"]
        title = f"Využití: {util:.0f}% | Čekání: {wait:.1f} min | P(přetížení): {p_ov*100:.0f}%"
        bars += f"""<div style="display:flex;flex-direction:column;align-items:center;gap:2px;flex:1;min-width:0;" title="{title}">
  <div style="font-size:0.58rem;color:#555;">{util:.0f}%</div>
  {ov_badge}
  <div style="width:100%;background:#eef0f4;border-radius:3px 3px 0 0;height:70px;display:flex;align-items:flex-end;">
    <div style="width:100%;height:{pct_h:.1f}%;background:{col};border-radius:3px 3px 0 0;"></div>
  </div>
  <div style="font-size:0.6rem;color:#888;">{h}</div>
</div>"""
    return f'<div style="display:flex;gap:2px;">{bars}</div>'


def _kpi_badges(kpi_row) -> str:
    if kpi_row is None:
        return ""
    def _fmt(col, label, icon):
        val = kpi_row.get(col)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return ""
        return (
            f'<div style="background:#f5f7fa;border-radius:6px;padding:6px 10px;text-align:center;">'
            f'<div style="font-size:0.65rem;color:#aaa;font-weight:600;">{icon} {label}</div>'
            f'<div style="font-size:1rem;font-weight:800;color:#1a2340;">{int(val):,}</div></div>'
        )
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px;">'
        + _fmt("POCET_NAVSTEV_CELKEM",  "Návštěv celkem",   "🚶")
        + _fmt("POCET_SCHUZEK_ONLINE",  "Schůzky online",   "💻")
        + _fmt("POCET_SCHUZEK_FYZICKY", "Schůzky fyzické",  "🤝")
        + _fmt("POCET_BEZHOT_WALK_IN",  "Walk-in bez hotov.","💳")
        + _fmt("POCET_HOT_WALK_IN",     "Walk-in hotov.",    "💵")
        + _fmt("NR_NEW_ARRIVALS",       "Noví klienti",      "🆕")
        + "</div>"
    )


def _positions_html(spec_row) -> str:
    if spec_row is None:
        return ""
    detail = spec_row.get("_POZICE_DETAIL", {})
    if not detail:
        return "<span style='color:#bbb;font-size:0.75rem;'>Data o pozicích nejsou k dispozici</span>"
    items = sorted(detail.items(), key=lambda x: -x[1])
    rows = "".join(
        f'<tr><td style="padding:3px 8px;font-size:0.75rem;color:#444;">{pos}</td>'
        f'<td style="padding:3px 8px;font-size:0.75rem;font-weight:700;text-align:right;">{int(cnt)}</td></tr>'
        for pos, cnt in items
    )
    bankers = int(spec_row.get("BANKERS_COUNT", 0))
    obch    = int(spec_row.get("OBCHODNI_FTE",  0))
    total   = int(spec_row.get("_total_spec",   0))
    summary = (
        f'<div style="display:flex;gap:16px;margin-bottom:10px;">'
        f'<div><span style="font-size:0.65rem;color:#aaa;">Bankéři (OB)</span>'
        f'<div style="font-size:1.1rem;font-weight:800;color:#1a2340;">{bankers}</div></div>'
        f'<div><span style="font-size:0.65rem;color:#aaa;">Obchodní FTE</span>'
        f'<div style="font-size:1.1rem;font-weight:800;color:#1a2340;">{obch}</div></div>'
        f'<div><span style="font-size:0.65rem;color:#aaa;">Celkem spec.</span>'
        f'<div style="font-size:1.1rem;font-weight:800;color:#1a2340;">{total}</div></div>'
        f'</div>'
    )
    return summary + f'<table style="border-collapse:collapse;">{rows}</table>'


def _od_html(od_row: dict | None) -> str:
    if not od_row:
        return "<span style='color:#bbb;font-size:0.75rem;'>Otevírací doba není k dispozici</span>"
    days = [
        ("PONDELI", "Po"), ("UTERY", "Út"), ("STREDA", "St"),
        ("CTVRTEK", "Čt"), ("PATEK", "Pá"), ("SOBOTA", "So"), ("NEDELE", "Ne"),
    ]
    rows = ""
    for key, label in days:
        od_from = str(od_row.get(f"{key}_OD", "") or "").strip()
        od_to   = str(od_row.get(f"{key}_DO", "") or "").strip()
        # Alternativní sloupce
        if not od_from:
            od_from = str(od_row.get(f"{key}_DOP._OD", "") or "").strip()
        if not od_to:
            od_to = str(od_row.get(f"{key}_DOP._DO", "") or "").strip()
        if od_from in ("", "nan", "00:00", "0:00") and od_to in ("", "nan", "00:00", "0:00"):
            time_str = '<span style="color:#ccc;">zavřeno</span>'
        else:
            time_str = f'<b>{od_from}</b> – <b>{od_to}</b>'
        rows += (
            f'<tr><td style="padding:3px 8px;font-size:0.75rem;color:#666;font-weight:600;">{label}</td>'
            f'<td style="padding:3px 8px;font-size:0.75rem;">{time_str}</td></tr>'
        )
    return f'<table style="border-collapse:collapse;">{rows}</table>'


# =============================================================================
# RENDEROVÁNÍ POBOČKOVÉ KARTY
# =============================================================================

def render_branch_card(
    branch_code: int,
    branch_name: str,
    df_b: pd.DataFrame,
    kpi_row,
    spec_row,
    od_row: dict | None,
    n_clients: int,
) -> str:
    hours = list(range(HOUR_FROM, HOUR_TO + 1))

    # Průměrné dny dle skupiny dní
    avg_work   = avg_by_hour(df_b, range(5), hours)   # Po–Pá
    avg_sat    = avg_by_hour(df_b, [5],      hours)   # So
    avg_sun    = avg_by_hour(df_b, [6],      hours)   # Ne
    avg_all    = avg_by_hour(df_b, range(7), hours)   # Po–Ne

    # Složení návštěv
    mix        = visit_type_mix(df_b, kpi_row)
    svc_avg    = avg_service_time_min(mix)

    # Počet bankéřů (pro MC)
    n_bankers  = int(spec_row["BANKERS_COUNT"]) if spec_row is not None else 2
    n_bankers  = max(n_bankers, 1)

    # Monte Carlo — pouze pracovní dny
    print(f"    MC simulace: pobočka {branch_code} ({n_bankers} bankéřů)…", end="\r")
    mc_results = run_monte_carlo(avg_work, n_bankers, mix)

    # Celkové statistiky
    total_visits  = len(df_b[df_b["_WEEKDAY"].between(0, 4)])
    n_work_days   = max(df_b[df_b["_WEEKDAY"].between(0, 4)]["_DATE"].nunique(), 1)
    avg_per_day   = total_visits / n_work_days
    peak_hour_val = int(avg_work.idxmax()) if avg_work.sum() > 0 else None
    peak_avg_val  = float(avg_work.max())

    # Kapacitní hodnocení
    avg_util_peak = mc_results.get(peak_hour_val, {}).get("avg_util_pct", 0) if peak_hour_val else 0
    if avg_util_peak < 50:
        cap_badge = '<span style="background:#45b065;color:#fff;border-radius:4px;padding:2px 8px;font-size:0.72rem;">V pohodě</span>'
    elif avg_util_peak < 75:
        cap_badge = '<span style="background:#e8a020;color:#fff;border-radius:4px;padding:2px 8px;font-size:0.72rem;">Mírné zatížení</span>'
    elif avg_util_peak < 90:
        cap_badge = '<span style="background:#e07020;color:#fff;border-radius:4px;padding:2px 8px;font-size:0.72rem;">Vysoké zatížení</span>'
    else:
        cap_badge = '<span style="background:#d62728;color:#fff;border-radius:4px;padding:2px 8px;font-size:0.72rem;">⚠ Přetížení!</span>'

    peak_badge = (
        f'<span style="background:#2770f0;color:#fff;border-radius:4px;padding:2px 8px;font-size:0.72rem;font-weight:700;">{peak_hour_val}:00</span>'
        if peak_hour_val is not None else "—"
    )

    # Segment mix ze visits
    seg_html = ""
    if "CLIENT_SEGMENT" in df_b.columns and df_b["CLIENT_SEGMENT"].notna().any():
        by_seg = df_b["CLIENT_SEGMENT"].value_counts().head(6)
        total_seg = by_seg.sum()
        seg_bars = "".join(
            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">'
            f'<span style="font-size:0.7rem;font-weight:600;color:#444;min-width:60px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="{sn}">{sn}</span>'
            f'<div style="flex:1;background:#f0f0f0;border-radius:4px;height:8px;">'
            f'<div style="width:{sc/total_seg*100:.1f}%;background:#2770f0;height:100%;border-radius:4px;"></div></div>'
            f'<span style="font-size:0.68rem;color:#888;">{sc/total_seg*100:.0f}%</span></div>'
            for sn, sc in by_seg.items()
        )
        seg_html = f'<div style="margin-bottom:6px;"><div style="font-size:0.7rem;font-weight:700;color:#444;margin-bottom:4px;">Segmenty klientů</div>{seg_bars}</div>'

    # Generování heatmap a grafů
    hm_work  = _heatmap(df_b, hours, range(5))
    hm_wkend = _heatmap(df_b, hours, [5, 6])

    hour_colors_mc = []
    for h in hours:
        util = mc_results.get(h, {}).get("avg_util_pct", 0)
        if util < 50:   hour_colors_mc.append("#45b065")
        elif util < 75: hour_colors_mc.append("#e8a020")
        elif util < 90: hour_colors_mc.append("#e07020")
        else:           hour_colors_mc.append("#d62728")

    # MC tabulka
    mc_rows = "".join(
        f'<tr style="border-bottom:1px solid #f5f5f5;">'
        f'<td style="padding:4px 8px;font-size:0.72rem;font-weight:700;">{h}:00</td>'
        f'<td style="padding:4px 8px;font-size:0.72rem;">{mc_results.get(h,{}).get("avg_arrivals",0):.1f}</td>'
        f'<td style="padding:4px 8px;font-size:0.72rem;">{mc_results.get(h,{}).get("avg_util_pct",0):.0f}%</td>'
        f'<td style="padding:4px 8px;font-size:0.72rem;">{mc_results.get(h,{}).get("avg_wait_min",0):.1f} min</td>'
        f'<td style="padding:4px 8px;font-size:0.72rem;color:{"#d62728" if mc_results.get(h,{}).get("p_overload",0)>0.2 else "#333"};">'
        f'{mc_results.get(h,{}).get("p_overload",0)*100:.0f}%</td></tr>'
        for h in hours
    )

    clients_str = f"{n_clients:,}" if n_clients > 0 else "—"

    return f"""
<div class="branch-card" data-code="{branch_code}" data-name="{branch_name.lower()}"
     id="branch-{branch_code}"
     style="background:#fff;border:1px solid #e0e4ea;border-radius:10px;
            padding:20px 24px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,.06);">

  <!-- HLAVIČKA -->
  <div style="display:flex;align-items:flex-start;justify-content:space-between;
              flex-wrap:wrap;gap:10px;margin-bottom:16px;">
    <div>
      <span style="font-size:0.68rem;color:#aaa;font-weight:600;text-transform:uppercase;">
        Pobočka {branch_code}
      </span>
      <div style="font-size:1.1rem;font-weight:800;color:#1a2340;">{branch_name}</div>
    </div>
    <div style="display:flex;gap:12px;flex-wrap:wrap;">
      <div style="text-align:center;"><div style="font-size:0.62rem;color:#aaa;font-weight:600;">Prac. návštěvy</div>
        <div style="font-size:1rem;font-weight:800;">{total_visits:,}</div></div>
      <div style="text-align:center;"><div style="font-size:0.62rem;color:#aaa;font-weight:600;">Prům./den</div>
        <div style="font-size:1rem;font-weight:800;">{avg_per_day:.1f}</div></div>
      <div style="text-align:center;"><div style="font-size:0.62rem;color:#aaa;font-weight:600;">Klientů</div>
        <div style="font-size:1rem;font-weight:800;">{clients_str}</div></div>
      <div style="text-align:center;"><div style="font-size:0.62rem;color:#aaa;font-weight:600;">Bankéři</div>
        <div style="font-size:1rem;font-weight:800;">{n_bankers}</div></div>
      <div style="text-align:center;"><div style="font-size:0.62rem;color:#aaa;font-weight:600;">Špičková hod.</div>
        <div style="margin-top:2px;">{peak_badge}</div></div>
      <div style="text-align:center;"><div style="font-size:0.62rem;color:#aaa;font-weight:600;">Kapacita</div>
        <div style="margin-top:2px;">{cap_badge}</div></div>
    </div>
  </div>

  <!-- KPI BADGES -->
  {_kpi_badges(kpi_row)}

  <!-- GRID: HODINOVÝ PROVOZ + HEATMAPA -->
  <div style="margin-bottom:14px;">
    <div style="font-size:0.7rem;font-weight:700;color:#444;text-transform:uppercase;letter-spacing:.04em;margin-bottom:6px;">
      Průměrný počet návštěv dle hodiny — pracovní dny (Po–Pá)
    </div>
    {_bar(avg_work, [str(h) for h in hours], colors=hour_colors_mc, height=70)}
    <div style="font-size:0.65rem;color:#aaa;margin-top:3px;">
      Barva = kapacitní vytížení bankéřů dle Monte Carlo (🟢 &lt;50% · 🟡 50–75% · 🟠 75–90% · 🔴 &gt;90%)
    </div>
  </div>

  <!-- 3-SLOUPEC: den v týdnu, heatmapa prac. dny, heatmapa víkend -->
  <div style="display:grid;grid-template-columns:1fr auto auto;gap:20px;margin-bottom:14px;align-items:start;">
    <div>
      <div style="font-size:0.7rem;font-weight:700;color:#444;text-transform:uppercase;margin-bottom:6px;">Průměr dle dne v týdnu</div>
      {_bar(avg_by_hour(df_b, range(7), range(7)).reindex(range(7), fill_value=0), WEEKDAY_NAMES, height=50, fmt="{:.0f}")}
    </div>
    <div>
      <div style="font-size:0.7rem;font-weight:700;color:#444;text-transform:uppercase;margin-bottom:6px;">Heatmapa Po–Pá</div>
      {hm_work}
    </div>
    <div>
      <div style="font-size:0.7rem;font-weight:700;color:#444;text-transform:uppercase;margin-bottom:6px;">Heatmapa So–Ne</div>
      {hm_wkend}
    </div>
  </div>

  <!-- 2-SLOUPEC: skladba návštěv, segmenty -->
  <div style="display:grid;grid-template-columns:auto 1fr;gap:20px;margin-bottom:14px;align-items:start;">
    <div>
      <div style="font-size:0.7rem;font-weight:700;color:#444;text-transform:uppercase;margin-bottom:6px;">Skladba návštěv</div>
      {_donut(mix)}
      <div style="font-size:0.65rem;color:#888;margin-top:4px;">Ø servisní čas: {svc_avg:.0f} min</div>
    </div>
    <div>
      {seg_html}
      <div style="font-size:0.7rem;font-weight:700;color:#444;text-transform:uppercase;margin-bottom:6px;">Obsazení pozic</div>
      {_positions_html(spec_row)}
    </div>
  </div>

  <!-- MONTE CARLO DETAIL -->
  <details style="margin-top:10px;">
    <summary style="font-size:0.75rem;font-weight:700;color:#2770f0;cursor:pointer;user-select:none;">
      📊 Monte Carlo simulace kapacity — detail po hodinách ({n_bankers} bankéřů, {MC_RUNS} běhů/hodinu)
    </summary>
    <div style="margin-top:10px;">
      <div style="margin-bottom:10px;">
        {_mc_capacity_chart(mc_results, hours)}
      </div>
      <table style="width:100%;border-collapse:collapse;font-size:0.75rem;">
        <thead><tr style="background:#f5f7fa;font-size:0.68rem;text-transform:uppercase;color:#888;">
          <th style="padding:5px 8px;text-align:left;">Hodina</th>
          <th style="padding:5px 8px;text-align:left;">Ø příchozí</th>
          <th style="padding:5px 8px;text-align:left;">Využití</th>
          <th style="padding:5px 8px;text-align:left;">Čekání</th>
          <th style="padding:5px 8px;text-align:left;">P(přetíž.)</th>
        </tr></thead>
        <tbody>{mc_rows}</tbody>
      </table>
      <div style="font-size:0.65rem;color:#aaa;margin-top:6px;">
        Metodika: Poissonovy příchody λ=ø návštěv/hod · Servisní časy log-normální dle skladby
        (online schůzka 45±10 min · fyzická 45±10 · bez-hot. walk-in 18±5 · hot. walk-in 7±3) ·
        M/M/c fronta · P(přetíž.) = podíl simulací kde čeká více klientů než je bankéřů.
      </div>
    </div>
  </details>

  <!-- OTEVÍRACÍ DOBA -->
  <details style="margin-top:6px;">
    <summary style="font-size:0.75rem;font-weight:700;color:#555;cursor:pointer;user-select:none;">
      🕐 Otevírací doba
    </summary>
    <div style="margin-top:8px;">{_od_html(od_row)}</div>
  </details>

</div>"""


# =============================================================================
# CELÝ HTML REPORT
# =============================================================================

def render_full_report(cards_html: str, summary_rows: list, n_branches: int) -> str:
    summary_table_rows = "".join(
        f"""<tr style="border-bottom:1px solid #f0f0f0;cursor:pointer;"
               onclick="document.getElementById('branch-{r['bc']}').scrollIntoView({{behavior:'smooth'}})">
          <td style="padding:6px 10px;font-weight:600;">{r['bc']}</td>
          <td style="padding:6px 10px;">{r['name']}</td>
          <td style="padding:6px 10px;text-align:right;">{r['total']:,}</td>
          <td style="padding:6px 10px;text-align:right;">{r['clients']:,}</td>
          <td style="padding:6px 10px;text-align:right;">{r['avg']:.1f}</td>
          <td style="padding:6px 10px;text-align:center;">
            <span style="background:#2770f0;color:#fff;border-radius:4px;padding:1px 7px;font-size:0.75rem;">{r['peak']}:00</span>
          </td>
          <td style="padding:6px 10px;text-align:right;">{r['bankers']}</td>
          <td style="padding:6px 10px;">{r['cap_badge']}</td>
        </tr>"""
        for r in summary_rows
    )

    return f"""<!DOCTYPE html>
<html lang="cs">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>Průměrný obchodní den — hodinová návštěvnost & kapacita poboček 2025</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0;}}
    body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
          background:#f4f6fb;color:#222;line-height:1.45;}}
    .page{{max-width:1140px;margin:0 auto;padding:32px 18px;}}
    h1{{font-size:1.55rem;font-weight:800;color:#1a2340;margin-bottom:4px;}}
    h2{{font-size:1.05rem;font-weight:700;color:#1a2340;margin:28px 0 10px;}}
    .subtitle{{font-size:0.82rem;color:#888;margin-bottom:24px;}}
    .box{{background:#fff;border:1px solid #e0e4ea;border-radius:10px;
          padding:18px 22px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,.05);}}
    #search-bar{{width:100%;padding:10px 14px;font-size:0.95rem;border:1px solid #d0d4db;
                 border-radius:8px;outline:none;margin-bottom:6px;}}
    #search-bar:focus{{border-color:#2770f0;box-shadow:0 0 0 3px rgba(39,112,240,.12);}}
    .hidden{{display:none!important;}}
    details>summary{{list-style:none;}}
    details>summary::-webkit-details-marker{{display:none;}}
  </style>
</head>
<body>
<div class="page">

  <h1>🚶 Průměrný obchodní den — hodinová návštěvnost & kapacita poboček</h1>
  <div class="subtitle">
    Zdroj: VISITS_2025.csv · kpis_grouped_2026.pkl · export_specialiste.pkl · report_od_pobocky_dbs_04_2026.xlsx &nbsp;|&nbsp;
    Pobočky: {n_branches} &nbsp;|&nbsp;
    Hodiny: {HOUR_FROM}–{HOUR_TO} hod
  </div>

  <!-- VYHLEDÁVAČ -->
  <div class="box">
    <div style="font-size:0.8rem;font-weight:700;color:#444;margin-bottom:8px;">🔍 Vyhledat pobočku</div>
    <input id="search-bar" type="text" placeholder="Název pobočky nebo ID (např. Praha, 101, Brno)…" autocomplete="off">
    <div id="search-hint" style="font-size:0.72rem;color:#aaa;">Zobrazeno: <span id="visible-count">{n_branches}</span> / {n_branches} poboček</div>
  </div>

  <!-- PŘEHLEDOVÁ TABULKA -->
  <h2>📋 Přehled poboček</h2>
  <div class="box" style="overflow-x:auto;">
    <table style="width:100%;border-collapse:collapse;font-size:0.8rem;" id="summary-table">
      <thead>
        <tr style="background:#f5f7fa;font-size:0.68rem;text-transform:uppercase;color:#888;">
          <th style="padding:7px 10px;text-align:left;">Kód</th>
          <th style="padding:7px 10px;text-align:left;">Název</th>
          <th style="padding:7px 10px;text-align:right;">Návštěvy</th>
          <th style="padding:7px 10px;text-align:right;">Klientů</th>
          <th style="padding:7px 10px;text-align:right;">Ø/den</th>
          <th style="padding:7px 10px;text-align:center;">Špička</th>
          <th style="padding:7px 10px;text-align:right;">Bankéři</th>
          <th style="padding:7px 10px;text-align:left;">Kapacita</th>
        </tr>
      </thead>
      <tbody id="summary-body">
        {summary_table_rows}
      </tbody>
    </table>
  </div>

  <!-- DETAIL POBOČEK -->
  <h2>🏢 Detail poboček</h2>
  <div id="branches-container">
    {cards_html}
  </div>

</div>

<script>
(function() {{
  const input    = document.getElementById('search-bar');
  const cards    = document.querySelectorAll('.branch-card');
  const counter  = document.getElementById('visible-count');
  const sumRows  = document.querySelectorAll('#summary-body tr');
  const total    = cards.length;

  input.addEventListener('input', function() {{
    const q = this.value.trim().toLowerCase();
    let vis = 0;
    cards.forEach(function(card, i) {{
      const name = card.dataset.name || '';
      const code = card.dataset.code || '';
      const match = !q || name.includes(q) || code.includes(q);
      card.classList.toggle('hidden', !match);
      if (sumRows[i]) sumRows[i].classList.toggle('hidden', !match);
      if (match) vis++;
    }});
    counter.textContent = vis;
  }});
}});
</script>
</body>
</html>"""


# =============================================================================
# HLAVNÍ FUNKCE
# =============================================================================

def main():
    visits_raw, kpis_raw, spec_raw, od_raw, parties_raw = load_all()

    if visits_raw is None:
        print("❌ Bez visits dat nelze pokračovat.")
        sys.exit(1)

    print("\n⚙️  Příprava dat...")
    visits = prepare_visits(visits_raw)

    kpis_df   = prepare_kpis(kpis_raw)   if kpis_raw   is not None else None
    spec_df, _ = prepare_specialiste(spec_raw) if spec_raw is not None else (None, [])
    od_detail  = prepare_od(od_raw)      if od_raw     is not None else {}
    parties_df = prepare_parties(parties_raw) if parties_raw is not None else None

    # Počty klientů per pobočka
    client_counts: dict = {}
    if parties_df is not None and "BRANCH_CODE" in parties_df.columns:
        client_counts = (
            parties_df.groupby("BRANCH_CODE").size().to_dict()
        )

    # Název pobočky z visits nebo specialiste
    name_map: dict = {}
    if spec_df is not None and "branch_id" in spec_df.columns and "branch_name" in spec_df.columns:
        name_map = spec_df.dropna(subset=["branch_id"]).set_index(
            spec_df["branch_id"].astype(int)
        )["branch_name"].to_dict()
    for bc, grp in visits.groupby("BRANCH_CODE"):
        if int(bc) not in name_map:
            cands = grp["_NAME"].dropna() if "_NAME" in grp.columns else pd.Series(dtype=str)
            name_map[int(bc)] = cands.mode().iloc[0] if not cands.empty and not cands.mode().empty else str(int(bc))

    branches = sorted(visits["BRANCH_CODE"].dropna().unique().astype(int))
    print(f"   Počet poboček s visits daty: {len(branches)}")

    print("\n📊 Generuji karty poboček (Monte Carlo)...")
    cards_html   = ""
    summary_rows = []

    hours = list(range(HOUR_FROM, HOUR_TO + 1))

    for idx, bc in enumerate(branches, 1):
        df_b = visits[visits["BRANCH_CODE"] == bc].copy()
        name = name_map.get(bc, str(bc))

        # KPI řádek
        kpi_row = None
        if kpis_df is not None and "BRANCH_CODE" in kpis_df.columns:
            matches = kpis_df[kpis_df["BRANCH_CODE"] == bc]
            kpi_row = matches.iloc[0].to_dict() if not matches.empty else None

        # Specialiste řádek
        spec_row = None
        if spec_df is not None and "branch_id" in spec_df.columns:
            matches = spec_df[spec_df["branch_id"] == bc]
            spec_row = matches.iloc[0].to_dict() if not matches.empty else None

        od_row    = od_detail.get(bc)
        n_clients = int(client_counts.get(bc, 0))

        # Render karty
        cards_html += render_branch_card(bc, name, df_b, kpi_row, spec_row, od_row, n_clients)

        # Summary řádek
        avg_work = avg_by_hour(df_b, range(5), hours)
        peak_h   = int(avg_work.idxmax()) if avg_work.sum() > 0 else 0
        n_bankers = int(spec_row["BANKERS_COUNT"]) if spec_row else 2
        n_bankers = max(n_bankers, 1)
        mc_peak  = monte_carlo_hour(float(avg_work.get(peak_h, 0)), n_bankers,
                                    visit_type_mix(df_b, kpi_row), n_runs=500)
        util_peak = mc_peak["avg_util_pct"]
        if util_peak < 50:   cap_badge = '<span style="background:#45b065;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.7rem;">V pohodě</span>'
        elif util_peak < 75: cap_badge = '<span style="background:#e8a020;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.7rem;">Mírné</span>'
        elif util_peak < 90: cap_badge = '<span style="background:#e07020;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.7rem;">Vysoké</span>'
        else:                cap_badge = '<span style="background:#d62728;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.7rem;">⚠ Přetížení</span>'

        total_v = len(df_b[df_b["_WEEKDAY"].between(0, 4)])
        n_days  = max(df_b[df_b["_WEEKDAY"].between(0, 4)]["_DATE"].nunique(), 1)

        summary_rows.append({
            "bc": bc, "name": name, "total": total_v,
            "clients": n_clients, "avg": total_v / n_days,
            "peak": peak_h, "bankers": n_bankers,
            "cap_badge": cap_badge,
        })

        if idx % 20 == 0 or idx == len(branches):
            print(f"   {idx}/{len(branches)} poboček hotovo        ")

    print("\n✍️  Sestavuji HTML...")
    html = render_full_report(cards_html, summary_rows, len(branches))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Report uložen: {OUTPUT_FILE}  ({os.path.getsize(OUTPUT_FILE)//1024} kB)")


if __name__ == "__main__":
    main()
