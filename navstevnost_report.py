"""
Průměrný obchodní den — hodinová návštěvnost & kapacitní simulace
=================================================================
Spuštění (ze složky internirating_report/):
    python navstevnost_report.py

Výstup: navstevnost_obchodni_den.html
"""

import os
import sys
import math
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# KONFIGURACE
# =============================================================================

VISITS_PATH      = "../in/tables/VISITS_2025.csv"
KPIS_PATH        = "kpis_grouped_2026.pkl"
SPECIALISTE_PATH = "export_specialiste.pkl"
OD_PATH          = "../vypocet_ir_2026/zdroje/report_od_pobocky_dbs_04_2026.xlsx"
PARTIES_PATH     = "parties_2026.pkl"

OUTPUT_FILE = "navstevnost_obchodni_den.html"

HOUR_FROM = 7
HOUR_TO   = 19   # včetně
MC_RUNS   = 2000
WORKING_DAYS_PER_YEAR = 250

# Pozice — jen osobní bankéři obsluhují klienty
OB_EXACT = {"OSOBNI_BANKER_-_JUNIOR", "OSOBNI_BANKER_-_MEDIOR", "OSOBNI_BANKER_-_SENIOR"}
OB_JUNIOR_KEYS = {"OSOBNI_BANKER_-_JUNIOR", "osobní bankéř - junior", "osobni_banker_-_junior"}
BKP_KEYS = {"bankéř klientské péče - medior", "bankir_klientske_pece_-_medior",
             "banker_klientske_pece_-_medior"}

OBCHODNI_POZICE = {
    "bankéř klientské péče - junior", "bankéř klientské péče - medior",
    "firemní bankéř - master", "firemní bankéř - medior", "firemní bankéř - senior",
    "hypoteční specialista - medior", "hypoteční specialista - senior",
    "hypoteční specialista vcb - medior", "hypoteční specialista vcb - senior",
    "investiční specialista - medior",
    "manaž. segm. erste premier - team leader s portfoliem",
    "osobní bankéř - junior", "osobní bankéř - master",
    "osobní bankéř - medior", "osobní bankéř - senior",
    "pobočkový specialista - hypo", "podpora firemních bankéřů",
    "pojišťovací specialista - medior",
    "premier bankéř - master", "premier bankéř - medior", "premier bankéř - senior",
    "privátní bankéř - medior", "privátní bankéř - senior",
    "privátní bankéř - wealth management",
    "remote firemní bankéř - medior", "remote premier bankéř - medior",
    "spec. pro firemní pojištění - senior",
}

WEEKDAY_NAMES = ["Po", "Út", "St", "Čt", "Pá", "So", "Ne"]
MONTH_NAMES   = ["Led","Úno","Bře","Dub","Kvě","Čvn","Čvc","Srp","Zář","Říj","Lis","Pro"]

# =============================================================================
# NAČTENÍ DAT
# =============================================================================

def _try_load(label, path, loader):
    if not os.path.exists(path):
        print(f"  ⚠️  {label}: nenalezen ({path})")
        return None
    try:
        df = loader(path)
        print(f"  ✅ {label}: {len(df):,} řádků")
        return df
    except Exception as e:
        print(f"  ❌ {label}: {e}")
        return None


def load_all():
    print("📂 Načítám datové zdroje...")
    visits   = _try_load("VISITS_2025",        VISITS_PATH,      lambda p: pd.read_csv(p, low_memory=False))
    kpis     = _try_load("kpis_grouped_2026",  KPIS_PATH,        pd.read_pickle)
    spec     = _try_load("export_specialiste", SPECIALISTE_PATH, pd.read_pickle)
    od       = _try_load("oteviraci_doba",     OD_PATH,          lambda p: pd.read_excel(p, dtype=str))
    parties  = _try_load("parties_2026",       PARTIES_PATH,     pd.read_pickle)
    return visits, kpis, spec, od, parties


# =============================================================================
# PŘÍPRAVA DAT
# =============================================================================

def prep_visits(raw: pd.DataFrame) -> pd.DataFrame:
    v = raw.copy()
    v.columns = [c.strip().upper() for c in v.columns]
    id_col = next((c for c in ["BRANCH_ID", "BRANCH_CODE", "POBOCKA"] if c in v.columns), None)
    if not id_col:
        raise ValueError(f"Visits: nenalezen ID sloupec. Dostupné: {list(v.columns)}")
    v["BRANCH_CODE"] = pd.to_numeric(v[id_col], errors="coerce")
    v["_DT"]      = pd.to_datetime(v["VISIT_DATE"], errors="coerce")
    v["_WEEKDAY"] = v["_DT"].dt.weekday
    v["_DATE"]    = v["_DT"].dt.date
    v["_MONTH"]   = v["_DT"].dt.month
    if "VISIT_TIME" in v.columns:
        v["_HOUR"] = pd.to_numeric(
            v["VISIT_TIME"].astype(str).str.split(":").str[0], errors="coerce")
    else:
        v["_HOUR"] = None
    return v.dropna(subset=["BRANCH_CODE", "_DT"])


def prep_kpis(raw: pd.DataFrame) -> pd.DataFrame:
    k = raw.copy()
    k.columns = [c.strip().upper() for c in k.columns]
    id_col = next((c for c in ["POBOCKA_ID", "BRANCH_CODE", "BRANCH_ID"] if c in k.columns), None)
    if id_col:
        k["BRANCH_CODE"] = pd.to_numeric(k[id_col], errors="coerce")
    return k


def prep_spec(raw: pd.DataFrame):
    """Vrátí (df, ob_col_list, bkp_col_list, ob_junior_col_list)."""
    s = raw.copy()
    renames = {"BRANCH_ID": "branch_id", "BRANCH_NAME": "branch_name",
               "GPS_X": "gps_x", "GPS_Y": "gps_y", "EVIDENCNI_STAV": "evidencni_stav"}
    s.rename(columns={k: v for k, v in renames.items() if k in s.columns}, inplace=True)
    s["branch_id"] = pd.to_numeric(s.get("branch_id", pd.Series(dtype=float)), errors="coerce")

    id_cols  = {"branch_id", "branch_name", "gps_x", "gps_y", "evidencni_stav"}
    poz_cols = [c for c in s.columns if c not in id_cols]

    for c in poz_cols:
        s[c] = pd.to_numeric(s[c], errors="coerce").fillna(0)

    # OB bankéři (junior/medior/senior) — jediní, kdo obsluhují klienty
    ob_cols = [c for c in poz_cols if c.upper() in OB_EXACT]
    # OB junior zvlášť (odbavuje servis pokud není BKP)
    ob_jr_cols = [c for c in poz_cols
                  if c.upper() in OB_JUNIOR_KEYS or c.lower().strip() == "osobní bankéř - junior"]
    # BKP medior — primárně obsluhuje servisní návštěvy
    bkp_cols = [c for c in poz_cols
                if c.lower().strip() in BKP_KEYS
                or "klientsk" in c.lower() and "medior" in c.lower()]

    s["OB_COUNT"]  = s[ob_cols].sum(axis=1)  if ob_cols  else 0
    s["BKP_COUNT"] = s[bkp_cols].sum(axis=1) if bkp_cols else 0
    s["OB_JR_COUNT"] = s[ob_jr_cols].sum(axis=1) if ob_jr_cols else 0

    # Obchodní FTE a celkový počet
    obch_cols = [c for c in poz_cols if c.lower().strip() in OBCHODNI_POZICE]
    s["OBCHODNI_FTE"] = s[obch_cols].sum(axis=1) if obch_cols else 0
    s["_total_spec"]  = s[poz_cols].sum(axis=1)

    # Detail pozic pro zobrazení
    s["_POZICE_DETAIL"] = s.apply(
        lambda r: {c: r[c] for c in poz_cols if r[c] > 0}, axis=1)

    return s, poz_cols


def prep_od(raw: pd.DataFrame) -> dict:
    od = raw.copy()
    od.columns = [c.strip().upper() for c in od.columns]
    id_col = next((c for c in ["KOD_POBOCKY", "BRANCH_CODE", "POBOCKA_ID"] if c in od.columns), None)
    if not id_col:
        return {}
    od["_BC"] = pd.to_numeric(od[id_col], errors="coerce")
    return {int(r["_BC"]): r.to_dict() for _, r in od.dropna(subset=["_BC"]).iterrows()}


def prep_parties(raw: pd.DataFrame) -> pd.DataFrame:
    p = raw.copy()
    p.columns = [c.strip().upper() for c in p.columns]
    id_col = next((c for c in ["DBS_HOME_BRANCH_CODE", "BRANCH_CODE"] if c in p.columns), None)
    if id_col:
        p["BRANCH_CODE"] = pd.to_numeric(p[id_col], errors="coerce")
    return p


# =============================================================================
# VISIT MIX — bez hotovostních walk-in
# =============================================================================

def visit_mix_no_cash(df_b: pd.DataFrame, kpi_row: dict | None) -> dict:
    """
    Vrátí podíly tří typů: schuzka_online, schuzka_fyzicka, servis (bez-hot. walk-in).
    Hotovostní walk-in se VYLUČUJE.
    Priorita: 1) ATTENDANCE_TYPE ve visits  2) KPI sloupce  3) síťový fallback
    """
    result = {"schuzka_online": 0.0, "schuzka_fyzicka": 0.0, "servis": 0.0}

    if "ATTENDANCE_TYPE" in df_b.columns and df_b["ATTENDANCE_TYPE"].notna().any():
        raw_cnt = df_b["ATTENDANCE_TYPE"].value_counts()
        for typ, val in raw_cnt.items():
            t = str(typ).lower()
            if "hot" in t or "cash" in t or "pokladna" in t:
                pass  # ignoruj hotovostní
            elif "online" in t or "digi" in t:
                result["schuzka_online"] += val
            elif "schůzka" in t or "schuzka" in t or "meeting" in t or "fyzick" in t:
                result["schuzka_fyzicka"] += val
            else:
                result["servis"] += val
        s = sum(result.values())
        if s > 0:
            return {k: v / s for k, v in result.items()}

    if kpi_row:
        def g(col): return float(kpi_row.get(col, 0) or 0)
        total = g("POCET_SCHUZEK_ONLINE") + g("POCET_SCHUZEK_FYZICKY") + g("POCET_BEZHOT_WALK_IN")
        if total > 0:
            return {
                "schuzka_online":  g("POCET_SCHUZEK_ONLINE")  / total,
                "schuzka_fyzicka": g("POCET_SCHUZEK_FYZICKY") / total,
                "servis":          g("POCET_BEZHOT_WALK_IN")  / total,
            }

    return {"schuzka_online": 0.13, "schuzka_fyzicka": 0.27, "servis": 0.60}


# =============================================================================
# AGREGACE NÁVŠTĚV
# =============================================================================

def avg_by_hour(df_b: pd.DataFrame, weekdays, hours) -> pd.Series:
    sub = df_b[df_b["_WEEKDAY"].isin(weekdays)]
    if sub.empty or sub["_HOUR"].isna().all():
        return pd.Series(0.0, index=hours)
    n_days = max(sub["_DATE"].nunique(), 1)
    return sub.groupby("_HOUR").size().reindex(hours, fill_value=0) / n_days


def avg_by_weekday(df_b: pd.DataFrame) -> pd.Series:
    if df_b.empty:
        return pd.Series(0.0, index=range(7))
    daily = df_b.groupby(["_DATE", "_WEEKDAY"]).size().reset_index(name="cnt")
    return daily.groupby("_WEEKDAY")["cnt"].mean().reindex(range(7), fill_value=0)


# =============================================================================
# MONTE CARLO — DVOU-FRONTOVÝ MODEL
# =============================================================================
# Pravidla:
#   • Schůzka online    → OB fronta, bankéř LOCKED na 45 min (nelze přerušit walk-inem)
#   • Schůzka fyzická   → OB fronta, log-norm(45, 10)
#   • Servisní návštěva (bez-hot. walk-in):
#       – 70 %: ≤ 15 min (Uniform 5–15)
#       – 20 %: 30 min (Uniform 20–35)
#       – 10 %: 15 min + eskalace → OB fronta 45 min meeting
#     → odbavuje: BKP medior (pokud existuje), jinak OB junior
#   • Hotovostní walk-in: IGNORUJE SE
# =============================================================================

def _lognormal_params(mean, std):
    var = std ** 2
    mu  = np.log(mean ** 2 / np.sqrt(mean ** 2 + var))
    sig = np.sqrt(np.log(1 + var / mean ** 2))
    return mu, sig

_LN_MTG = _lognormal_params(45, 10)  # fyzická schůzka


def simulate_hour(avg_arr: float, ob_count: int, bkp_count: int, ob_jr_count: int,
                  mix: dict, rng: np.random.Generator) -> dict:
    """
    Simuluje jednu hodinu. Vrátí statistiky.
    bkp_count:   počet BKP medior (primárně odbavují servis)
    ob_jr_count: počet OB junior (záloha pro servis, pokud není BKP)
    ob_count:    CELKOVÝ počet OB (junior+medior+senior)
    """
    HOUR_MIN = 60.0
    zero = {"util_ob": 0.0, "util_bkp": 0.0,
            "avg_wait_ob": 0.0, "avg_wait_bkp": 0.0,
            "p_overload_ob": 0.0, "p_wait15_ob": 0.0,
            "p_overload_bkp": 0.0, "p_wait15_bkp": 0.0,
            "avg_arr": avg_arr}

    if avg_arr <= 0 or ob_count <= 0:
        return zero

    p_on  = mix["schuzka_online"]
    p_fy  = mix["schuzka_fyzicka"]
    # p_srv = mix["servis"] (zbytek)

    has_bkp = bkp_count > 0

    util_ob_acc    = 0.0
    util_bkp_acc   = 0.0
    wait_ob_list   = []
    wait_bkp_list  = []
    overload_ob_c  = 0
    overload_bkp_c = 0

    for _ in range(MC_RUNS):
        n = rng.poisson(avg_arr)
        if n == 0:
            continue

        arrivals = np.sort(rng.uniform(0, HOUR_MIN, n))

        ob_free  = np.zeros(ob_count)
        bkp_free = np.zeros(max(bkp_count, 1))  # vždy aspoň 1 slot (OB junior)

        ob_busy_min  = 0.0
        bkp_busy_min = 0.0

        for t in arrivals:
            r = rng.random()

            if r < p_on:
                # Online schůzka → OB, locked 45 min
                i = int(np.argmin(ob_free))
                start = max(t, ob_free[i])
                wait_ob_list.append(start - t)
                svc = 45.0 + rng.normal(0, 2)
                ob_free[i] = start + max(svc, 30)
                ob_busy_min += max(svc, 30)

            elif r < p_on + p_fy:
                # Fyzická schůzka → OB
                i = int(np.argmin(ob_free))
                start = max(t, ob_free[i])
                wait_ob_list.append(start - t)
                svc = max(10.0, rng.lognormal(_LN_MTG[0], _LN_MTG[1]))
                ob_free[i] = start + svc
                ob_busy_min += svc

            else:
                # Servisní návštěva
                r2 = rng.random()
                if r2 < 0.70:
                    svc_bkp = rng.uniform(5, 15)
                    escalate = False
                elif r2 < 0.90:
                    svc_bkp = rng.uniform(20, 35)
                    escalate = False
                else:
                    svc_bkp = 15.0
                    escalate = True

                if has_bkp:
                    # BKP fronta
                    i = int(np.argmin(bkp_free))
                    start = max(t, bkp_free[i])
                    wait_bkp_list.append(start - t)
                    bkp_free[i] = start + svc_bkp
                    bkp_busy_min += svc_bkp
                    if escalate:
                        # Po BKP servisní části → OB meeting 45 min
                        t_esc = bkp_free[i]
                        j = int(np.argmin(ob_free))
                        ob_start = max(t_esc, ob_free[j])
                        ob_free[j] = ob_start + 45
                        ob_busy_min += 45
                        wait_ob_list.append(ob_start - t_esc)
                else:
                    # Bez BKP: OB junior (nebo nejvolnější OB) odbavuje servis
                    # Zkusíme prioritizovat OB junior; pokud nemáme inf, použijeme nejvolnější OB
                    i = int(np.argmin(ob_free))
                    start = max(t, ob_free[i])
                    wait_ob_list.append(start - t)
                    ob_free[i] = start + svc_bkp
                    ob_busy_min += svc_bkp
                    if escalate:
                        t_esc = ob_free[i]
                        j = int(np.argmin(ob_free))
                        ob_start = max(t_esc, ob_free[j])
                        ob_free[j] = ob_start + 45
                        ob_busy_min += 45
                        wait_ob_list.append(ob_start - t_esc)

        # Utilization za tento run
        util_ob_acc  += min(1.0, ob_busy_min  / (HOUR_MIN * ob_count))
        util_bkp_acc += min(1.0, bkp_busy_min / (HOUR_MIN * max(bkp_count, 1)))

        # Přetížení = někdo čeká déle než je dostupný čas zbývající v hodině
        if ob_free.max() > HOUR_MIN * 1.2:
            overload_ob_c += 1
        if bkp_count > 0 and bkp_free.max() > HOUR_MIN * 1.2:
            overload_bkp_c += 1

    d = MC_RUNS
    w_ob  = np.array(wait_ob_list)
    w_bkp = np.array(wait_bkp_list)

    return {
        "avg_arr":         avg_arr,
        "util_ob":         util_ob_acc / d * 100,
        "util_bkp":        util_bkp_acc / d * 100 if has_bkp else None,
        "avg_wait_ob":     float(w_ob.mean())  if len(w_ob)  else 0.0,
        "avg_wait_bkp":    float(w_bkp.mean()) if len(w_bkp) else 0.0,
        "p_overload_ob":   overload_ob_c  / d,
        "p_overload_bkp":  overload_bkp_c / d if has_bkp else None,
        "p_wait15_ob":     float(np.mean(w_ob  > 15)) if len(w_ob)  else 0.0,
        "p_wait15_bkp":    float(np.mean(w_bkp > 15)) if len(w_bkp) else 0.0,
    }


def run_mc_for_hours(avg_day: pd.Series, ob_count: int, bkp_count: int,
                     ob_jr_count: int, mix: dict) -> dict:
    rng = np.random.default_rng(seed=42)
    return {
        int(h): simulate_hour(float(lam), ob_count, bkp_count, ob_jr_count, mix, rng)
        for h, lam in avg_day.items()
    }


# =============================================================================
# MODEL 2 — škálování dle počtu klientů
# =============================================================================

def model2_hours(avg_work: pd.Series, n_clients: int, network_vpc_day: float) -> pd.Series:
    """
    Škáluje hodinové průměry z reálných dat na očekávané hodnoty dle počtu klientů.
    network_vpc_day = síťový průměr: návštěvy/klient/pracovní den
    """
    if n_clients <= 0 or avg_work.sum() == 0 or network_vpc_day <= 0:
        return avg_work.copy()
    expected_daily = n_clients * network_vpc_day
    actual_daily   = float(avg_work.sum())
    scale = expected_daily / actual_daily
    return avg_work * scale


# =============================================================================
# HTML KOMPONENTY
# =============================================================================

def _color_util(util: float) -> str:
    if util < 50:   return "#45b065"
    elif util < 70: return "#e8c020"
    elif util < 85: return "#e07020"
    else:           return "#d62728"


def _bar_chart(values: pd.Series, labels, colors=None, height=70, fmt="{:.1f}") -> str:
    vmax = max(float(values.max()), 0.001)
    bars = ""
    for i, (lbl, val) in enumerate(zip(labels, values)):
        pct = float(val) / vmax * 100
        col = (colors[i] if colors else None) or "#2770f0"
        fw  = "700" if float(val) == float(values.max()) else "400"
        bars += (
            f'<div style="display:flex;flex-direction:column;align-items:center;gap:1px;flex:1;min-width:0;">'
            f'<div style="font-size:0.58rem;color:#555;font-weight:{fw};">{fmt.format(float(val))}</div>'
            f'<div style="width:100%;background:#eef0f4;border-radius:3px 3px 0 0;height:{height}px;'
            f'display:flex;align-items:flex-end;">'
            f'<div style="width:100%;height:{pct:.1f}%;background:{col};border-radius:3px 3px 0 0;"></div>'
            f'</div>'
            f'<div style="font-size:0.58rem;color:#888;white-space:nowrap;">{lbl}</div></div>'
        )
    return f'<div style="display:flex;gap:2px;">{bars}</div>'


def _capacity_timeline(mc: dict, hours, ob_count: int, bkp_count: int,
                       model_label: str, model_color: str) -> str:
    """
    Grafická kapacitní vizualizace — dvě řady:
      Řada OB:  pro každou hodinu barevný blok s % využití
      Řada BKP: totéž pro BKP (pokud existuje)
    """
    def _cell(h, util, role):
        col = _color_util(util)
        p_ov = mc.get(h, {}).get(f"p_overload_{role}", 0) or 0
        wait  = mc.get(h, {}).get(f"avg_wait_{role}", 0) or 0
        badge = f'<div style="font-size:0.5rem;line-height:1;">⚠ {p_ov*100:.0f}%</div>' if p_ov > 0.2 else ""
        tip = f"Hodina {h}: využití {util:.0f}%, čekání {wait:.1f} min, P(přetíž) {p_ov*100:.0f}%"
        return (
            f'<div title="{tip}" style="flex:1;min-width:0;background:{col};border-radius:3px;'
            f'height:36px;display:flex;flex-direction:column;align-items:center;justify-content:center;'
            f'color:#fff;font-size:0.6rem;font-weight:700;">'
            f'{util:.0f}%{badge}</div>'
        )

    cells_ob = "".join(
        _cell(h, mc.get(h, {}).get("util_ob", 0) or 0, "ob")
        for h in hours
    )
    row_ob = (
        f'<div style="display:flex;align-items:center;gap:3px;">'
        f'<div style="font-size:0.62rem;color:#555;font-weight:700;min-width:38px;">OB ({ob_count})</div>'
        f'<div style="display:flex;gap:2px;flex:1;">{cells_ob}</div></div>'
    )

    row_bkp = ""
    if bkp_count > 0:
        cells_bkp = "".join(
            _cell(h, mc.get(h, {}).get("util_bkp") or 0, "bkp")
            for h in hours
        )
        row_bkp = (
            f'<div style="display:flex;align-items:center;gap:3px;margin-top:3px;">'
            f'<div style="font-size:0.62rem;color:#555;font-weight:700;min-width:38px;">BKP ({bkp_count})</div>'
            f'<div style="display:flex;gap:2px;flex:1;">{cells_bkp}</div></div>'
        )

    hour_header = (
        '<div style="display:flex;align-items:center;gap:3px;">'
        '<div style="min-width:38px;"></div>'
        '<div style="display:flex;gap:2px;flex:1;">'
        + "".join(
            f'<div style="flex:1;font-size:0.55rem;color:#aaa;text-align:center;">{h}</div>'
            for h in hours
        )
        + '</div></div>'
    )

    legend = (
        '<div style="display:flex;gap:10px;margin-top:5px;flex-wrap:wrap;">'
        + "".join(
            f'<div style="display:flex;align-items:center;gap:3px;">'
            f'<div style="width:10px;height:10px;border-radius:2px;background:{c};"></div>'
            f'<span style="font-size:0.62rem;color:#666;">{lbl}</span></div>'
            for c, lbl in [("#45b065","<50%"),("#e8c020","50–70%"),
                           ("#e07020","70–85%"),("#d62728",">85%")]
        )
        + '</div>'
    )

    return (
        f'<div style="background:#f8f9fc;border-radius:8px;padding:10px 14px;'
        f'border-left:4px solid {model_color};margin-bottom:10px;">'
        f'<div style="font-size:0.7rem;font-weight:700;color:#333;margin-bottom:5px;">'
        f'{model_label}</div>'
        f'{hour_header}{row_ob}{row_bkp}{legend}</div>'
    )


def _mc_detail_table(mc1: dict, mc2: dict, hours, bkp_count: int) -> str:
    rows = ""
    for h in hours:
        r1 = mc1.get(h, {})
        r2 = mc2.get(h, {})

        def _cell_util(r, role):
            u = r.get(f"util_{role}") or 0
            col = _color_util(u)
            return (f'<td style="padding:3px 6px;text-align:center;">'
                    f'<span style="background:{col};color:#fff;border-radius:3px;'
                    f'padding:1px 5px;font-size:0.7rem;">{u:.0f}%</span></td>')

        def _cell_wait(r, role):
            w = r.get(f"avg_wait_{role}") or 0
            p = r.get(f"p_wait15_{role}") or 0
            flag = " ⚠" if p > 0.3 else ""
            return f'<td style="padding:3px 6px;text-align:right;font-size:0.72rem;">{w:.1f} min{flag}</td>'

        arr1 = r1.get("avg_arr", 0)
        arr2 = r2.get("avg_arr", 0)

        bkp_cells = (
            _cell_util(r1, "bkp") + _cell_wait(r1, "bkp")
            + _cell_util(r2, "bkp") + _cell_wait(r2, "bkp")
        ) if bkp_count > 0 else ""

        rows += (
            f'<tr style="border-bottom:1px solid #f0f0f0;">'
            f'<td style="padding:3px 8px;font-size:0.72rem;font-weight:700;">{h}:00</td>'
            f'<td style="padding:3px 8px;font-size:0.72rem;text-align:right;">{arr1:.1f}</td>'
            f'<td style="padding:3px 8px;font-size:0.72rem;text-align:right;">{arr2:.1f}</td>'
            + _cell_util(r1, "ob") + _cell_wait(r1, "ob")
            + _cell_util(r2, "ob") + _cell_wait(r2, "ob")
            + bkp_cells
            + '</tr>'
        )

    bkp_header = (
        '<th colspan="2" style="padding:4px 6px;text-align:center;background:#f5f7fa;">BKP M1 / M2</th>'
        if bkp_count > 0 else ""
    )

    return (
        f'<table style="width:100%;border-collapse:collapse;font-size:0.72rem;">'
        f'<thead><tr style="font-size:0.65rem;text-transform:uppercase;color:#888;background:#f5f7fa;">'
        f'<th style="padding:5px 8px;">Hod.</th>'
        f'<th style="padding:5px 6px;text-align:right;">Příchozí M1</th>'
        f'<th style="padding:5px 6px;text-align:right;">Příchozí M2</th>'
        f'<th colspan="2" style="padding:4px 6px;text-align:center;">OB — Model 1</th>'
        f'<th colspan="2" style="padding:4px 6px;text-align:center;">OB — Model 2</th>'
        f'{bkp_header}</tr></thead>'
        f'<tbody>{rows}</tbody></table>'
    )


def _heatmap(df_b: pd.DataFrame, hours, weekdays) -> str:
    sub = df_b[df_b["_WEEKDAY"].isin(weekdays)]
    if sub.empty or sub["_HOUR"].isna().all():
        return "<span style='color:#ccc;font-size:0.72rem;'>Chybí data</span>"
    hm = sub.groupby(["_WEEKDAY", "_HOUR"]).size().unstack(fill_value=0)
    hm = hm.reindex(index=list(weekdays), columns=hours, fill_value=0)
    day_cnt = sub.groupby("_WEEKDAY")["_DATE"].nunique()
    for wd in weekdays:
        n = day_cnt.get(wd, 1)
        if n > 0 and wd in hm.index:
            hm.loc[wd] /= n
    vmax = max(float(hm.values.max()), 1)

    hdr = "<tr><th></th>" + "".join(
        f"<th style='font-size:0.55rem;color:#aaa;padding:1px 2px;text-align:center;'>{h}</th>"
        for h in hours
    ) + "</tr>"
    body = ""
    for wd in weekdays:
        row = f"<tr><td style='font-size:0.6rem;color:#666;font-weight:600;padding:1px 4px;'>{WEEKDAY_NAMES[wd]}</td>"
        for h in hours:
            val = hm.at[wd, h] if (wd in hm.index and h in hm.columns) else 0
            a = max(0.06, val / vmax)
            row += f"<td style='background:rgba(39,112,240,{a:.2f});width:16px;height:12px;border-radius:2px;'></td>"
        row += "</tr>"
        body += row
    return f"<table style='border-collapse:separate;border-spacing:2px;'>{hdr}{body}</table>"


def _donut(mix: dict) -> str:
    labels = {
        "schuzka_online":  ("Online schůzky",  "#2770f0"),
        "schuzka_fyzicka": ("Fyzické schůzky", "#45b065"),
        "servis":          ("Servisní návštěvy","#e07020"),
    }
    items = [(labels[k][0], mix[k], labels[k][1]) for k in mix if k in labels]
    items.sort(key=lambda x: -x[1])
    cx, cy, r, ri = 50, 50, 40, 18
    start = -math.pi / 2
    paths = ""
    for _, pct, color in items:
        if pct <= 0: continue
        angle = pct * 2 * math.pi
        end   = start + angle
        large = 1 if angle > math.pi else 0
        x1, y1 = cx + r * math.cos(start), cy + r * math.sin(start)
        x2, y2 = cx + r * math.cos(end),   cy + r * math.sin(end)
        ix1, iy1 = cx + ri * math.cos(start), cy + ri * math.sin(start)
        ix2, iy2 = cx + ri * math.cos(end),   cy + ri * math.sin(end)
        paths += (f'<path d="M{ix1:.1f},{iy1:.1f} L{x1:.1f},{y1:.1f} '
                  f'A{r},{r} 0 {large},1 {x2:.1f},{y2:.1f} '
                  f'L{ix2:.1f},{iy2:.1f} A{ri},{ri} 0 {large},0 {ix1:.1f},{iy1:.1f}Z" '
                  f'fill="{color}"/>')
        start = end
    legend = "".join(
        f'<div style="display:flex;align-items:center;gap:4px;margin-bottom:3px;">'
        f'<div style="width:9px;height:9px;border-radius:2px;background:{c};flex-shrink:0;"></div>'
        f'<span style="font-size:0.62rem;color:#555;">{lbl} <b>{pct*100:.0f}%</b></span></div>'
        for lbl, pct, c in items if pct > 0
    )
    return (f'<div style="display:flex;align-items:center;gap:10px;">'
            f'<svg width="100" height="100" viewBox="0 0 100 100">{paths}</svg>'
            f'<div>{legend}</div></div>')


def _kpi_badges(kpi: dict | None) -> str:
    if not kpi:
        return ""
    def _b(col, lbl, icon):
        v = kpi.get(col)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        return (f'<div style="background:#f5f7fa;border-radius:6px;padding:5px 10px;text-align:center;">'
                f'<div style="font-size:0.6rem;color:#aaa;font-weight:600;">{icon} {lbl}</div>'
                f'<div style="font-size:0.95rem;font-weight:800;color:#1a2340;">{int(v):,}</div></div>')
    return ('<div style="display:flex;flex-wrap:wrap;gap:7px;margin-bottom:12px;">'
            + _b("POCET_NAVSTEV_CELKEM", "Návštěv celkem",   "🚶")
            + _b("POCET_SCHUZEK_ONLINE", "Schůzky online",   "💻")
            + _b("POCET_SCHUZEK_FYZICKY","Schůzky fyzické",  "🤝")
            + _b("POCET_BEZHOT_WALK_IN", "Servis bez-hot.",  "💳")
            + _b("POCET_HOT_WALK_IN",    "Walk-in hotov.",   "💵")
            + _b("NR_NEW_ARRIVALS",      "Noví klienti",     "🆕")
            + '</div>')


def _positions_html(spec: dict | None) -> str:
    if not spec:
        return "<span style='color:#bbb;font-size:0.72rem;'>Data o pozicích nejsou k dispozici</span>"
    detail = spec.get("_POZICE_DETAIL", {})
    ob  = int(spec.get("OB_COUNT", 0))
    bkp = int(spec.get("BKP_COUNT", 0))
    tot = int(spec.get("_total_spec", 0))
    summary = (
        f'<div style="display:flex;gap:14px;margin-bottom:8px;">'
        f'<div><div style="font-size:0.6rem;color:#aaa;">Bankéři OB</div>'
        f'<div style="font-size:1rem;font-weight:800;">{ob}</div></div>'
        f'<div><div style="font-size:0.6rem;color:#aaa;">BKP medior</div>'
        f'<div style="font-size:1rem;font-weight:800;">{bkp}</div></div>'
        f'<div><div style="font-size:0.6rem;color:#aaa;">Celkem spec.</div>'
        f'<div style="font-size:1rem;font-weight:800;">{tot}</div></div></div>'
    )
    rows = "".join(
        f'<tr><td style="padding:2px 8px;font-size:0.72rem;color:#444;">{p}</td>'
        f'<td style="padding:2px 8px;font-size:0.72rem;font-weight:700;text-align:right;">{int(cnt)}</td></tr>'
        for p, cnt in sorted(detail.items(), key=lambda x: -x[1])
    ) if detail else ""
    return summary + (f'<table style="border-collapse:collapse;">{rows}</table>' if rows else "")


def _od_html(od: dict | None) -> str:
    if not od:
        return "<span style='color:#bbb;font-size:0.72rem;'>Otevírací doba není k dispozici</span>"
    days = [("PONDELI","Po"),("UTERY","Út"),("STREDA","St"),
            ("CTVRTEK","Čt"),("PATEK","Pá"),("SOBOTA","So"),("NEDELE","Ne")]
    rows = ""
    for key, lbl in days:
        f_ = str(od.get(f"{key}_OD", od.get(f"{key}_DOP._OD", "")) or "").strip()
        t_ = str(od.get(f"{key}_DO", od.get(f"{key}_DOP._DO", "")) or "").strip()
        if f_ in ("", "nan", "00:00") and t_ in ("", "nan", "00:00"):
            ts = '<span style="color:#ccc;">zavřeno</span>'
        else:
            ts = f'<b>{f_}</b> – <b>{t_}</b>'
        rows += (f'<tr><td style="padding:2px 8px;font-size:0.72rem;color:#666;font-weight:600;">{lbl}</td>'
                 f'<td style="padding:2px 8px;font-size:0.72rem;">{ts}</td></tr>')
    return f'<table style="border-collapse:collapse;">{rows}</table>'


# =============================================================================
# KARTA POBOČKY
# =============================================================================

def render_card(bc: int, name: str, df_b: pd.DataFrame, kpi: dict | None,
                spec: dict | None, od: dict | None, n_clients: int,
                network_vpc_day: float, hours: list) -> str:

    # Průměrné hodiny
    avg_work = avg_by_hour(df_b, range(5), hours)      # Po–Pá
    avg_wd   = avg_by_weekday(df_b)

    # Složení návštěv (bez hotovostních)
    mix = visit_mix_no_cash(df_b, kpi)

    ob_count    = max(int(spec.get("OB_COUNT",    0) if spec else 0), 1)
    bkp_count   = int(spec.get("BKP_COUNT",   0) if spec else 0)
    ob_jr_count = int(spec.get("OB_JR_COUNT", 0) if spec else 0)

    # ── Model 1: reálná data ─────────────────────────────────────────────────
    mc1 = run_mc_for_hours(avg_work, ob_count, bkp_count, ob_jr_count, mix)

    # ── Model 2: z počtu klientů ─────────────────────────────────────────────
    avg_work_m2 = model2_hours(avg_work, n_clients, network_vpc_day)
    mc2 = run_mc_for_hours(avg_work_m2, ob_count, bkp_count, ob_jr_count, mix)

    # Statistiky
    total_work   = len(df_b[df_b["_WEEKDAY"].between(0, 4)])
    n_work_days  = max(df_b[df_b["_WEEKDAY"].between(0, 4)]["_DATE"].nunique(), 1)
    avg_per_day  = total_work / n_work_days
    peak_h       = int(avg_work.idxmax()) if avg_work.sum() > 0 else None
    peak_util_m1 = mc1.get(peak_h, {}).get("util_ob", 0) if peak_h else 0

    def _cap_badge(util):
        if util < 50:   return '<span style="background:#45b065;color:#fff;border-radius:4px;padding:2px 8px;font-size:0.7rem;">V pohodě</span>'
        elif util < 70: return '<span style="background:#e8c020;color:#fff;border-radius:4px;padding:2px 8px;font-size:0.7rem;">Mírné zatížení</span>'
        elif util < 85: return '<span style="background:#e07020;color:#fff;border-radius:4px;padding:2px 8px;font-size:0.7rem;">Vysoké zatížení</span>'
        else:           return '<span style="background:#d62728;color:#fff;border-radius:4px;padding:2px 8px;font-size:0.7rem;">⚠ Přetížení!</span>'

    peak_badge = (f'<span style="background:#2770f0;color:#fff;border-radius:4px;padding:2px 8px;'
                  f'font-size:0.72rem;font-weight:700;">{peak_h}:00</span>'
                  if peak_h else "—")

    # Barvy OB sloupců dle MC1
    ob_colors_m1 = [_color_util(mc1.get(h, {}).get("util_ob", 0) or 0) for h in hours]
    ob_colors_m2 = [_color_util(mc2.get(h, {}).get("util_ob", 0) or 0) for h in hours]

    seg_html = ""
    if "CLIENT_SEGMENT" in df_b.columns and df_b["CLIENT_SEGMENT"].notna().any():
        by_seg = df_b["CLIENT_SEGMENT"].value_counts().head(5)
        tot = by_seg.sum()
        seg_html = '<div style="font-size:0.68rem;font-weight:700;color:#444;margin-bottom:4px;">Segmenty klientů</div>'
        for sn, sc in by_seg.items():
            pct = sc / tot * 100
            seg_html += (
                f'<div style="display:flex;align-items:center;gap:5px;margin-bottom:3px;">'
                f'<span style="font-size:0.65rem;color:#444;min-width:55px;overflow:hidden;'
                f'text-overflow:ellipsis;white-space:nowrap;" title="{sn}">{sn}</span>'
                f'<div style="flex:1;background:#f0f0f0;border-radius:3px;height:7px;">'
                f'<div style="width:{pct:.0f}%;background:#2770f0;height:100%;border-radius:3px;"></div></div>'
                f'<span style="font-size:0.62rem;color:#888;">{pct:.0f}%</span></div>'
            )

    return f"""
<div class="branch-card" data-code="{bc}" data-name="{name.lower()}"
     id="b{bc}"
     style="background:#fff;border:1px solid #e0e4ea;border-radius:10px;
            padding:18px 22px;margin-bottom:18px;box-shadow:0 1px 4px rgba(0,0,0,.05);">

  <!-- HLAVIČKA -->
  <div style="display:flex;align-items:flex-start;justify-content:space-between;
              flex-wrap:wrap;gap:8px;margin-bottom:14px;">
    <div>
      <span style="font-size:0.65rem;color:#aaa;font-weight:600;text-transform:uppercase;">Pobočka {bc}</span>
      <div style="font-size:1.05rem;font-weight:800;color:#1a2340;">{name}</div>
    </div>
    <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;">
      <div style="text-align:center;"><div style="font-size:0.6rem;color:#aaa;font-weight:600;">Návštěvy (prac.)</div>
        <div style="font-size:0.95rem;font-weight:800;">{total_work:,}</div></div>
      <div style="text-align:center;"><div style="font-size:0.6rem;color:#aaa;font-weight:600;">Prům./den</div>
        <div style="font-size:0.95rem;font-weight:800;">{avg_per_day:.1f}</div></div>
      <div style="text-align:center;"><div style="font-size:0.6rem;color:#aaa;font-weight:600;">Klientů</div>
        <div style="font-size:0.95rem;font-weight:800;">{n_clients:,}</div></div>
      <div style="text-align:center;"><div style="font-size:0.6rem;color:#aaa;font-weight:600;">OB bankéři</div>
        <div style="font-size:0.95rem;font-weight:800;">{ob_count}</div></div>
      <div style="text-align:center;"><div style="font-size:0.6rem;color:#aaa;font-weight:600;">BKP medior</div>
        <div style="font-size:0.95rem;font-weight:800;">{bkp_count}</div></div>
      <div style="text-align:center;"><div style="font-size:0.6rem;color:#aaa;font-weight:600;">Špička</div>
        <div style="margin-top:1px;">{peak_badge}</div></div>
      <div style="text-align:center;"><div style="font-size:0.6rem;color:#aaa;font-weight:600;">Kapacita (M1)</div>
        <div style="margin-top:1px;">{_cap_badge(peak_util_m1)}</div></div>
    </div>
  </div>

  <!-- KPI -->
  {_kpi_badges(kpi)}

  <!-- KAPACITNÍ VIZUALIZACE -->
  <div style="margin-bottom:14px;">
    <div style="font-size:0.7rem;font-weight:700;color:#333;text-transform:uppercase;margin-bottom:6px;">
      🎛 Kapacitní vytížení bankéřů — {ob_count} OB{f" + {bkp_count} BKP" if bkp_count else ""}
    </div>
    {_capacity_timeline(mc1, hours, ob_count, bkp_count, "Model 1 — reálné návštěvy", "#2770f0")}
    {_capacity_timeline(mc2, hours, ob_count, bkp_count,
       f"Model 2 — z {n_clients:,} klientů (síťový průměr {network_vpc_day*250:.1f} návštěv/klient/rok)", "#9b3dca")}
  </div>

  <!-- HODINOVÝ GRAF M1 / M2 -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:14px;">
    <div>
      <div style="font-size:0.68rem;font-weight:700;color:#444;margin-bottom:5px;">
        Model 1 — průměr návštěv/hod (Po–Pá)
      </div>
      {_bar_chart(avg_work, [str(h) for h in hours], colors=ob_colors_m1, height=60)}
    </div>
    <div>
      <div style="font-size:0.68rem;font-weight:700;color:#9b3dca;margin-bottom:5px;">
        Model 2 — očekávaný průměr/hod z klientů
      </div>
      {_bar_chart(avg_work_m2, [str(h) for h in hours], colors=ob_colors_m2, height=60)}
    </div>
  </div>

  <!-- 3 PANELY: den v týdnu, heatmapy -->
  <div style="display:grid;grid-template-columns:1fr auto auto;gap:16px;margin-bottom:14px;align-items:start;">
    <div>
      <div style="font-size:0.68rem;font-weight:700;color:#444;margin-bottom:5px;">Průměr dle dne v týdnu</div>
      {_bar_chart(avg_wd, WEEKDAY_NAMES, height=45, fmt="{:.0f}")}
    </div>
    <div>
      <div style="font-size:0.68rem;font-weight:700;color:#444;margin-bottom:5px;">Heatmapa Po–Pá</div>
      {_heatmap(df_b, hours, range(5))}
    </div>
    <div>
      <div style="font-size:0.68rem;font-weight:700;color:#444;margin-bottom:5px;">Heatmapa So–Ne</div>
      {_heatmap(df_b, hours, [5, 6])}
    </div>
  </div>

  <!-- SKLADBA + POZICE + SEGMENTY -->
  <div style="display:grid;grid-template-columns:auto 1fr 1fr;gap:16px;margin-bottom:14px;align-items:start;">
    <div>
      <div style="font-size:0.68rem;font-weight:700;color:#444;margin-bottom:5px;">Skladba návštěv (bez hotov.)</div>
      {_donut(mix)}
      <div style="font-size:0.62rem;color:#888;margin-top:3px;">
        Pravidla: servis: 70%≤15min · 20%=30min · 10%→eskalace 45min<br>
        Online schůzka blokuje bankéře (bez přerušení)
      </div>
    </div>
    <div>
      <div style="font-size:0.68rem;font-weight:700;color:#444;margin-bottom:5px;">Obsazení pozic</div>
      {_positions_html(spec)}
    </div>
    <div>
      {seg_html if seg_html else '<span style="color:#ccc;font-size:0.72rem;">Segmenty nejsou k dispozici</span>'}
    </div>
  </div>

  <!-- DETAIL MC TABULKA -->
  <details style="margin-top:8px;">
    <summary style="font-size:0.72rem;font-weight:700;color:#2770f0;cursor:pointer;user-select:none;padding:4px 0;">
      📊 Detail Monte Carlo simulace — po hodinách ({MC_RUNS} běhů/hodinu)
    </summary>
    <div style="margin-top:8px;overflow-x:auto;">
      {_mc_detail_table(mc1, mc2, hours, bkp_count)}
      <div style="font-size:0.6rem;color:#aaa;margin-top:5px;">
        Model 1: reálné průměry z VISITS_2025.csv (bez hotovostního walk-in) ·
        Model 2: škálováno na {n_clients:,} klientů × síťový průměr ·
        Poisson příchody · BKP odbavuje servis, OB odbavuje schůzky ·
        Bez BKP: servis přebírá OB junior · ⚠ = P(čekání&gt;15 min) &gt; 30%
      </div>
    </div>
  </details>

  <!-- OTEVÍRACÍ DOBA -->
  <details style="margin-top:4px;">
    <summary style="font-size:0.72rem;font-weight:700;color:#666;cursor:pointer;user-select:none;padding:4px 0;">
      🕐 Otevírací doba
    </summary>
    <div style="margin-top:6px;">{_od_html(od)}</div>
  </details>

</div>"""


# =============================================================================
# HLAVNÍ FUNKCE
# =============================================================================

def main():
    visits_raw, kpis_raw, spec_raw, od_raw, parties_raw = load_all()
    if visits_raw is None:
        print("❌ Bez visits dat nelze pokračovat.")
        sys.exit(1)

    print("\n⚙️  Příprava dat...")
    visits    = prep_visits(visits_raw)
    kpis_df   = prep_kpis(kpis_raw)   if kpis_raw   is not None else None
    spec_df, _= prep_spec(spec_raw)   if spec_raw   is not None else (None, [])
    od_dict   = prep_od(od_raw)       if od_raw     is not None else {}
    parties_df= prep_parties(parties_raw) if parties_raw is not None else None

    # ── Aktivní pobočky — pouze ty, které jsou v IR (kpis nebo specialiste) ──
    active_set: set = set()
    if kpis_df is not None and "BRANCH_CODE" in kpis_df.columns:
        active_set |= set(kpis_df["BRANCH_CODE"].dropna().astype(int))
    if spec_df is not None and "branch_id" in spec_df.columns:
        active_set |= set(spec_df["branch_id"].dropna().astype(int))

    all_branches = sorted(visits["BRANCH_CODE"].dropna().unique().astype(int))
    if active_set:
        branches = [bc for bc in all_branches if bc in active_set]
        skipped  = len(all_branches) - len(branches)
        print(f"   Pobočky v IR: {len(branches)} (přeskočeno {skipped} neaktivních)")
    else:
        branches = all_branches
        print(f"   Pobočky: {len(branches)} (bez filtru IR — chybí kpis/specialiste)")

    # ── Počty klientů per pobočka ─────────────────────────────────────────────
    client_counts: dict = {}
    if parties_df is not None and "BRANCH_CODE" in parties_df.columns:
        client_counts = parties_df.groupby("BRANCH_CODE").size().to_dict()

    # ── Síťový průměr návštěv/klient/pracovní den ─────────────────────────────
    total_net_visits  = len(visits[visits["BRANCH_CODE"].isin(branches) & visits["_WEEKDAY"].between(0, 4)])
    total_net_clients = sum(client_counts.get(bc, 0) for bc in branches)
    network_vpc_day   = (total_net_visits / total_net_clients / WORKING_DAYS_PER_YEAR
                         if total_net_clients > 0 else 1 / WORKING_DAYS_PER_YEAR)
    print(f"   Síťový průměr: {network_vpc_day * WORKING_DAYS_PER_YEAR:.2f} návštěv/klient/rok")

    # ── Lookup mapy ──────────────────────────────────────────────────────────
    name_map: dict = {}
    if spec_df is not None and "branch_id" in spec_df.columns and "branch_name" in spec_df.columns:
        name_map = (spec_df.dropna(subset=["branch_id"])
                    .set_index(spec_df["branch_id"].dropna().astype(int))
                    ["branch_name"].to_dict())

    kpi_map: dict = {}
    if kpis_df is not None and "BRANCH_CODE" in kpis_df.columns:
        kpi_map = {int(r["BRANCH_CODE"]): r.to_dict()
                   for _, r in kpis_df.dropna(subset=["BRANCH_CODE"]).iterrows()}

    spec_map: dict = {}
    if spec_df is not None and "branch_id" in spec_df.columns:
        spec_map = {int(r["branch_id"]): r.to_dict()
                    for _, r in spec_df.dropna(subset=["branch_id"]).iterrows()}

    hours = list(range(HOUR_FROM, HOUR_TO + 1))

    print("\n📊 Generuji karty poboček...")
    cards_html   = ""
    summary_rows = []

    for idx, bc in enumerate(branches, 1):
        df_b       = visits[visits["BRANCH_CODE"] == bc].copy()
        name       = name_map.get(bc, str(bc))
        kpi        = kpi_map.get(bc)
        spec       = spec_map.get(bc)
        od         = od_dict.get(bc)
        n_clients  = int(client_counts.get(bc, 0))

        print(f"  [{idx}/{len(branches)}] {name} ({bc})…", end="\r")

        cards_html += render_card(bc, name, df_b, kpi, spec, od,
                                  n_clients, network_vpc_day, hours)

        # Summary
        avg_work  = avg_by_hour(df_b, range(5), hours)
        peak_h    = int(avg_work.idxmax()) if avg_work.sum() > 0 else 0
        ob_count  = max(int(spec.get("OB_COUNT", 0) if spec else 0), 1)
        bkp_count = int(spec.get("BKP_COUNT", 0) if spec else 0)
        mix       = visit_mix_no_cash(df_b, kpi)
        rng_s     = np.random.default_rng(42)
        mc_peak   = simulate_hour(float(avg_work.get(peak_h, 0)), ob_count, bkp_count,
                                  0, mix, rng_s)
        u = mc_peak.get("util_ob", 0) or 0
        if u < 50:   cb = '<span style="background:#45b065;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.68rem;">V pohodě</span>'
        elif u < 70: cb = '<span style="background:#e8c020;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.68rem;">Mírné</span>'
        elif u < 85: cb = '<span style="background:#e07020;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.68rem;">Vysoké</span>'
        else:        cb = '<span style="background:#d62728;color:#fff;border-radius:3px;padding:1px 6px;font-size:0.68rem;">⚠ Přetížení</span>'

        total_v = len(df_b[df_b["_WEEKDAY"].between(0, 4)])
        n_days  = max(df_b[df_b["_WEEKDAY"].between(0, 4)]["_DATE"].nunique(), 1)
        summary_rows.append({
            "bc": bc, "name": name, "total": total_v,
            "clients": n_clients, "avg": total_v / n_days,
            "peak": peak_h, "ob": ob_count, "bkp": bkp_count,
            "cap_badge": cb,
        })

    print(f"\n  ✅ {len(branches)} poboček zpracováno")

    # ── Přehledová tabulka ────────────────────────────────────────────────────
    sum_rows_html = "".join(
        f'<tr style="border-bottom:1px solid #f0f0f0;cursor:pointer;" '
        f'onclick="document.getElementById(\'b{r["bc"]}\').scrollIntoView({{behavior:\'smooth\'}})">'
        f'<td style="padding:5px 10px;font-weight:600;">{r["bc"]}</td>'
        f'<td style="padding:5px 10px;">{r["name"]}</td>'
        f'<td style="padding:5px 10px;text-align:right;">{r["total"]:,}</td>'
        f'<td style="padding:5px 10px;text-align:right;">{r["clients"]:,}</td>'
        f'<td style="padding:5px 10px;text-align:right;">{r["avg"]:.1f}</td>'
        f'<td style="padding:5px 10px;text-align:center;">'
        f'<span style="background:#2770f0;color:#fff;border-radius:3px;padding:1px 7px;font-size:0.72rem;">'
        f'{r["peak"]}:00</span></td>'
        f'<td style="padding:5px 10px;text-align:right;">{r["ob"]}</td>'
        f'<td style="padding:5px 10px;text-align:right;">{r["bkp"]}</td>'
        f'<td style="padding:5px 10px;">{r["cap_badge"]}</td></tr>'
        for r in summary_rows
    )

    n = len(branches)
    html = f"""<!DOCTYPE html>
<html lang="cs">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>Kapacita poboček — průměrný obchodní den 2025</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0;}}
    body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
          background:#f4f6fb;color:#222;line-height:1.45;}}
    .page{{max-width:1160px;margin:0 auto;padding:30px 16px;}}
    h1{{font-size:1.5rem;font-weight:800;color:#1a2340;margin-bottom:4px;}}
    h2{{font-size:1rem;font-weight:700;color:#1a2340;margin:24px 0 10px;}}
    .subtitle{{font-size:0.8rem;color:#888;margin-bottom:22px;}}
    .box{{background:#fff;border:1px solid #e0e4ea;border-radius:10px;
          padding:16px 20px;margin-bottom:18px;box-shadow:0 1px 4px rgba(0,0,0,.05);}}
    #search{{width:100%;padding:10px 14px;font-size:0.95rem;border:1px solid #d0d4db;
             border-radius:8px;outline:none;}}
    #search:focus{{border-color:#2770f0;box-shadow:0 0 0 3px rgba(39,112,240,.12);}}
    .hidden{{display:none!important;}}
    details>summary{{list-style:none;}}
    details>summary::-webkit-details-marker{{display:none;}}
  </style>
</head>
<body>
<div class="page">

  <h1>🚶 Kapacita poboček — průměrný obchodní den 2025</h1>
  <div class="subtitle">
    Zdroje: VISITS_2025 · kpis_grouped_2026 · export_specialiste · oteviraci_doba · parties_2026
    &nbsp;|&nbsp; Pobočky IR: <b>{n}</b> &nbsp;|&nbsp; Hodiny {HOUR_FROM}–{HOUR_TO}
    &nbsp;|&nbsp; Síťový průměr: <b>{network_vpc_day * WORKING_DAYS_PER_YEAR:.2f}</b> návštěv/klient/rok
  </div>

  <!-- VYHLEDÁVAČ -->
  <div class="box">
    <div style="font-size:0.78rem;font-weight:700;color:#444;margin-bottom:7px;">🔍 Vyhledat pobočku</div>
    <input id="search" type="text"
           placeholder="Název nebo ID pobočky (např. Praha, 101, Brno)…" autocomplete="off">
    <div style="font-size:0.7rem;color:#aaa;margin-top:6px;">
      Zobrazeno: <span id="vis-count">{n}</span> / {n} poboček &nbsp;·&nbsp;
      Kliknutím na řádek v tabulce přeskočíte na detail pobočky.
    </div>
  </div>

  <!-- PŘEHLEDOVÁ TABULKA -->
  <h2>📋 Přehled poboček</h2>
  <div class="box" style="overflow-x:auto;">
    <table style="width:100%;border-collapse:collapse;font-size:0.78rem;" id="sum-tbl">
      <thead><tr style="background:#f5f7fa;font-size:0.65rem;text-transform:uppercase;color:#888;">
        <th style="padding:6px 10px;text-align:left;">Kód</th>
        <th style="padding:6px 10px;text-align:left;">Název</th>
        <th style="padding:6px 10px;text-align:right;">Návštěvy</th>
        <th style="padding:6px 10px;text-align:right;">Klientů</th>
        <th style="padding:6px 10px;text-align:right;">Prům./den</th>
        <th style="padding:6px 10px;text-align:center;">Špička</th>
        <th style="padding:6px 10px;text-align:right;">OB</th>
        <th style="padding:6px 10px;text-align:right;">BKP</th>
        <th style="padding:6px 10px;text-align:left;">Kapacita M1</th>
      </tr></thead>
      <tbody id="sum-body">{sum_rows_html}</tbody>
    </table>
  </div>

  <!-- DETAIL POBOČEK -->
  <h2>🏢 Detail poboček</h2>
  <div id="cards">{cards_html}</div>

</div>
<script>
(function() {{
  var inp   = document.getElementById('search');
  var cards = document.querySelectorAll('.branch-card');
  var rows  = document.querySelectorAll('#sum-body tr');
  var cnt   = document.getElementById('vis-count');
  var total = cards.length;

  function filter() {{
    var q = inp.value.trim().toLowerCase();
    var vis = 0;
    for (var i = 0; i < total; i++) {{
      var c = cards[i];
      var match = !q
        || (c.dataset.name || '').indexOf(q) !== -1
        || (c.dataset.code || '').indexOf(q) !== -1;
      c.classList.toggle('hidden', !match);
      if (rows[i]) rows[i].classList.toggle('hidden', !match);
      if (match) vis++;
    }}
    cnt.textContent = vis;
  }}

  inp.addEventListener('input', filter);
  inp.addEventListener('search', filter);
}})();
</script>
</body>
</html>"""

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Uloženo: {OUTPUT_FILE}  ({os.path.getsize(OUTPUT_FILE) // 1024} kB)")


if __name__ == "__main__":
    main()
