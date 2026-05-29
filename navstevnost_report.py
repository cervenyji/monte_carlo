"""
Průměrný obchodní den — hodinová návštěvnost poboček
=====================================================
Načte VISITS_2025.csv a vygeneruje HTML report s přehledem průměrného
obchodního dne pro každou pobočku (počet návštěv dle hodiny dne).

Spuštění (ze složky internirating_report/):
    python navstevnost_report.py

Vstup:  ../in/tables/VISITS_2025.csv
Výstup: navstevnost_obchodni_den.html
"""

import os
import sys
import warnings
import math

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Konfigurace
# ---------------------------------------------------------------------------

VISITS_PATH = "../in/tables/VISITS_2025.csv"
OUTPUT_FILE = "navstevnost_obchodni_den.html"

# Rozsah hodin zobrazených v grafu
HOUR_FROM = 7
HOUR_TO   = 19  # včetně

# Barva sloupců grafu (CSS gradient)
BAR_COLOR_PEAK = "#2770f0"
BAR_COLOR_LOW  = "#a8c4f5"

# ---------------------------------------------------------------------------
# Načtení dat
# ---------------------------------------------------------------------------

def load_visits(path: str) -> pd.DataFrame:
    print(f"📂 Načítám: {path}")
    if not os.path.exists(path):
        print(f"❌ Soubor nenalezen: {path}")
        print("   Ujistěte se, že spouštíte skript ze složky internirating_report/")
        sys.exit(1)
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    print(f"   ✅ Načteno {len(df):,} řádků, sloupce: {list(df.columns)}")
    return df


def parse_visits(df: pd.DataFrame) -> pd.DataFrame:
    # Detekce ID pobočky
    id_col = next((c for c in ["BRANCH_ID", "BRANCH_CODE", "POBOCKA"] if c in df.columns), None)
    if id_col is None:
        raise ValueError(f"Nenalezen sloupec s ID pobočky. Dostupné: {list(df.columns)}")
    df["_BRANCH"] = pd.to_numeric(df[id_col], errors="coerce")

    # Parsování data
    if "VISIT_DATE" in df.columns:
        df["_DT"] = pd.to_datetime(df["VISIT_DATE"], errors="coerce")
        df["_WEEKDAY"] = df["_DT"].dt.weekday   # 0=Po … 4=Pá
        df["_DATE"]    = df["_DT"].dt.date
    else:
        raise ValueError("Nenalezen sloupec VISIT_DATE")

    # Parsování hodiny
    if "VISIT_TIME" in df.columns:
        df["_HOUR"] = pd.to_numeric(
            df["VISIT_TIME"].astype(str).str.split(":").str[0], errors="coerce"
        )
    elif "VISIT_HOUR" in df.columns:
        df["_HOUR"] = pd.to_numeric(df["VISIT_HOUR"], errors="coerce")
    else:
        df["_HOUR"] = None

    # Název pobočky (volitelný)
    name_col = next((c for c in ["BRANCH_NAME", "POBOCKA_NAZEV", "NAME"] if c in df.columns), None)
    if name_col:
        df["_NAME"] = df[name_col].astype(str)
    else:
        df["_NAME"] = df["_BRANCH"].astype(str)

    return df


# ---------------------------------------------------------------------------
# Výpočet průměrného dne
# ---------------------------------------------------------------------------

def compute_avg_day(df_branch: pd.DataFrame, hour_from: int, hour_to: int) -> pd.Series:
    """
    Pro danou pobočku vrátí průměrný počet návštěv na hodinu přes pracovní dny.
    Průměruje se přes počet unikátních pracovních dní (ne týdnů).
    """
    hours = list(range(hour_from, hour_to + 1))

    work = df_branch[df_branch["_WEEKDAY"].between(0, 4)].copy()
    if work.empty or work["_HOUR"].isna().all():
        return pd.Series(0.0, index=hours)

    # Počet unikátních pracovních dní
    n_days = work["_DATE"].nunique()
    if n_days == 0:
        return pd.Series(0.0, index=hours)

    by_hour = work.groupby("_HOUR").size().reindex(hours, fill_value=0)
    return by_hour / n_days


def compute_avg_weekday(df_branch: pd.DataFrame) -> pd.Series:
    """Průměrný počet návštěv dle dne v týdnu (Po-Pá)."""
    work = df_branch[df_branch["_WEEKDAY"].between(0, 4)]
    if work.empty:
        return pd.Series(0.0, index=range(5))
    n_weeks = work["_DATE"].nunique() / 5  # hrubý odhad
    n_weeks = max(n_weeks, 1)
    by_wd = work.groupby("_WEEKDAY").size().reindex(range(5), fill_value=0)
    return by_wd / n_weeks


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

WEEKDAY_NAMES = ["Po", "Út", "St", "Čt", "Pá"]
MONTH_NAMES   = ["Led","Úno","Bře","Dub","Kvě","Čvn","Čvc","Srp","Zář","Říj","Lis","Pro"]


def _bar_chart(values: pd.Series, labels, color_peak=BAR_COLOR_PEAK, color_low=BAR_COLOR_LOW,
               height_px=80, label_fmt="{:.1f}") -> str:
    """Vodorovný sloupcový bar chart jako inline HTML."""
    vmax = max(values.max(), 0.01)
    bars = ""
    for label, val in zip(labels, values):
        pct = val / vmax * 100
        is_peak = (val == values.max())
        color = color_peak if is_peak else color_low
        val_str = label_fmt.format(val)
        bars += f"""
        <div style="display:flex;flex-direction:column;align-items:center;gap:2px;flex:1;min-width:0;">
          <div style="font-size:0.62rem;color:#555;font-weight:{'700' if is_peak else '400'};">{val_str}</div>
          <div style="width:100%;background:#eef0f4;border-radius:3px 3px 0 0;height:{height_px}px;display:flex;align-items:flex-end;">
            <div style="width:100%;height:{pct:.1f}%;background:{color};border-radius:3px 3px 0 0;transition:height .2s;"></div>
          </div>
          <div style="font-size:0.62rem;color:#888;">{label}</div>
        </div>"""
    return f'<div style="display:flex;gap:3px;align-items:flex-end;">{bars}</div>'


def _mini_heatmap(hm_df: pd.DataFrame, hour_from: int, hour_to: int) -> str:
    """Heatmapa den × hodina jako HTML tabulka."""
    if hm_df is None or hm_df.empty:
        return ""
    vmax = max(hm_df.values.max(), 1)
    hours = list(range(hour_from, hour_to + 1))

    header = "<tr><th style='font-size:0.6rem;color:#aaa;padding:1px 3px;'></th>"
    for h in hours:
        header += f"<th style='font-size:0.6rem;color:#aaa;padding:1px 3px;text-align:center;'>{h}</th>"
    header += "</tr>"

    rows = ""
    for wd_idx, wd_name in enumerate(WEEKDAY_NAMES):
        row = f"<tr><td style='font-size:0.65rem;color:#666;padding:1px 4px;font-weight:600;'>{wd_name}</td>"
        for h in hours:
            val = hm_df.at[wd_idx, h] if (wd_idx in hm_df.index and h in hm_df.columns) else 0
            ratio = val / vmax
            alpha = max(0.07, ratio)
            row += f"<td style='background:rgba(39,112,240,{alpha:.2f});width:18px;height:14px;border-radius:2px;text-align:center;'></td>"
        row += "</tr>"
        rows += row

    return f"""
    <table style="border-collapse:separate;border-spacing:2px;">
      {header}{rows}
    </table>"""


def render_branch_card(branch_code, branch_name, df_branch, hour_from, hour_to) -> str:
    hours   = list(range(hour_from, hour_to + 1))
    avg_day = compute_avg_day(df_branch, hour_from, hour_to)
    avg_wd  = compute_avg_weekday(df_branch)

    total_visits = len(df_branch[df_branch["_WEEKDAY"].between(0, 4)])
    n_days       = df_branch[df_branch["_WEEKDAY"].between(0, 4)]["_DATE"].nunique()
    peak_hour    = int(avg_day.idxmax()) if avg_day.sum() > 0 else None
    peak_val     = float(avg_day.max())

    # Heatmapa weekday × hour
    work = df_branch[df_branch["_WEEKDAY"].between(0, 4)].copy()
    hm_df = None
    if not work.empty and work["_HOUR"].notna().any():
        hm = work.groupby(["_WEEKDAY", "_HOUR"]).size().unstack(fill_value=0)
        hm = hm.reindex(index=range(5), columns=hours, fill_value=0)
        # průměr per den
        day_counts = work.groupby("_WEEKDAY")["_DATE"].nunique()
        for wd in range(5):
            cnt = day_counts.get(wd, 1)
            if cnt > 0 and wd in hm.index:
                hm.loc[wd] = hm.loc[wd] / cnt
        hm_df = hm

    # Měsíční přehled
    by_month = df_branch[df_branch["_WEEKDAY"].between(0, 4)].groupby(
        df_branch["_DT"].dt.month
    ).size().reindex(range(1, 13), fill_value=0)

    hour_chart = _bar_chart(avg_day, [str(h) for h in hours], height_px=70)
    wd_chart   = _bar_chart(avg_wd, WEEKDAY_NAMES, height_px=50)
    heatmap_html = _mini_heatmap(hm_df, hour_from, hour_to)
    month_chart  = _bar_chart(by_month, MONTH_NAMES, height_px=40, label_fmt="{:.0f}")

    peak_badge = (
        f'<span style="background:#2770f0;color:#fff;border-radius:4px;padding:2px 8px;'
        f'font-size:0.75rem;font-weight:700;">{peak_hour}:00</span>'
        if peak_hour is not None else ""
    )

    return f"""
<div style="background:#fff;border:1px solid #e0e4ea;border-radius:10px;padding:20px 24px;
            margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,.06);">

  <!-- Hlavička -->
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;flex-wrap:wrap;gap:8px;">
    <div>
      <span style="font-size:0.7rem;color:#aaa;font-weight:600;text-transform:uppercase;letter-spacing:.05em;">
        Pobočka {branch_code}
      </span>
      <div style="font-size:1.05rem;font-weight:700;color:#1a2340;">{branch_name}</div>
    </div>
    <div style="display:flex;gap:16px;flex-wrap:wrap;">
      <div style="text-align:center;">
        <div style="font-size:0.65rem;color:#aaa;font-weight:600;text-transform:uppercase;">Celkem návštěv</div>
        <div style="font-size:1.1rem;font-weight:800;color:#1a2340;">{total_visits:,}</div>
      </div>
      <div style="text-align:center;">
        <div style="font-size:0.65rem;color:#aaa;font-weight:600;text-transform:uppercase;">Prac. dní</div>
        <div style="font-size:1.1rem;font-weight:800;color:#1a2340;">{n_days}</div>
      </div>
      <div style="text-align:center;">
        <div style="font-size:0.65rem;color:#aaa;font-weight:600;text-transform:uppercase;">Průměr/den</div>
        <div style="font-size:1.1rem;font-weight:800;color:#1a2340;">{(total_visits/n_days if n_days>0 else 0):.1f}</div>
      </div>
      <div style="text-align:center;">
        <div style="font-size:0.65rem;color:#aaa;font-weight:600;text-transform:uppercase;">Špička</div>
        <div style="margin-top:2px;">{peak_badge} <span style="font-size:0.75rem;color:#666;">({peak_val:.1f} ø/den)</span></div>
      </div>
    </div>
  </div>

  <!-- Hodinový přehled -->
  <div style="margin-bottom:12px;">
    <div style="font-size:0.72rem;font-weight:700;color:#444;text-transform:uppercase;letter-spacing:.04em;margin-bottom:6px;">
      Průměrný počet návštěv dle hodiny (pracovní dny)
    </div>
    {hour_chart}
  </div>

  <!-- Spodní řada: den v týdnu + heatmapa + měsíce -->
  <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:20px;align-items:start;margin-top:14px;">

    <!-- Den v týdnu -->
    <div>
      <div style="font-size:0.7rem;font-weight:700;color:#444;text-transform:uppercase;letter-spacing:.04em;margin-bottom:6px;">
        Průměr dle dne v týdnu
      </div>
      {wd_chart}
    </div>

    <!-- Heatmapa -->
    <div>
      <div style="font-size:0.7rem;font-weight:700;color:#444;text-transform:uppercase;letter-spacing:.04em;margin-bottom:6px;">
        Heatmapa den × hodina
      </div>
      {heatmap_html if heatmap_html else '<span style="color:#bbb;font-size:0.75rem;">Data o čase nejsou k dispozici</span>'}
    </div>

    <!-- Měsíce -->
    <div>
      <div style="font-size:0.7rem;font-weight:700;color:#444;text-transform:uppercase;letter-spacing:.04em;margin-bottom:6px;">
        Celkem návštěv dle měsíce
      </div>
      {month_chart}
    </div>

  </div>
</div>"""


def render_summary_table(summary: pd.DataFrame) -> str:
    rows = ""
    for _, r in summary.iterrows():
        rows += f"""
        <tr style="border-bottom:1px solid #f0f0f0;">
          <td style="padding:6px 10px;font-weight:600;color:#1a2340;">{int(r['branch_code'])}</td>
          <td style="padding:6px 10px;">{r['branch_name']}</td>
          <td style="padding:6px 10px;text-align:right;">{int(r['total_visits']):,}</td>
          <td style="padding:6px 10px;text-align:right;">{int(r['n_days'])}</td>
          <td style="padding:6px 10px;text-align:right;">{r['avg_per_day']:.1f}</td>
          <td style="padding:6px 10px;text-align:center;">
            <span style="background:#2770f0;color:#fff;border-radius:4px;padding:2px 8px;font-size:0.8rem;font-weight:700;">
              {int(r['peak_hour']) if not math.isnan(r['peak_hour']) else '–'}:00
            </span>
          </td>
          <td style="padding:6px 10px;text-align:right;">{r['peak_avg']:.1f}</td>
        </tr>"""
    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
      <thead>
        <tr style="background:#f5f7fa;font-size:0.7rem;text-transform:uppercase;color:#888;letter-spacing:.05em;">
          <th style="padding:8px 10px;text-align:left;">Kód</th>
          <th style="padding:8px 10px;text-align:left;">Název</th>
          <th style="padding:8px 10px;text-align:right;">Celkem návštěv</th>
          <th style="padding:8px 10px;text-align:right;">Prac. dní</th>
          <th style="padding:8px 10px;text-align:right;">Průměr/den</th>
          <th style="padding:8px 10px;text-align:center;">Hodina špičky</th>
          <th style="padding:8px 10px;text-align:right;">Špička ø/den</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


def render_network_hour_chart(df_all: pd.DataFrame, hour_from: int, hour_to: int) -> str:
    """Bar chart průměrného dne za celou síť."""
    hours = list(range(hour_from, hour_to + 1))
    work = df_all[df_all["_WEEKDAY"].between(0, 4)].copy()
    if work.empty or work["_HOUR"].isna().all():
        return "<p style='color:#aaa;'>Data o hodinách nejsou k dispozici.</p>"

    n_branch_days = work.groupby("_BRANCH")["_DATE"].nunique().sum()
    n_branches = work["_BRANCH"].nunique()
    avg_per_branch_day = (
        work.groupby("_HOUR").size().reindex(hours, fill_value=0)
        / (n_branch_days / n_branches if n_branches > 0 else 1)
    )
    return _bar_chart(avg_per_branch_day, [str(h) for h in hours], height_px=100)


# ---------------------------------------------------------------------------
# Hlavní funkce
# ---------------------------------------------------------------------------

def main():
    df_raw = load_visits(VISITS_PATH)
    df = parse_visits(df_raw)

    branches = sorted(df["_BRANCH"].dropna().unique().astype(int))
    print(f"   Pobočky: {len(branches)}")

    # Průměrný název pobočky
    name_map = (
        df.dropna(subset=["_BRANCH"])
        .groupby("_BRANCH")["_NAME"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else str(x.iloc[0]))
        .to_dict()
    )

    # Přehledová tabulka
    summary_rows = []
    for bc in branches:
        db = df[df["_BRANCH"] == bc]
        work = db[db["_WEEKDAY"].between(0, 4)]
        n_days = work["_DATE"].nunique()
        total  = len(work)
        avg_day = compute_avg_day(db, HOUR_FROM, HOUR_TO)
        peak_h  = float(avg_day.idxmax()) if avg_day.sum() > 0 else float("nan")
        summary_rows.append({
            "branch_code":  bc,
            "branch_name":  name_map.get(bc, str(bc)),
            "total_visits": total,
            "n_days":       n_days,
            "avg_per_day":  total / n_days if n_days > 0 else 0,
            "peak_hour":    peak_h,
            "peak_avg":     float(avg_day.max()),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("total_visits", ascending=False)

    # Síťový hodinový přehled
    network_chart = render_network_hour_chart(df, HOUR_FROM, HOUR_TO)
    summary_table = render_summary_table(summary_df)

    # Karty poboček
    print("📊 Generuji karty poboček...")
    cards_html = ""
    for idx, bc in enumerate(branches, 1):
        db = df[df["_BRANCH"] == bc]
        name = name_map.get(bc, str(bc))
        cards_html += render_branch_card(bc, name, db, HOUR_FROM, HOUR_TO)
        if idx % 50 == 0:
            print(f"   {idx}/{len(branches)}")

    # Sestavení HTML
    html = f"""<!DOCTYPE html>
<html lang="cs">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Průměrný obchodní den — hodinová návštěvnost poboček 2025</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f4f6fb; color: #222; line-height: 1.4; }}
    .page {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px; }}
    h1 {{ font-size: 1.6rem; font-weight: 800; color: #1a2340; margin-bottom: 4px; }}
    .subtitle {{ font-size: 0.85rem; color: #888; margin-bottom: 28px; }}
    h2 {{ font-size: 1.1rem; font-weight: 700; color: #1a2340; margin: 28px 0 12px; }}
    .section-box {{ background:#fff; border:1px solid #e0e4ea; border-radius:10px;
                    padding:20px 24px; margin-bottom:24px;
                    box-shadow:0 1px 4px rgba(0,0,0,.06); overflow-x:auto; }}
    .legend {{ font-size:0.72rem; color:#aaa; margin-top:6px; }}
  </style>
</head>
<body>
<div class="page">

  <h1>🚶 Průměrný obchodní den — hodinová návštěvnost poboček</h1>
  <div class="subtitle">
    Zdroj: VISITS_2025.csv &nbsp;|&nbsp;
    Počet poboček: {len(branches)} &nbsp;|&nbsp;
    Pouze pracovní dny (Po–Pá) &nbsp;|&nbsp;
    Hodiny: {HOUR_FROM}:00–{HOUR_TO}:00
  </div>

  <!-- Síťový přehled -->
  <h2>📊 Průměrný obchodní den — celá síť</h2>
  <div class="section-box">
    <div style="font-size:0.72rem;font-weight:700;color:#444;text-transform:uppercase;
                letter-spacing:.04em;margin-bottom:8px;">
      Průměrný počet návštěv / hodinu / pobočku (pracovní dny, celá síť)
    </div>
    {network_chart}
    <div class="legend">Hodnoty jsou průměrovány přes všechny pobočky a pracovní dny.</div>
  </div>

  <!-- Přehledová tabulka -->
  <h2>📋 Přehled poboček</h2>
  <div class="section-box">
    {summary_table}
  </div>

  <!-- Karty poboček -->
  <h2>🏢 Detail poboček</h2>
  {cards_html}

</div>
</body>
</html>"""

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n✅ Report uložen: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
