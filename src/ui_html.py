"""
ui_html.py
----------
Pure HTML/CSS builders for the dashboard look (no Streamlit import, so they can be previewed in any browser).
Every builder returns ONE line of HTML without blank lines - Streamlit's markdown parser would otherwise
split the block and show raw tags.
"""
import html

# ------------------------------------------------------------------ icons (stroke style, 24x24 grid)
_ICON_PATHS = {
    "user": '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "users": '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>'
             '<path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>'
                '<path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>',
    "sliders": '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/>'
               '<line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/>'
               '<line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/>'
               '<line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/>'
               '<line x1="17" y1="16" x2="23" y2="16"/>',
    "network": '<circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/>'
               '<line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/>',
    "bars": '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/>'
            '<line x1="6" y1="20" x2="6" y2="14"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/>',
    "star": '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
    "bulb": '<path d="M9 18h6"/><path d="M10 22h4"/><path d="M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.3 1 2.3h6c0-1 .4-1.8 1-2.3A7 7 0 0 0 12 2z"/>',
    "layout": '<rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/>'
              '<line x1="9" y1="21" x2="9" y2="9"/>',
    "percent": '<line x1="19" y1="5" x2="5" y2="19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "trend": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    "info": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    "cap": '<path d="M22 10L12 5 2 10l10 5 10-5z"/><path d="M6 12v5c0 1.5 3 3 6 3s6-1.5 6-3v-5"/>'
           '<line x1="22" y1="10" x2="22" y2="16"/>',
    "cog": '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06'
           'a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33'
           'l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9'
           'a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09'
           'a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9'
           'a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>',
}


def icon(name: str, size: int = 22) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
            f'aria-hidden="true">{_ICON_PATHS[name]}</svg>')


# Decorative rings + graduation cap shown on the right side of the hero banner.
_HERO_DECO = ('<svg xmlns="http://www.w3.org/2000/svg" width="150" height="110" viewBox="0 0 150 110" fill="none" aria-hidden="true">'
              '<circle cx="75" cy="55" r="52" stroke="rgba(255,255,255,.28)" stroke-width="1.5" stroke-dasharray="3 5"/>'
              '<circle cx="75" cy="55" r="40" stroke="rgba(255,255,255,.35)" stroke-width="1.5"/>'
              '<circle cx="75" cy="55" r="30" fill="rgba(255,255,255,.14)"/>'
              '<path d="M75 36 L102 49 L75 62 L48 49 Z" fill="#fff"/>'
              '<path d="M60 56 V69 C60 74 68 77 75 77 C82 77 90 74 90 69 V56 L75 63 Z" fill="rgba(255,255,255,.88)"/>'
              '<path d="M102 49 V64" stroke="#fff" stroke-width="2.2" stroke-linecap="round"/>'
              '<circle cx="102" cy="66" r="2.6" fill="#fff"/></svg>')

# ------------------------------------------------------------------ builders


def hero_html(title: str, subtitle: str, icon_name: str = "trend") -> str:
    return (f'<div class="hero"><div class="hero-ic">{icon(icon_name, 40)}</div>'
            f'<div class="hero-body"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>'
            f'<div class="hero-deco">{_HERO_DECO}</div></div>')


def note_html(text: str) -> str:
    return f'<div class="note"><span class="note-ic">{icon("info", 20)}</span><span>{html.escape(text)}</span></div>'


def kpi_cards_html(items: list) -> str:
    """items: (icon_name, label, value, tone[, tooltip]); tone in blue|green|indigo|violet|cyan."""
    cards = []
    for item in items:
        icon_name, label, value, tone = item[:4]
        tip = html.escape(item[4] if len(item) > 4 else f"{label}: {value}", quote=True)
        cards.append(f'<div class="kpi kpi-{tone}" title="{tip}"><div class="kpi-ic">{icon(icon_name, 22)}</div>'
                     f'<div class="kpi-label">{html.escape(label)}</div><div class="kpi-value">{html.escape(str(value))}</div></div>')
    return f'<div class="kpi-grid">{"".join(cards)}</div>'


def section_title_html(text: str, icon_name: str = "network") -> str:
    return (f'<div class="sec-title"><span class="sec-ic">{icon(icon_name, 18)}</span>'
            f'<span>{html.escape(text)}</span></div>')


# (icon, title, sub-text). The first PIPELINE_ROW1 steps go on the first row, the rest on the second.
PIPELINE_STEPS = [("user", "User Input", "form on Prediction page"),
                  ("database", "Data Validation", "ranges, types, 24 h rule"),
                  ("sliders", "Preprocessing", "impute, scale, encode"),
                  ("network", "ML Model", "selected by cross-validation"),
                  ("bars", "Probability", "P(pass) from the model"),
                  ("shield", "Risk Assessment", "Low / Moderate / High"),
                  ("star", "Explainable AI", "Shapley contributions"),
                  ("bulb", "Recommendations", "prioritised + what-if"),
                  ("layout", "Dashboard", "cards, gauges, history")]
PIPELINE_ROW1 = 6


def pipeline_html(steps=None, row1: int = PIPELINE_ROW1) -> str:
    steps = steps or PIPELINE_STEPS

    def row(chunk, cls):
        parts = []
        for i, (ic, title, sub) in enumerate(chunk):
            parts.append(f'<div class="pstep"><div class="pstep-ic">{icon(ic, 24)}</div>'
                         f'<b>{html.escape(title)}</b><span>{html.escape(sub)}</span></div>')
            if i < len(chunk) - 1:
                parts.append('<div class="parrow">&rarr;</div>')
        return f'<div class="prow {cls}">{"".join(parts)}</div>'

    return f'<div class="pipeline">{row(steps[:row1], "prow-6")}{row(steps[row1:], "prow-3")}</div>'


# ------------------------------------------------------------------ stylesheet
CSS = """
<style>
:root {--brand:#0A9BC4; --brand-dark:#0B2C5E; --surface:#F1F5FB; --line:#D3E6F3; --ink:#16233A; --accent:#0A9BC4;}
html, body, [class*="css"], .stMarkdown {font-family: "Segoe UI", Tahoma, Arial, "Noto Sans Arabic", sans-serif;}
/* Mixed Arabic/English text: each paragraph picks its own direction from its first strong character. */
.stMarkdown p, .stMarkdown li, .stCaption, input[type="text"] {unicode-bidi: plaintext;}

/* ---- page background and layout */
.stApp {background: linear-gradient(180deg, #E8F4FB 0%, #F4FAFE 38%, #FFFFFF 100%);}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 2rem; max-width: 1240px;}

/* ---- sidebar: deep navy navigation */
section[data-testid="stSidebar"] {background: linear-gradient(180deg, #0B2C5E 0%, #0A3A78 100%); border-right: none;}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a,
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] {
    color: #DCE8F8 !important; border-radius: 10px; padding: .55rem .8rem; margin: 2px 0; font-weight: 500;}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a *,
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] * {color: inherit !important;}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover,
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover {background: rgba(255,255,255,.10) !important;}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"],
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] {
    background: linear-gradient(90deg, #1596D8, #19AEE6) !important; color: #fff !important; font-weight: 700;
    box-shadow: 0 2px 8px rgba(20,163,220,.35);}
section[data-testid="stSidebar"] .stMarkdown h3 {color: #fff; font-size: 1.05rem;}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color: #A9C0E3;}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button,
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] button {color: #fff;}
section[data-testid="stSidebar"] hr {border-color: rgba(255,255,255,.15);}

/* ---- hero banner */
.hero {position: relative; overflow: hidden; display: flex; align-items: center; gap: 1rem;
       background: linear-gradient(100deg, #0A4DA6 0%, #0F7FD0 55%, #12CFE4 100%);
       padding: 1.5rem 1.7rem; border-radius: 14px; color: #fff; margin-bottom: 1rem;
       box-shadow: 0 8px 22px rgba(15,100,190,.22);}
.hero-ic {flex: none; align-self: flex-start; color: #fff; opacity: .95; padding-top: .15rem;}
.hero-body {flex: 1; min-width: 0;}
.hero h1 {margin: 0; font-size: 1.85rem; font-weight: 700; color: #fff; padding: 0; line-height: 1.25;}
.hero p {margin: .45rem 0 0; opacity: .95; font-size: 1rem; max-width: 46rem; color: #fff;}
.hero-deco {flex: none; margin-right: -.4rem;}

/* ---- note / disclaimer */
.note {display: flex; align-items: center; gap: .75rem; background: #E3F5FB; border: 1px solid #BDE3F2;
       border-radius: 12px; padding: .85rem 1.1rem; color: #1D4E6E; font-size: .92rem; line-height: 1.45; margin-bottom: 1.1rem;}
.note-ic {flex: none; display: inline-flex; color: #14A3DC;}

/* ---- KPI cards */
.kpi-grid {display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .85rem; margin: .2rem 0 1.4rem;}
.kpi {border-radius: 12px; padding: .95rem 1.1rem 1rem; border: 1px solid transparent; min-width: 0;
      box-shadow: 0 1px 3px rgba(22,58,117,.06);}
.kpi-ic {display: inline-flex; margin-bottom: .35rem;}
.kpi-label {font-size: .9rem; color: #33475F; font-weight: 500;}
.kpi-value {font-size: 1.95rem; font-weight: 700; line-height: 1.2; margin-top: .1rem;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.kpi-blue   {background: #E4F0FC; border-color: #CFE2F7;} .kpi-blue   .kpi-ic, .kpi-blue   .kpi-value {color: #1A63C9;}
.kpi-green  {background: #E1F6EA; border-color: #C5EBD4;} .kpi-green  .kpi-ic, .kpi-green  .kpi-value {color: #1F9D5B;}
.kpi-indigo {background: #E9EEFD; border-color: #D3DCF8;} .kpi-indigo .kpi-ic, .kpi-indigo .kpi-value {color: #2A4FD6;}
.kpi-violet {background: #EFEBFB; border-color: #DDD5F6;} .kpi-violet .kpi-ic, .kpi-violet .kpi-value {color: #4A47C9;}
.kpi-cyan   {background: #E0F4F9; border-color: #C2E7F1;} .kpi-cyan   .kpi-ic, .kpi-cyan   .kpi-value {color: #0D8DB4;}

/* ---- section title */
.sec-title {display: flex; align-items: center; gap: .6rem; font-size: 1.3rem; font-weight: 700; color: #11284A; margin: .4rem 0 .8rem;}
.sec-ic {display: inline-flex; align-items: center; justify-content: center; width: 34px; height: 34px;
         border-radius: 10px; background: #D6EEFA; color: #1596D8;}

/* ---- pipeline */
.pipeline {display: flex; flex-direction: column; gap: .7rem; margin-bottom: 1.4rem;}
.prow {display: flex; align-items: stretch; gap: .35rem;}
.pstep {flex: 1 1 0; min-width: 0; text-align: center; padding: .8rem .5rem .75rem; border-radius: 12px;
        background: linear-gradient(180deg, #F2FBFE, #E7F5FB); border: 1px solid #C9E6F2; color: var(--ink);}
.pstep-ic {display: inline-flex; color: #1673C4; margin-bottom: .3rem;}
.pstep b {display: block; font-size: .93rem; color: #11284A;}
.pstep span {display: block; font-size: .78rem; color: #5B6F88; margin-top: .15rem; line-height: 1.3;}
.parrow {align-self: center; font-size: 1.15rem; color: #1596D8; flex: none;}
.prow-3 .pstep {padding: .65rem .8rem;}


/* ---- shared look for every page: white rounded cards, cyan accents */
[data-testid="stVerticalBlockBorderWrapper"] {background: #fff; border: 1px solid #D3E6F3 !important; border-radius: 14px !important;
    box-shadow: 0 2px 8px rgba(11,44,94,.05);}
.stMarkdown h2, .stMarkdown h3, [data-testid="stHeading"] h2, [data-testid="stHeading"] h3 {color: #0B2C5E; font-weight: 700;}
.stMarkdown h4, .stMarkdown h5 {color: #0B2C5E;}
.stButton > button, .stDownloadButton > button, [data-testid="stPageLink"] a, .stFormSubmitButton > button {
    border-radius: 10px; border: 1px solid #BFE0EF; background: #fff; color: #0B2C5E; font-weight: 600;}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stPageLink"] a:hover {border-color: #0A9BC4; color: #0A9BC4; background: #F1FAFE;}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
    background: linear-gradient(90deg, #0A9BC4, #12BFE4); color: #fff; border: none; box-shadow: 0 3px 10px rgba(10,155,196,.30);}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {color: #fff; background: linear-gradient(90deg, #088AB0, #0FB0D3);}
[data-testid="stTabs"] button[aria-selected="true"] {color: #0A9BC4; font-weight: 700;}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {background-color: #0A9BC4;}
[data-testid="stExpander"] {border: 1px solid #D3E6F3; border-radius: 12px; background: #fff;}
[data-testid="stAlert"] {border-radius: 12px;}
[data-testid="stAlert"][data-baseweb="notification"] {border: 1px solid #D3E6F3;}
[data-testid="stDataFrame"] {border: 1px solid #D3E6F3; border-radius: 12px; overflow: hidden;}
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"] {border-radius: 10px;}
[data-testid="stSlider"] [role="slider"] {background-color: #0A9BC4;}
[data-testid="stMetricValue"] {color: #0B2C5E;}

/* ---- existing components */
[data-testid="stMetric"] {background: #fff; border: 1px solid var(--line); border-radius: 12px;
                          padding: .75rem 1rem; box-shadow: 0 2px 8px rgba(11,44,94,.05);}
[data-testid="stMetricLabel"] {color: #4a5b78;}
.verdict {padding: .9rem; border-radius: 12px; text-align: center; font-size: 1.5rem; font-weight: 700;
          border: 2px solid; margin: .4rem 0 1rem 0; background: #fff;}
.badge {display: inline-block; padding: .2rem .75rem; border-radius: 999px; color: #fff; font-weight: 600;}

@media (max-width: 1000px) {
  .kpi-grid {grid-template-columns: repeat(3, minmax(0, 1fr));}
  .prow {flex-wrap: wrap;} .pstep {flex: 1 1 140px;} .parrow {display: none;}
}
@media (max-width: 640px) {
  .hero {padding: 1.1rem 1rem;} .hero h1 {font-size: 1.35rem;} .hero-deco, .hero-ic {display: none;}
  .kpi-grid {grid-template-columns: repeat(2, minmax(0, 1fr));} .kpi-value {font-size: 1.5rem;}
  .verdict {font-size: 1.2rem;}
}
</style>
"""
