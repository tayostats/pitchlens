"""PitchLens studio + fan view.  Run:  streamlit run pitchlens_app.py   (API must be running on :8000)"""
import time
import requests
import pandas as pd
import altair as alt
import streamlit as st
from html import escape as esc

st.set_page_config(page_title="PitchLens", page_icon="⚽", layout="wide")
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Karla:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg-base:#070b14;--bg-card:#0d1422;--bg-card2:#111827;--border:#1a2640;--border2:#243050;--amber:#f5a623;--green:#00d68f;
--red:#ff5f6d;--blue:#60a5fa;--text-hi:#e8ecf4;--text-mid:#a0aec0;--text-lo:#4a5568;--font-head:'Syne',sans-serif;--font-body:'Karla',sans-serif;
--radius-lg:20px;--radius-md:12px;--radius-sm:8px}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg-base)!important;font-family:var(--font-body)!important}
[data-testid="stSidebar"]{background:#060a12!important;border-right:1px solid #12192b!important}
[data-testid="stSidebar"] .stMarkdown h2{font-family:var(--font-head)!important;font-size:1rem!important;color:var(--amber)!important;letter-spacing:.12em;text-transform:uppercase}
#MainMenu,footer{visibility:hidden}header{background:var(--bg-base)!important}
h1,h2,h3{font-family:var(--font-head)!important;color:var(--text-hi)!important}
.block-container{padding-top:1.4rem}
.section-label{font-family:var(--font-head);font-size:.65rem;letter-spacing:.18em;color:var(--amber);text-transform:uppercase;margin:2rem 0 1rem;display:flex;align-items:center;gap:10px}
.section-label::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,#1a2640 0%,transparent 100%)}
.hero-wrap{background:linear-gradient(135deg,#0a1120 0%,#070b14 60%,#0d1422 100%);border:1px solid var(--border);border-radius:var(--radius-lg);padding:1.8rem 2.2rem 1.5rem;margin-bottom:1.4rem;position:relative;overflow:hidden}
.hero-wrap::before{content:"";position:absolute;inset:0;background-image:radial-gradient(circle at 80% 50%,rgba(245,166,35,.06) 0%,transparent 55%),radial-gradient(circle at 20% 80%,rgba(0,214,143,.04) 0%,transparent 45%);pointer-events:none}
.hero-wrap::after{content:"";position:absolute;inset:0;background-image:radial-gradient(circle,#1a2640 1px,transparent 1px);background-size:28px 28px;opacity:.45;pointer-events:none}
.hero-inner{position:relative;z-index:1}
.hero-eyebrow{font-size:.62rem;letter-spacing:.2em;color:var(--amber);text-transform:uppercase;margin-bottom:.5rem}
.hero-title{font-family:var(--font-head);font-size:2.4rem;font-weight:800;color:var(--text-hi);line-height:1.1;margin:0 0 .5rem}
.hero-meta{font-size:.78rem;color:var(--text-lo)}.hero-meta span{color:var(--text-mid)}
.board{display:flex;align-items:center;justify-content:space-between;padding:20px 28px;border-radius:var(--radius-lg);border:1px solid var(--border2);box-shadow:0 8px 24px rgba(0,0,0,.35);background:linear-gradient(90deg,var(--hc) -30%,var(--bg-card) 38%,var(--bg-card) 62%,var(--ac) 130%);color:var(--text-hi);margin-bottom:10px}
.board .t{font-family:var(--font-head);font-size:1.2rem;font-weight:700;width:30%}.board .s{font-family:var(--font-head);font-size:3.2rem;font-weight:800}
.pill{background:rgba(245,166,35,.12);border:1px solid var(--amber);color:var(--amber);padding:3px 16px;border-radius:99px;font-family:var(--font-head);font-weight:700}
.card{background:linear-gradient(160deg,#0d1422 0%,#111827 100%);border:1px solid var(--border);border-left:5px solid var(--c);border-radius:var(--radius-md);padding:12px 18px;margin-bottom:10px;box-shadow:0 8px 24px rgba(0,0,0,.3);transition:border-color .2s,box-shadow .2s;color:var(--text-mid)}
.card:hover{border-color:var(--border2);box-shadow:0 12px 32px rgba(0,0,0,.4)}
.card .h{font-family:var(--font-head);font-weight:700;color:var(--text-hi);margin-bottom:4px}
.card .m{color:var(--text-lo);font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;margin-top:6px}
.chip{font-size:.6rem;letter-spacing:.14em;text-transform:uppercase;color:var(--amber);border:1px solid var(--amber-dim,#7a5010);border-radius:99px;padding:1px 8px;margin-left:8px}
.bar{display:flex;height:9px;border-radius:6px;overflow:hidden}
.row{display:flex;justify-content:space-between;font-size:.85rem;color:var(--text-mid);margin-top:12px;margin-bottom:3px}.row b{color:var(--text-hi);font-family:var(--font-head)}
.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid var(--border)}
.stTabs [data-baseweb="tab"]{font-family:var(--font-head);font-size:.78rem;letter-spacing:.06em;color:var(--text-mid)}
.stTabs [aria-selected="true"]{color:var(--amber)!important}
.stButton>button{background:var(--bg-card2);border:1px solid var(--border2);color:var(--text-hi);border-radius:var(--radius-sm);font-family:var(--font-head);font-weight:600}
.stButton>button:hover{border-color:var(--amber);color:var(--amber)}
[data-testid="stExpander"]{background:var(--bg-card);border:1px solid var(--border)!important;border-radius:var(--radius-md)}
[data-testid="stMetricValue"]{font-family:var(--font-head)!important;color:var(--text-hi)!important}
[data-testid="stMetricLabel"]{font-size:.7rem!important;letter-spacing:.1em!important;text-transform:uppercase!important;color:var(--text-lo)!important}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero-wrap"><div class="hero-inner"><div class="hero-eyebrow">Explainable Match Intelligence</div><div class="hero-title">PitchLens</div><div class="hero-meta"><span>Every moment explained. Every fan, their own match.</span> · 100% synthetic match data</div></div></div>', unsafe_allow_html=True)
API = "http://localhost:8000"


def get(path, **params):
    r = requests.get(f"{API}{path}", params=params, timeout=90)
    r.raise_for_status()
    return r.json()


def post(path, body):
    r = requests.post(f"{API}{path}", json=body, timeout=90)
    r.raise_for_status()
    return r.json()


try:
    meta, players, health = get("/match/meta"), get("/players"), get("/health")
except Exception as ex:
    st.error(f"Can't reach the API at {API}. Start it with: uvicorn pitchlens_api:app --port 8000\n\n{ex}")
    st.stop()

# ---- sidebar: personalization (stage 5) ------------------------------------------------------
st.sidebar.markdown("## ⚽ PitchLens")
st.sidebar.caption("Claude narration: " + ("ON" if health["claude_enabled"] else "OFF (template fallback)"))
persona = {"Analyst": "analyst", "Casual fan": "casual", "Player focus": "player_focus"}[
    st.sidebar.radio("Viewing mode", ["Analyst", "Casual fan", "Player focus"], index=1)]
lang_names = {"English": "en", "Français": "fr", "Português": "pt", "Yorùbá": "yo", "Hausa": "ha", "Igbo": "ig", "Naija Pidgin": "pcm"}
lang = lang_names[st.sidebar.selectbox("Language", list(lang_names))]
club_opts = {"No favourite": None, **{v: k for k, v in meta["teams"].items()}}
fav = club_opts[st.sidebar.selectbox("Favourite club", list(club_opts))]
focus = None
if persona == "player_focus":
    focus = st.sidebar.selectbox("Follow player", [p["name"] for p in players])

# ---- match clock ------------------------------------------------------------------------------
st.session_state.setdefault("minute", 0)
st.session_state.setdefault("playing", False)
if st.session_state.playing and st.session_state.minute < 90:
    st.session_state.minute += 1
elif st.session_state.minute >= 90:
    st.session_state.playing = False
c1, c2, c3 = st.sidebar.columns(3)
if c1.button("▶ Play"):
    st.session_state.playing = True
if c2.button("⏸ Pause"):
    st.session_state.playing = False
if c3.button("⏮ Reset"):
    st.session_state.playing, st.session_state.minute = False, 0
    st.rerun()
minute = st.sidebar.slider("Match minute", 0, 90, key="minute")

# ---- scoreboard -------------------------------------------------------------------------------
state = get("/match/state", minute=minute)
H, A = state["HOME"], state["AWAY"]
mom = pd.DataFrame(get("/match/momentum"))
now = mom[mom.minute == max(minute, 1)].iloc[0]
COL = {"Arsenal": "#EF0107", "Chelsea": "#1b5fc1"}
hc, ac = COL.get(H["name"], "#EF0107"), COL.get(A["name"], "#1b5fc1")
pct = (now["index"] + 100) / 2
st.markdown(f'''<div class="board" style="--hc:{hc};--ac:{ac}"><div class="t">{H["name"]}</div><div class="s">{state["score"]["HOME"]}</div><div class="pill">{minute}'</div><div class="s">{state["score"]["AWAY"]}</div><div class="t" style="text-align:right">{A["name"]}</div></div>
<div class="bar" style="background:{ac}"><div style="width:{pct}%;background:{hc}"></div></div>
<div class="row"><span>◀ {H["name"]} control</span><span><b>{now.state.upper()}</b> · {now.turnovers_per_min} turnovers/min</span><span>{A["name"]} control ▶</span></div>''', unsafe_allow_html=True)

tab_live, tab_stats, tab_narr, tab_recap, tab_json = st.tabs(
    ["📺 Live overlays", "📊 Match intelligence", "🎙 Narrate a moment", "📝 Recap", "{ } Overlay JSON"])

with tab_live:
    cards = get("/overlays", minute=minute, persona=persona, language=lang, favorite_team=fav, focus_player=focus, last=8)
    if not cards:
        st.info("Press ▶ Play or drag the minute slider.")
    for c in reversed(cards):
        icon = {"goal": "🥅", "big_chance": "🎯", "press_win": "🪤", "momentum_shift": "🌊", "sprint": "⚡",
                "milestone": "🏃", "key_pass": "🎩", "player_pulse": "📍"}.get(c["type"], "⏱")
        col = hc if c["team"] == "HOME" else ac if c["team"] == "AWAY" else "#94a3b8"
        st.markdown(f'<div class="card" style="--c:{col}"><div class="h">{icon} {c["show_at_s"] // 60}\' · {esc(c["headline"])}{" ⭐" if c["tone"] == "favorite" else ""}<span class="chip">{c["type"].replace("_", " ")}</span></div><div>{esc(c["body"])}</div><div class="m">{c["layout"]} · priority {c["priority"]} · {c["duration_s"]}s on screen</div></div>', unsafe_allow_html=True)

with tab_stats:
    st.markdown('<div class="section-label">Head to Head</div>', unsafe_allow_html=True)
    rows = [("Possession %", "possession_pct"), ("Territory %", "territory_pct"), ("Passes", "passes"),
            ("Pass accuracy %", "pass_accuracy_pct"), ("Avg pass difficulty (0-100)", "avg_pass_difficulty"),
            ("Shots", "shots"), ("On target", "shots_on_target"), ("xG", "xg"), ("Tackles", "tackles"),
            ("Pressures", "pressures"), ("PPDA (lower = more pressing)", "ppda"), ("Top player speed km/h", "top_player_speed_kmh"),
            ("Max shot speed km/h", "max_shot_speed_kmh"), ("Max pass speed km/h", "max_pass_speed_kmh")]
    for label, k in [("Possession %", "possession_pct"), ("Territory %", "territory_pct"), ("Pass accuracy %", "pass_accuracy_pct"),
                     ("Shots", "shots"), ("xG", "xg"), ("Tackles", "tackles"), ("Pressures", "pressures")]:
        tot = (H[k] + A[k]) or 1
        st.markdown(f'<div class="row"><b>{H[k]}</b><span>{label}</span><b>{A[k]}</b></div><div class="bar" style="background:{ac}"><div style="width:{100 * H[k] / tot}%;background:{hc}"></div></div>', unsafe_allow_html=True)
    with st.expander("Full stats table"):
        st.dataframe(pd.DataFrame({H["name"]: [H[k] for _, k in rows], A["name"]: [A[k] for _, k in rows]},
                                  index=[n for n, _ in rows]), width="stretch")
    st.markdown('<div class="section-label">Momentum &amp; Control</div>', unsafe_allow_html=True)
    st.altair_chart(alt.Chart(mom[mom.minute <= max(minute, 1)]).mark_bar().encode(
        x=alt.X("minute:Q", scale=alt.Scale(domain=[0, 90])),
        y=alt.Y("index:Q", scale=alt.Scale(domain=[-100, 100]), title=f"{A['name']} ◀   ▶ {H['name']}"),
        color=alt.condition("datum.index > 0", alt.value(hc), alt.value(ac))).properties(height=260)
        .configure(background="#0d1422").configure_view(strokeWidth=0)
        .configure_axis(labelColor="#a0aec0", titleColor="#a0aec0", gridColor="#1a2640", domainColor="#1a2640"), width="stretch")
    st.caption("Index = blend of possession, territory, xG and pressing over a rolling 5 minutes. Turnover rate separates control from chaos.")

with tab_narr:
    moments = get("/match/moments", minute=minute)
    if moments:
        pick = st.selectbox("Moment", moments[::-1], format_func=lambda m: f"{m['minute']}' {m['headline']}")
        with st.expander("Why it matters (computed evidence)", expanded=True):
            st.write(pick["why"])
            st.json(pick["facts"])
        if st.button("Narrate with Claude"):
            r = post("/narrate", dict(moment_id=pick["id"], persona=persona, language=lang, favorite_team=fav, focus_player=focus))
            st.success(r["text"])
            st.caption(f"engine: {r['engine']}")

with tab_recap:
    if st.button("Generate recap up to this minute"):
        with st.spinner("Writing recap..."):
            r = post("/recap", dict(minute=minute, persona=persona, language=lang, favorite_team=fav, focus_player=focus))
        st.write(r["recap"])
        st.caption(f"engine: {r['engine']}")

with tab_json:
    st.caption("Machine-readable, timecoded cards a graphics/rendering layer can consume. Live version: GET /stream (SSE).")
    st.json(cards[-3:] if cards else [])

if st.session_state.playing:
    time.sleep(0.8)
    st.rerun()