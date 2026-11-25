import datetime as dt
from pathlib import Path

import joblib, numpy as np, pandas as pd, shap, streamlit as st
import matplotlib.pyplot as plt, folium
from streamlit_folium import st_folium

from chat_engine import Chat, CITIES

st.set_page_config(page_title="HealthRadar", page_icon="🩺", layout="wide")

# =====================================================================
#  DARK THEME.  To re-colour the whole app, change the hex values in the
#  ":root{ ... }" block below.  Fonts: Poppins (headings) + Inter (body).
# =====================================================================
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@500;600;700&display=swap');

:root{
  --bg:#0F131A;       /* page background        */
  --surface:#171C26;  /* cards, chat bubbles    */
  --surface2:#1D2330; /* inner panels           */
  --line:#2A3240;     /* thin borders           */
  --text:#E6EAF2;     /* main text              */
  --soft:#B8C2D6;     /* secondary text         */
  --muted:#8C97AB;    /* captions               */
  --indigo:#6F7BE8;   /* main accent            */
  --violet:#8B7CF6;
  --teal:#4DB6AC;
  --rose:#E5546B;     /* high risk              */
  --amber:#D9A441;    /* moderate               */
  --green:#3FB68B;    /* low                    */
}

.stApp{background:var(--bg); color:var(--text);}
html, body, .stApp, [data-testid="stMarkdownContainer"], label, input, textarea{font-family:'Inter', sans-serif;}
h1, h2, h3, h4{font-family:'Poppins', sans-serif !important; font-weight:600 !important; color:var(--text);}
.block-container{max-width:1080px; padding-top:1.4rem; padding-bottom:3rem;}
footer, #MainMenu{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}

/* ---------- hero ---------- */
.hero{position:relative; overflow:hidden; border-radius:22px; padding:32px 36px; margin-bottom:18px;
  background:linear-gradient(120deg,#1B2440 0%,#2A2250 55%,#3B2147 100%); border:1px solid var(--line);}
.hero:before{content:""; position:absolute; left:0; top:0; right:0; height:3px;
  background:linear-gradient(90deg,var(--teal),var(--violet),var(--rose));}
.hero .badge{display:inline-block; background:rgba(255,255,255,.07); color:var(--soft); padding:6px 14px;
  border-radius:999px; font-weight:600; font-size:.74rem; letter-spacing:1px; border:1px solid rgba(255,255,255,.12);}
.hero .hero-title{font-family:'Poppins',sans-serif; font-weight:700; font-size:2.7rem; line-height:1.1;
  margin:14px 0 6px; color:#F4F6FB;}
.hero .hero-title span{color:var(--teal);}
.hero p{max-width:580px; font-size:1.02rem; margin:0 0 14px; color:var(--soft);}
.hero .chips span{display:inline-block; margin:4px 8px 0 0; padding:6px 14px; border-radius:999px;
  background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.14); color:#D5DCEC;
  font-weight:600; font-size:.8rem;}

/* ---------- notice ---------- */
.notice{background:rgba(217,164,65,.08); border:1px solid rgba(217,164,65,.25); border-left:4px solid var(--amber);
  border-radius:12px; padding:12px 16px; margin-bottom:16px; color:#E6D3A8; font-size:.9rem;}
.notice b{color:#F1DFB4;}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"]{gap:4px; background:var(--surface); padding:5px; border-radius:14px;
  width:fit-content; border:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{border-radius:10px; padding:8px 22px; height:auto; background:transparent;}
.stTabs [data-baseweb="tab"] p{font-family:'Poppins',sans-serif; font-weight:600; font-size:.92rem; color:var(--muted);}
.stTabs [aria-selected="true"]{background:#3B4390;}
.stTabs [aria-selected="true"] p{color:#fff !important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{display:none;}

/* ---------- chat ---------- */
[data-testid="stChatMessage"]{background:var(--surface); border:1px solid var(--line); border-radius:16px;
  padding:14px 18px; margin-bottom:12px; color:var(--text);}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li{color:var(--text);}
[data-testid="stChatMessage"] strong{color:#A5B4FC; font-weight:700;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){background:#2B3470; border-color:#38428A;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) strong{color:#fff;}
[data-testid="stChatInput"]{border-radius:16px; border:1px solid var(--line); background:var(--surface);}
[data-testid="stChatInput"]:focus-within{border-color:var(--violet);}
[data-testid="stChatInput"] textarea{color:var(--text);}

/* ---------- buttons ---------- */
.stButton > button{border-radius:12px; border:1px solid #3B4390; background:#3B4390; padding:.55rem 1.1rem;
  transition:background .15s ease, transform .15s ease;}
.stButton > button p{color:#fff !important; font-family:'Poppins',sans-serif; font-weight:600; font-size:.88rem;}
.stButton > button:hover{background:#4953A8; border-color:#4953A8; transform:translateY(-1px);}
[class*="st-key-more_"] button{background:#B5603F; border-color:#B5603F;}
[class*="st-key-more_"] button:hover{background:#C66F4D; border-color:#C66F4D;}
[class*="st-key-sat_"] button{background:#2E8F70; border-color:#2E8F70;}
[class*="st-key-sat_"] button:hover{background:#38A181; border-color:#38A181;}
[class*="st-key-new_"] button{background:transparent; border:1px solid #3A4458;}
[class*="st-key-new_"] button:hover{background:#1D2330; border-color:#4A5468;}
[class*="st-key-new_"] button p{color:var(--soft) !important;}
[data-testid="stWidgetLabel"] p{font-weight:600; color:var(--soft);}

/* ---------- result cards ---------- */
.rcard{background:var(--surface2); border-radius:16px; padding:16px 18px; border:1px solid var(--line);
  border-top:4px solid var(--green);}
.rcard.high{border-top-color:var(--rose);} .rcard.mid{border-top-color:var(--amber);} .rcard.low{border-top-color:var(--green);}
.rtop{display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;}
.rbadge{font-size:.66rem; font-weight:700; letter-spacing:1px; color:var(--soft); background:rgba(255,255,255,.07);
  padding:3px 10px; border-radius:999px;}
.rpill{font-size:.7rem; font-weight:700; padding:3px 10px; border-radius:999px; color:#0F131A; background:var(--green);}
.rcard.high .rpill{background:var(--rose); color:#fff;} .rcard.mid .rpill{background:var(--amber);}
.rname{font-family:'Poppins',sans-serif; font-weight:600; font-size:1.02rem; color:var(--text); min-height:1.6rem;}
.rpct{font-family:'Poppins',sans-serif; font-weight:700; font-size:2.5rem; line-height:1.1; color:#F4F6FB;}
.rpct span{font-size:1.15rem; margin-left:2px; color:var(--muted);}
.rbar{height:6px; background:#2A3240; border-radius:999px; overflow:hidden; margin-top:10px;}
.rbar div{height:100%; border-radius:999px; background:var(--green);}
.rcard.high .rbar div{background:var(--rose);} .rcard.mid .rbar div{background:var(--amber);}
.why{background:var(--surface2); border:1px solid var(--line); border-left:4px solid var(--violet);
  border-radius:12px; padding:12px 16px; margin:10px 0 6px; color:#D6DCEB;}
.why b{color:#A5B4FC;}
.tips{color:var(--muted); font-size:.85rem; margin:4px 2px 0;}
.tips b{color:#A5B4FC;}

/* ---------- map dashboard ---------- */
.stat{background:var(--surface); border-radius:16px; padding:16px 18px; border:1px solid var(--line);}
.stat .n{font-family:'Poppins',sans-serif; font-weight:700; font-size:1.9rem; color:#F4F6FB; line-height:1.15;}
.stat .n.small{font-size:1.1rem; padding-top:.5rem; padding-bottom:.4rem;}
.stat .l{color:var(--muted); font-weight:600; font-size:.74rem; text-transform:uppercase; letter-spacing:1px;}
.demo{display:inline-block; background:rgba(217,164,65,.12); color:#E2C07A; font-weight:600; font-size:.78rem;
  padding:6px 14px; border-radius:999px; margin:6px 0 14px; border:1px solid rgba(217,164,65,.25);}
.alert{background:var(--surface); border-radius:12px; padding:12px 16px; margin:8px 0; border:1px solid var(--line);
  border-left:4px solid var(--green); color:var(--text);}
.alert.red{border-left-color:var(--rose); background:rgba(229,84,107,.09);}
.alert.orange{border-left-color:var(--amber); background:rgba(217,164,65,.09);}
.legend{display:flex; gap:18px; flex-wrap:wrap; margin:10px 0 16px;}
.legend span{font-weight:500; color:var(--soft); font-size:.86rem;}
.dot{display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px;}
iframe{border-radius:16px; border:1px solid var(--line);}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{background:#0B0F15; border-right:1px solid #1F2632;}
.side-title{font-family:'Poppins',sans-serif; font-weight:700; font-size:1.55rem; line-height:1.2; color:#F4F6FB;}
.side-sub{color:var(--muted); font-weight:500; font-size:.84rem; margin-bottom:12px;}
.side-card{background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:12px 16px;
  margin:12px 0; font-size:.86rem; color:var(--soft);}
.side-card b{font-family:'Poppins',sans-serif; color:#A5B4FC;}
.side-card ol{margin:6px 0 0 16px; padding:0;}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

HERO = ('<div class="hero"><span class="badge">🩺 AI HEALTH COMPANION</span>'
        '<div class="hero-title">Health<span>Radar</span></div>'
        '<p>Describe how you feel in your own words. Get an explainable risk read, and see what is '
        'spreading across Indian cities.</p>'
        '<div class="chips"><span>🧠 Explainable AI</span><span>💬 Typo-friendly chat</span>'
        '<span>🗺️ Outbreak map</span></div></div>')

NOTICE = ('<div class="notice">⚠️ <b>HealthRadar is an educational project, not a medical device,</b> and it cannot '
          'diagnose illness. In an emergency (chest pain, trouble breathing, severe bleeding) call <b>112</b> '
          'or go to the nearest hospital.</div>')

ROSE, AMBER, GREEN, BLUE = "#E5546B", "#D9A441", "#3FB68B", "#6F8FE8"
CARD_BG = "#171C26"


# ---------------------------------------------------------------- model
@st.cache_resource
def load_model():
    b = joblib.load("model.joblib")
    return b["rf"], b["xgb"], b["le"], b["features"], shap.TreeExplainer(b["rf"])


rf, xgb, le, features, explainer = load_model()


def pretty(f):
    return f.replace("_", " ").strip()


def predict(symptoms):
    """Top-3 diseases with probability and the 6 strongest SHAP drivers for each."""
    x = pd.DataFrame([[1 if f in symptoms else 0 for f in features]], columns=features)
    p = (rf.predict_proba(x)[0] + xgb.predict_proba(x)[0]) / 2
    sv = explainer.shap_values(x)
    out = []
    for i in np.argsort(p)[::-1][:3]:
        vals = sv[i][0] if isinstance(sv, list) else sv[0, :, i]
        order = np.argsort(np.abs(vals))[::-1][:6]
        out.append({"disease": le.classes_[i], "prob": float(p[i]),
                    "drivers": [(features[j], float(vals[j]), int(x.iloc[0, j])) for j in order]})
    return out


def log_report(city, disease):
    f = Path("data/reports.csv")
    pd.DataFrame([{"date": dt.date.today().isoformat(), "city": city,
                   "disease": disease, "cases": 1}]).to_csv(
        f, mode="a", header=not f.exists(), index=False)


# ---------------------------------------------------------------- chat state
ss = st.session_state
if "chat" not in ss or not hasattr(ss.chat, "more_details"):
    ss.chat = Chat(features, predict, log_report)
chat = ss.chat


# ---------------------------------------------------------------- result UI
def card_html(rank, d):
    p = d["prob"]
    pct = round(p * 100)
    tier, label = ("high", "High") if p >= .6 else ("mid", "Moderate") if p >= .3 else ("low", "Low")
    badge = "TOP MATCH" if rank == 1 else f"#{rank}"
    return (f'<div class="rcard {tier}"><div class="rtop"><span class="rbadge">{badge}</span>'
            f'<span class="rpill">{label}</span></div><div class="rname">{d["disease"]}</div>'
            f'<div class="rpct">{pct}<span>%</span></div>'
            f'<div class="rbar"><div style="width:{max(pct, 3)}%"></div></div></div>')


def explain_html(d):
    ups = [pretty(f) for f, v, p in d["drivers"] if v > 0 and p == 1]
    if not ups:
        return ("None of your reported symptoms strongly pointed here. "
                "The score mostly reflects other typical symptoms being absent.")
    s = f"The biggest push toward <b>{d['disease']}</b> came from <b>{ups[0]}</b>"
    return s + (f", backed up by <b>{ups[1]}</b>." if len(ups) > 1 else ".")


def render_result(res, key, latest):
    with st.chat_message("assistant", avatar="🩺"):
        cols = st.columns(3)
        for rank, (c, d) in enumerate(zip(cols, res), start=1):
            c.markdown(card_html(rank, d), unsafe_allow_html=True)
        pick = st.selectbox("Why this score? Pick a condition:", [d["disease"] for d in res],
                            key=f"pick_{key}")
        d = next(x for x in res if x["disease"] == pick)
        st.markdown(f'<div class="why">💡 {explain_html(d)}</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(6.4, 3.2), facecolor=CARD_BG)
        ax.set_facecolor(CARD_BG)
        labels = [pretty(f) + (" (you reported)" if p else " (not reported)")
                  for f, v, p in d["drivers"]][::-1]
        vals = [v for _, v, _ in d["drivers"]][::-1]
        ax.barh(labels, vals, color=[ROSE if v > 0 else BLUE for v in vals], height=0.6)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.axvline(0, color="#4A5468", lw=1)
        ax.tick_params(axis="y", length=0, labelsize=9, colors="#C7D0E2")
        ax.tick_params(axis="x", length=0, labelsize=8, colors="#8C97AB")
        ax.set_xlabel("lowers the score   ←   SHAP value   →   raises the score",
                      fontsize=8.5, fontweight="bold", color="#8C97AB")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("🟥 raises the score   ·   🟦 lowers it   ·   "
                   "⚠️ Educational project, not medical advice. Please see a doctor for real concerns.")
        if latest:
            c1, c2, c3 = st.columns(3)
            c1.button("🔍 Not satisfied? Add details", key=f"more_{key}", on_click=chat.more_details)
            c2.button("👍 This helps", key=f"sat_{key}", on_click=chat.satisfied)
            c3.button("🔄 New check", key=f"new_{key}", on_click=chat.reset)


# ---------------------------------------------------------------- map helpers
def load_regional():
    df = pd.read_csv("data/regional_daily.csv", parse_dates=["date"])
    shift = pd.Timestamp.today().normalize() - df["date"].max().normalize()
    df["date"] = df["date"] + shift          # keeps the simulated outbreaks "recent"
    f = Path("data/reports.csv")
    if f.exists():
        df = pd.concat([df, pd.read_csv(f, parse_dates=["date"])])
    df["disease"] = df["disease"].str.strip()
    return df


def anomalies(df):
    end = df["date"].max().normalize()
    idx = pd.date_range(end - pd.Timedelta(days=41), end)
    rows = []
    for (city, dis), g in df.groupby(["city", "disease"]):
        s = g.groupby("date")["cases"].sum().reindex(idx, fill_value=0)
        base, recent = s.iloc[:-7], s.iloc[-7:]
        z = (recent.mean() - base.mean()) / (base.std() + 1)
        rows.append({"city": city, "disease": dis, "z": z,
                     "recent": int(recent.sum()), "baseline": round(base.mean() * 7, 1)})
    return pd.DataFrame(rows)


def stat_card(label, value, small=False):
    cls = "n small" if small else "n"
    return f'<div class="stat"><div class="l">{label}</div><div class="{cls}">{value}</div></div>'


# ---------------------------------------------------------------- layout
with st.sidebar:
    st.markdown('<div class="side-title">🩺 HealthRadar</div>'
                '<div class="side-sub">Predictive risk + outbreak radar</div>', unsafe_allow_html=True)
    if st.button("🔄 Start over"):
        chat.reset()
        st.rerun()
    st.markdown('<div class="side-card"><b>How it works</b><ol><li>Tell Radar your name and city</li>'
                '<li>Describe your symptoms in your own words</li>'
                '<li>See your top 3 matches and <i>why</i></li>'
                '<li>Not happy? Add more details</li></ol></div>'
                '<div class="side-card"><b>Good to know</b><br>Typos are fine. Say <i>"I don\'t have fever"</i> '
                'to remove a symptom, or type <i>restart</i> any time.</div>', unsafe_allow_html=True)
    st.caption("Regional feed is simulated. Symptom model trained on a public dataset.")

st.markdown(HERO, unsafe_allow_html=True)
st.markdown(NOTICE, unsafe_allow_html=True)

tab1, tab2 = st.tabs(["💬  Chat with Radar", "🗺️  Outbreak map"])

with tab1:
    last_result = max([i for i, (r, _) in enumerate(chat.msgs) if r == "result"], default=-1)
    for i, (role, content) in enumerate(chat.msgs):
        if role == "result":
            render_result(content, i, latest=(i == last_result))
        else:
            with st.chat_message(role, avatar="🩺" if role == "assistant" else None):
                st.markdown(content)
    if len(chat.msgs) <= 1:
        st.markdown('<div class="tips">💡 <b>Try saying:</b> "hi", "my name is Priya", then '
                    '"fever, headache and joint pain".</div>', unsafe_allow_html=True)
    prompt = st.chat_input("Type here... (spelling mistakes are fine)")
    if prompt:
        with st.spinner("Radar is thinking..."):
            chat.handle(prompt)
        st.rerun()

with tab2:
    st.markdown('<span class="demo">🧪 DEMO DATA · case counts on this map are simulated, not real reports</span>',
                unsafe_allow_html=True)
    df = load_regional()
    choice = st.selectbox("Disease", ["All diseases"] + sorted(df["disease"].unique()))
    a = anomalies(df)
    if choice != "All diseases":
        a = a[a["disease"] == choice]
    hot = a.sort_values("z", ascending=False).groupby("city").head(1)

    if len(hot):
        top = hot.iloc[0]
        n_alert = int((hot.z >= 1.5).sum())
        n_crit = int((hot.z >= 3).sum())
        s1, s2, s3 = st.columns(3)
        s1.markdown(stat_card("Cities monitored", int(hot.city.nunique())), unsafe_allow_html=True)
        s2.markdown(stat_card("Active alerts", f"{n_alert} <span style='font-size:.95rem;color:{ROSE}'>"
                                               f"({n_crit} critical)</span>"), unsafe_allow_html=True)
        s3.markdown(stat_card("Biggest spike", f"{top.city} · {top.disease}", small=True), unsafe_allow_html=True)
        st.write("")

    # Dark map tiles (free CARTO basemap, no API key). If they ever fail to load, swap the two
    # tiles/attr lines for:  tiles="OpenStreetMap"
    m = folium.Map(location=[22.5, 79], zoom_start=5,
                   tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
                   attr="&copy; OpenStreetMap contributors &copy; CARTO")
    for _, r in hot.iterrows():
        if r.city not in CITIES:
            continue
        color = ROSE if r.z >= 3 else AMBER if r.z >= 1.5 else GREEN
        folium.CircleMarker(CITIES[r.city], radius=8 + min(max(r.z, 0), 6) * 3,
                            color=color, weight=2, fill=True, fill_color=color, fill_opacity=.5,
                            tooltip=f"{r.city}: {r.disease}",
                            popup=f"<b>{r.city}</b><br>{r.disease}<br>Last 7 days: {r.recent} cases"
                                  f"<br>Normal week: ~{r.baseline}<br>Anomaly score: {r.z:.1f}").add_to(m)
    st_folium(m, height=520, width=1000, returned_objects=[])

    st.markdown(f'<div class="legend"><span><i class="dot" style="background:{ROSE}"></i>Outbreak alert</span>'
                f'<span><i class="dot" style="background:{AMBER}"></i>Watch</span>'
                f'<span><i class="dot" style="background:{GREEN}"></i>Normal</span></div>', unsafe_allow_html=True)

    shown = 0
    for _, r in hot.sort_values("z", ascending=False).head(3).iterrows():
        if r.z >= 1.5:
            shown += 1
            cls = "red" if r.z >= 3 else "orange"
            st.markdown(f'<div class="alert {cls}">🚨 <b>{r.city}</b> · {r.disease} is running about '
                        f'<b>{r.recent / max(r.baseline, 1):.1f}×</b> its usual weekly level.</div>',
                        unsafe_allow_html=True)
    if not shown:
        st.markdown('<div class="alert">✅ <b>All clear.</b> No unusual spikes right now.</div>',
                    unsafe_allow_html=True)