import datetime as dt
from pathlib import Path

import joblib, numpy as np, pandas as pd, shap, streamlit as st
import matplotlib.pyplot as plt, folium
from streamlit_folium import st_folium

from chat_engine import Chat, CITIES

st.set_page_config(page_title="HealthRadar", page_icon="🩺", layout="wide")

# =====================================================================
#  DESIGN: fonts, colours, cards.  Edit the hex colours here to re-theme.
#    indigo  #4F46E5   violet #7C3AED   pink #EC4899   teal #14B8A6
#  Fonts: Poppins (headings, bold) + Inter (body) from Google Fonts.
# =====================================================================
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@500;600;700;800&display=swap');

.stApp{
  background:
    radial-gradient(900px 420px at 6% -6%, #E0E7FF 0, transparent 60%),
    radial-gradient(800px 380px at 96% 0%, #FCE7F3 0, transparent 55%),
    #F8FAFC;
}
html, body, .stApp, [data-testid="stMarkdownContainer"], label, input, textarea{
  font-family:'Inter', sans-serif;
}
h1, h2, h3, h4{font-family:'Poppins', sans-serif !important; font-weight:700 !important; color:#1E293B;}
.block-container{max-width:1080px; padding-top:1.4rem; padding-bottom:3rem;}
footer, #MainMenu{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}

/* ---------- hero ---------- */
.hero{position:relative; overflow:hidden; border-radius:28px; padding:34px 38px; margin-bottom:20px; color:#fff;
  background:linear-gradient(120deg,#4F46E5 0%,#7C3AED 50%,#EC4899 100%);
  box-shadow:0 18px 40px rgba(79,70,229,.28);}
.hero:before,.hero:after{content:""; position:absolute; border-radius:50%; background:rgba(255,255,255,.12);}
.hero:before{width:260px; height:260px; right:-60px; top:-90px;}
.hero:after{width:150px; height:150px; right:170px; bottom:-80px;}
.hero .badge{display:inline-block; background:rgba(255,255,255,.2); padding:6px 14px; border-radius:999px;
  font-weight:700; font-size:.78rem; letter-spacing:.8px;}
.hero .hero-title{font-family:'Poppins',sans-serif; font-weight:800; font-size:2.8rem; line-height:1.1; margin:12px 0 6px;}
.hero .hero-title span{color:#FDE68A;}
.hero p{max-width:580px; font-size:1.05rem; opacity:.95; margin:0 0 14px; color:#fff;}
.hero .chips span{display:inline-block; margin:4px 8px 0 0; padding:6px 14px; border-radius:999px;
  background:#fff; color:#4F46E5; font-weight:700; font-size:.8rem;}

/* ---------- notice ---------- */
.notice{background:#FFFBEB; border:1px solid #FDE68A; border-left:6px solid #F59E0B; border-radius:14px;
  padding:12px 16px; margin-bottom:16px; color:#78350F; font-size:.92rem;}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"]{gap:6px; background:#fff; padding:6px; border-radius:999px; width:fit-content;
  box-shadow:0 6px 20px rgba(79,70,229,.12);}
.stTabs [data-baseweb="tab"]{border-radius:999px; padding:8px 22px; height:auto;}
.stTabs [data-baseweb="tab"] p{font-family:'Poppins',sans-serif; font-weight:700; font-size:.95rem;}
.stTabs [aria-selected="true"]{background:linear-gradient(120deg,#4F46E5,#EC4899);}
.stTabs [aria-selected="true"] p{color:#fff !important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{display:none;}

/* ---------- chat ---------- */
[data-testid="stChatMessage"]{background:#fff; border:1px solid #EEF2FF; border-radius:20px; padding:14px 18px;
  margin-bottom:12px; box-shadow:0 6px 18px rgba(15,23,42,.06);}
[data-testid="stChatMessage"] strong{color:#4338CA; font-weight:700;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
  background:linear-gradient(120deg,#4F46E5,#7C3AED); border:none;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) strong{color:#fff;}
[data-testid="stChatInput"]{border-radius:24px; border:2px solid #E0E7FF; background:#fff;
  box-shadow:0 10px 26px rgba(79,70,229,.14);}
[data-testid="stChatInput"]:focus-within{border-color:#7C3AED;}

/* ---------- buttons ---------- */
.stButton > button{border-radius:999px; border:none; padding:.55rem 1.2rem;
  background:linear-gradient(120deg,#4F46E5,#7C3AED); box-shadow:0 8px 18px rgba(79,70,229,.25);
  transition:transform .15s ease, box-shadow .15s ease;}
.stButton > button p{color:#fff !important; font-family:'Poppins',sans-serif; font-weight:700; font-size:.9rem;}
.stButton > button:hover{transform:translateY(-2px); box-shadow:0 12px 24px rgba(79,70,229,.32);}
[class*="st-key-more_"] button{background:linear-gradient(120deg,#F97316,#EC4899);
  box-shadow:0 8px 18px rgba(236,72,153,.28);}
[class*="st-key-sat_"] button{background:linear-gradient(120deg,#10B981,#14B8A6);
  box-shadow:0 8px 18px rgba(16,185,129,.28);}
[class*="st-key-new_"] button{background:#fff; border:2px solid #C7D2FE; box-shadow:none;}
[class*="st-key-new_"] button p{color:#4F46E5 !important;}
[data-testid="stWidgetLabel"] p{font-weight:700; color:#334155;}

/* ---------- result cards ---------- */
.rcard{background:#fff; border-radius:20px; padding:16px 18px; border:1px solid #EEF2FF; position:relative;
  box-shadow:0 8px 22px rgba(15,23,42,.07); border-top:6px solid #10B981; transition:transform .15s ease;}
.rcard:hover{transform:translateY(-3px);}
.rcard.high{border-top-color:#F43F5E;} .rcard.mid{border-top-color:#F59E0B;} .rcard.low{border-top-color:#10B981;}
.rtop{display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;}
.rbadge{font-size:.68rem; font-weight:800; letter-spacing:.8px; color:#6366F1; background:#EEF2FF;
  padding:3px 10px; border-radius:999px;}
.rpill{font-size:.72rem; font-weight:700; padding:3px 10px; border-radius:999px; color:#fff; background:#10B981;}
.rcard.high .rpill{background:#F43F5E;} .rcard.mid .rpill{background:#F59E0B;}
.rname{font-family:'Poppins',sans-serif; font-weight:700; font-size:1.05rem; color:#1E293B; min-height:1.6rem;}
.rpct{font-family:'Poppins',sans-serif; font-weight:800; font-size:2.6rem; line-height:1.1; color:#4F46E5;}
.rpct span{font-size:1.2rem; margin-left:2px; color:#818CF8;}
.rbar{height:8px; background:#EEF2FF; border-radius:999px; overflow:hidden; margin-top:8px;}
.rbar div{height:100%; border-radius:999px; background:linear-gradient(90deg,#10B981,#34D399);}
.rcard.high .rbar div{background:linear-gradient(90deg,#F43F5E,#FB7185);}
.rcard.mid .rbar div{background:linear-gradient(90deg,#F59E0B,#FBBF24);}
.why{background:linear-gradient(120deg,#EEF2FF,#FDF2F8); border-radius:14px; padding:12px 16px; margin:10px 0 6px;
  color:#1E293B;}
.key{font-size:.82rem; color:#64748B; font-weight:600; margin-bottom:6px;}
.tips{color:#64748B; font-size:.85rem; margin:4px 2px 0;}
.tips b{color:#4F46E5;}

/* ---------- map dashboard ---------- */
.stat{background:#fff; border-radius:18px; padding:16px 18px; border:1px solid #EEF2FF;
  box-shadow:0 8px 22px rgba(15,23,42,.06);}
.stat .n{font-family:'Poppins',sans-serif; font-weight:800; font-size:2rem; color:#4F46E5; line-height:1.15;}
.stat .n.small{font-size:1.15rem; padding-top:.45rem; padding-bottom:.35rem;}
.stat .l{color:#64748B; font-weight:600; font-size:.82rem; text-transform:uppercase; letter-spacing:.6px;}
.demo{display:inline-block; background:#FEF3C7; color:#92400E; font-weight:700; font-size:.8rem; padding:6px 14px;
  border-radius:999px; margin:6px 0 14px;}
.alert{background:#fff; border-radius:14px; padding:12px 16px; margin:8px 0; border-left:6px solid #10B981;
  box-shadow:0 6px 18px rgba(15,23,42,.06); color:#1E293B;}
.alert.red{border-left-color:#F43F5E; background:#FFF1F2;} .alert.orange{border-left-color:#F59E0B; background:#FFFBEB;}
.legend{display:flex; gap:18px; flex-wrap:wrap; margin:10px 0 16px;}
.legend span{font-weight:600; color:#475569; font-size:.88rem;}
.dot{display:inline-block; width:12px; height:12px; border-radius:50%; margin-right:6px;}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#EEF2FF 0%,#FDF2F8 100%); border-right:1px solid #E0E7FF;}
.side-title{font-family:'Poppins',sans-serif; font-weight:800; font-size:1.6rem; line-height:1.2;
  background:linear-gradient(120deg,#4F46E5,#EC4899); -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
.side-sub{color:#64748B; font-weight:600; font-size:.85rem; margin-bottom:12px;}
.side-card{background:#fff; border-radius:16px; padding:12px 16px; margin:12px 0; font-size:.88rem; color:#334155;
  box-shadow:0 6px 16px rgba(15,23,42,.06);}
.side-card b{font-family:'Poppins',sans-serif; color:#4F46E5;}
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

        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        labels = [pretty(f) + (" (you reported)" if p else " (not reported)")
                  for f, v, p in d["drivers"]][::-1]
        vals = [v for _, v, _ in d["drivers"]][::-1]
        ax.barh(labels, vals, color=["#F43F5E" if v > 0 else "#6366F1" for v in vals], height=0.62)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.axvline(0, color="#94A3B8", lw=1)
        ax.tick_params(axis="y", length=0, labelsize=9)
        ax.tick_params(axis="x", labelsize=8, colors="#64748B")
        ax.set_xlabel("lowers the score   ←   SHAP value   →   raises the score",
                      fontsize=8.5, fontweight="bold", color="#475569")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("🟥 pushes the score up   ·   🟦 pushes it down   ·   "
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
        s2.markdown(stat_card("Active alerts", f"{n_alert} <span style='font-size:1rem;color:#F43F5E'>"
                                               f"({n_crit} critical)</span>"), unsafe_allow_html=True)
        s3.markdown(stat_card("Biggest spike", f"{top.city} · {top.disease}", small=True), unsafe_allow_html=True)
        st.write("")

    m = folium.Map(location=[22.5, 79], zoom_start=5, tiles="OpenStreetMap")
    for _, r in hot.iterrows():
        if r.city not in CITIES:
            continue
        color = "#F43F5E" if r.z >= 3 else "#F59E0B" if r.z >= 1.5 else "#10B981"
        folium.CircleMarker(CITIES[r.city], radius=8 + min(max(r.z, 0), 6) * 3,
                            color=color, weight=2, fill=True, fill_color=color, fill_opacity=.55,
                            tooltip=f"{r.city}: {r.disease}",
                            popup=f"<b>{r.city}</b><br>{r.disease}<br>Last 7 days: {r.recent} cases"
                                  f"<br>Normal week: ~{r.baseline}<br>Anomaly score: {r.z:.1f}").add_to(m)
    st_folium(m, height=520, width=1000, returned_objects=[])

    st.markdown('<div class="legend"><span><i class="dot" style="background:#F43F5E"></i>Outbreak alert</span>'
                '<span><i class="dot" style="background:#F59E0B"></i>Watch</span>'
                '<span><i class="dot" style="background:#10B981"></i>Normal</span></div>', unsafe_allow_html=True)

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