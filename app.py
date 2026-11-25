import random, re, datetime as dt
from pathlib import Path
import joblib, numpy as np, pandas as pd, shap, streamlit as st
import matplotlib.pyplot as plt, folium
from streamlit_folium import st_folium

st.set_page_config(page_title="HealthRadar", page_icon="🩺", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    background-color: #09090b !important;
    color: #e4e4e7 !important;
}

[data-testid="stHeader"] {
    background-color: #09090b !important;
}

[data-testid="stSidebar"] {
    background-color: #111115 !important;
    border-right: 1px solid #27272a !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] button {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: #d4d4d8 !important;
}

[data-testid="stSidebar"] button {
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    border: 1px solid #8b5cf6 !important;
}

[data-testid="stChatMessage"] {
    background-color: #18181b !important;
    border: 1px solid #27272a !important;
    border-left: 4px solid #7c3aed !important;
    border-radius: 12px !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    line-height: 1.6 !important;
    color: #f4f4f5 !important;
}

[data-testid="stChatInput"] textarea {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    background-color: #18181b !important;
    color: #f4f4f5 !important;
    border: 1px solid #3b82f6 !important;
}

[data-testid="stChatInput"] button {
    background-color: #dc2626 !important;
    color: #ffffff !important;
}

button[data-baseweb="tab"] {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: #a1a1aa !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom-color: #ef4444 !important;
}

[data-testid="stSelectbox"] > div > div {
    background-color: #18181b !important;
    color: #f4f4f5 !important;
    border: 1px solid #7c3aed !important;
}

[data-testid="stMetric"] {
    background-color: #18181b !important;
    border: 1px solid #27272a !important;
    border-top: 3px solid #2563eb !important;
    border-radius: 10px !important;
    padding: 10px !important;
}

h1 {
    font-size: 2.35rem !important;
    font-weight: 700 !important;
    color: #a855f7 !important;
}
</style>
""", unsafe_allow_html=True)

CITIES = {"Bengaluru":(12.97,77.59),"Mumbai":(19.08,72.88),"Delhi":(28.61,77.21),
          "Chennai":(13.08,80.27),"Kolkata":(22.57,88.36),"Hyderabad":(17.39,78.49),
          "Pune":(18.52,73.86),"Ahmedabad":(23.02,72.57),"Jaipur":(26.91,75.79),
          "Lucknow":(26.85,80.95),"Kochi":(9.93,76.27),"Mangaluru":(12.91,74.86)}
ALIASES = {"bangalore":"bengaluru","bombay":"mumbai","madras":"chennai",
           "calcutta":"kolkata","cochin":"kochi"}
SYNONYMS = {"fever":"high_fever","temperature":"high_fever","tired":"fatigue",
            "exhausted":"fatigue","throwing up":"vomiting","puking":"vomiting",
            "loose motion":"diarrhoea","stomach ache":"stomach_pain",
            "tummy":"stomach_pain","sore throat":"throat_irritation",
            "sneezing":"continuous_sneezing","body ache":"muscle_pain",
            "body pain":"muscle_pain","dizzy":"dizziness","rash":"skin_rash",
            "short of breath":"breathlessness","breathing":"breathlessness"}

@st.cache_resource
def load_model():
    b = joblib.load("model.joblib")
    return b["rf"], b["xgb"], b["le"], b["features"], shap.TreeExplainer(b["rf"])
rf, xgb, le, features, explainer = load_model()

def pretty(f): return f.replace("_", " ").strip()

# ---------- conversation helpers ----------
ss = st.session_state
def bot(t): ss.msgs.append(("assistant", t))
def init():
    ss.update(msgs=[], stage="name", name="", city="", symptoms=[], result=None)
    bot(random.choice([
        "Hey there! 👋 I'm Radar, your health sidekick. What should I call you?",
        "Hi! I'm Radar. Before we dive in, what's your name?",
        "Hello! Radar here. Tell me your first name and we'll get started."]))
if "msgs" not in ss: init()

def extract(text):
    found = {f for f in features if pretty(f) in text}
    found |= {v for k, v in SYNONYMS.items() if k in text and v in features}
    return found

def predict(symptoms):
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

def risk_label(p): return "🔴 High" if p >= .6 else "🟠 Moderate" if p >= .3 else "🟢 Low"

def log_report(city, disease):
    f = Path("data/reports.csv")
    pd.DataFrame([{"date": dt.date.today().isoformat(), "city": city,
                   "disease": disease, "cases": 1}]).to_csv(
        f, mode="a", header=not f.exists(), index=False)

def finish():
    ss.result = predict(ss.symptoms)
    log_report(ss.city, ss.result[0]["disease"])
    ss.stage = "result"
    bot(random.choice([
        f"Okay {ss.name}, I've crunched the numbers. Here's what I'm seeing 👇",
        f"Thanks for your patience, {ss.name}! Here's my read on it 👇",
        f"Done! Here's what your symptoms point to, {ss.name} 👇"]))

def handle(text):
    ss.msgs.append(("user", text)); low = text.lower().strip()
    if low in ("restart", "start over", "reset"): init(); return
    if ss.stage == "name":
        s = re.sub(r"^(?:(?:hi|hello|hey)\b[,! ]*)?(?:(?:my name is|i am|i'm|im|this is|it's)\s+)?",
                   "", text.strip(), flags=re.I)
        w = re.sub(r"[^A-Za-z]", "", s.split()[0]) if s.split() else ""
        ss.name = (w or "friend").title(); ss.stage = "city"
        bot(random.choice([f"Nice to meet you, {ss.name}! Which city are you in right now?",
                           f"Great name, {ss.name}. Where are you located? (e.g. Bengaluru, Mumbai, Chennai)",
                           f"Welcome, {ss.name}! Tell me your city so I can check what's going around."]))
    elif ss.stage == "city":
        for a, c in ALIASES.items(): low = low.replace(a, c)
        m = next((c for c in CITIES if c.lower() in low), None)
        if m:
            ss.city = m; ss.stage = "symptoms"
            bot(random.choice([f"{m}, nice! Now tell me what's bothering you, in your own words. Something like *'fever, headache and body ache'* works.",
                               f"Got it, {m}. So what symptoms are you having? Type them naturally, I'll figure it out.",
                               f"Thanks! Now, how are you feeling, {ss.name}? List whatever symptoms you have."]))
        else:
            bot(random.choice([f"Hmm, I don't have that city on my map yet. I cover: {', '.join(CITIES)}. Which is closest to you?",
                               f"Couldn't place that one. Try one of these: {', '.join(CITIES)}."]))
    elif ss.stage == "symptoms":
        done = bool(re.search(r"\b(done|that's all|thats all|that's it|that is all|nothing else|nope|nah)\b", low)) \
               or low in ("no", "nothing")
        new = [s for s in extract(low) if s not in ss.symptoms]
        if new:
            ss.symptoms += new
            if done:
                finish()
            else:
                names = ", ".join(pretty(s) for s in new)
                bot(random.choice([f"Noted: **{names}**. Anything else? Or say *'that's all'* and I'll run the analysis.",
                                   f"Got it, I've added **{names}**. More symptoms, or shall we go ahead?",
                                   f"Okay, **{names}** recorded. Tell me more, or type *done*."]))
        elif done and ss.symptoms:
            finish()
        elif done:
            bot("I'd need at least one symptom to work with, even a small one. What are you feeling?")
        else:
            bot(random.choice(["I couldn't match that to a symptom I know. Try simpler words like *cough*, *vomiting*, *chills* or *joint pain*.",
                               "Hmm, that one went over my head 😅 Could you rephrase? For example: *'high fever and headache'*."]))
    else:
        bot("Type **restart** whenever you want to check another set of symptoms.")

def explain_text(d):
    ups = [pretty(f) for f, v, p in d["drivers"] if v > 0 and p == 1]
    if not ups:
        return "None of your reported symptoms strongly pointed here. The score mostly reflects other typical symptoms being absent."
    s = f"The biggest push toward **{d['disease']}** came from **{ups[0]}**"
    return s + (f", backed up by **{ups[1]}**." if len(ups) > 1 else ".")

def render_result(res):
    with st.chat_message("assistant"):
        cols = st.columns(3)
        for c, d in zip(cols, res):
            c.metric(d["disease"], f"{d['prob']*100:.0f}%", risk_label(d["prob"]))
        pick = st.selectbox("Why this score? Pick a condition:", [d["disease"] for d in res])
        d = next(x for x in res if x["disease"] == pick)
        st.write(explain_text(d))
        fig, ax = plt.subplots(figsize=(6, 3))
        labels = [pretty(f) + (" (you reported)" if p else " (not reported)")
                  for f, v, p in d["drivers"]][::-1]
        vals = [v for _, v, _ in d["drivers"]][::-1]
        ax.barh(labels, vals, color=["#d62728" if v > 0 else "#1f77b4" for v in vals])
        ax.axvline(0, color="k", lw=.8); ax.set_xlabel("SHAP value (right = pushes score up)")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.caption("⚠️ Educational project, not medical advice. Please see a doctor for real concerns.")

# ---------- map helpers ----------
def load_regional():
    df = pd.read_csv("data/regional_daily.csv", parse_dates=["date"])
    shift = pd.Timestamp.today().normalize() - df["date"].max().normalize()
    df["date"] = df["date"] + shift
    f = Path("data/reports.csv")
    if f.exists(): df = pd.concat([df, pd.read_csv(f, parse_dates=["date"])])
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

# ---------- layout ----------
with st.sidebar:
    st.title("🩺 HealthRadar")
    st.write("Predictive disease risk + outbreak visualizer.")
    if st.button("Start over"): init(); st.rerun()
    st.caption("Regional feed is simulated. Symptom model trained on a public dataset.")

st.info("⚠️ HealthRadar is an educational project, not a medical device, and it cannot diagnose illness. "
        "If you feel seriously unwell or have an emergency (chest pain, trouble breathing, severe bleeding), "
        "call **911** or go to the nearest hospital.")    

tab1, tab2 = st.tabs(["💬 Chat with Radar", "🗺️ Outbreak map"])

with tab1:
    for role, text in ss.msgs:
        with st.chat_message(role): st.markdown(text)
    if ss.stage == "result": render_result(ss.result)
    prompt = st.chat_input("Type here...")
    if prompt:
        handle(prompt); st.rerun()

with tab2:
    st.warning("🧪 Demo data: regional case counts on this map are simulated for demonstration. They are not real outbreak reports.")
    df = load_regional()
    choice = st.selectbox("Disease", ["All diseases"] + sorted(df["disease"].unique()))
    a = anomalies(df)
    if choice != "All diseases": a = a[a["disease"] == choice]
    hot = a.sort_values("z", ascending=False).groupby("city").head(1)
    m = folium.Map(location=[22.5, 79], zoom_start=5, tiles="OpenStreetMap")
    for _, r in hot.iterrows():
        color = "red" if r.z >= 3 else "orange" if r.z >= 1.5 else "green"
        folium.CircleMarker(CITIES[r.city], radius=8 + min(max(r.z, 0), 6) * 3,
            color=color, fill=True, fill_opacity=.6, tooltip=f"{r.city}: {r.disease}",
            popup=f"<b>{r.city}</b><br>{r.disease}<br>Last 7 days: {r.recent} cases"
                  f"<br>Normal week: ~{r.baseline}<br>Anomaly score: {r.z:.1f}").add_to(m)
    st_folium(m, height=520, width=1000, returned_objects=[])
    for _, r in hot.sort_values("z", ascending=False).head(3).iterrows():
        if r.z >= 1.5:
            st.warning(f"**{r.city}**: {r.disease} is running about {r.recent/max(r.baseline,1):.1f}x its usual weekly level.")