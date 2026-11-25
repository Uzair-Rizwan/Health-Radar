import datetime as dt
from pathlib import Path

import joblib, numpy as np, pandas as pd, shap, streamlit as st
import matplotlib.pyplot as plt, folium
from streamlit_folium import st_folium

from chat_engine import Chat, CITIES

st.set_page_config(page_title="HealthRadar", page_icon="🩺", layout="wide")


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


def risk_label(p):
    return "🔴 High" if p >= .6 else "🟠 Moderate" if p >= .3 else "🟢 Low"


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


def explain_text(d):
    ups = [pretty(f) for f, v, p in d["drivers"] if v > 0 and p == 1]
    if not ups:
        return ("None of your reported symptoms strongly pointed here. "
                "The score mostly reflects other typical symptoms being absent.")
    s = f"The biggest push toward **{d['disease']}** came from **{ups[0]}**"
    return s + (f", backed up by **{ups[1]}**." if len(ups) > 1 else ".")


def render_result(res, key, latest):
    with st.chat_message("assistant", avatar="🩺"):
        cols = st.columns(3)
        for c, d in zip(cols, res):
            c.metric(d["disease"], f"{d['prob'] * 100:.0f}%", risk_label(d["prob"]))
        pick = st.selectbox("Why this score? Pick a condition:", [d["disease"] for d in res],
                            key=f"pick_{key}")
        d = next(x for x in res if x["disease"] == pick)
        st.write(explain_text(d))
        fig, ax = plt.subplots(figsize=(6, 3))
        labels = [pretty(f) + (" (you reported)" if p else " (not reported)")
                  for f, v, p in d["drivers"]][::-1]
        vals = [v for _, v, _ in d["drivers"]][::-1]
        ax.barh(labels, vals, color=["#d62728" if v > 0 else "#1f77b4" for v in vals])
        ax.axvline(0, color="k", lw=.8)
        ax.set_xlabel("SHAP value (right = pushes score up)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("⚠️ Educational project, not medical advice. Please see a doctor for real concerns.")
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


# ---------------------------------------------------------------- layout
with st.sidebar:
    st.title("🩺 HealthRadar")
    st.write("Predictive disease risk + outbreak visualizer.")
    if st.button("Start over"):
        chat.reset()
        st.rerun()
    st.caption("Regional feed is simulated. Symptom model trained on a public dataset.")

st.info("⚠️ HealthRadar is an educational project, not a medical device, and it cannot diagnose illness. "
        "If you feel seriously unwell or have an emergency (chest pain, trouble breathing, severe bleeding), "
        "call **112** or go to the nearest hospital.")

tab1, tab2 = st.tabs(["💬 Chat with Radar", "🗺️ Outbreak map"])

with tab1:
    last_result = max([i for i, (r, _) in enumerate(chat.msgs) if r == "result"], default=-1)
    for i, (role, content) in enumerate(chat.msgs):
        if role == "result":
            render_result(content, i, latest=(i == last_result))
        else:
            with st.chat_message(role, avatar="🩺" if role == "assistant" else None):
                st.markdown(content)
    prompt = st.chat_input("Type here... (spelling mistakes are fine)")
    if prompt:
        with st.spinner("Radar is thinking..."):
            chat.handle(prompt)
        st.rerun()

with tab2:
    st.warning("🧪 Demo data: regional case counts on this map are simulated for demonstration. "
               "They are not real outbreak reports.")
    df = load_regional()
    choice = st.selectbox("Disease", ["All diseases"] + sorted(df["disease"].unique()))
    a = anomalies(df)
    if choice != "All diseases":
        a = a[a["disease"] == choice]
    hot = a.sort_values("z", ascending=False).groupby("city").head(1)
    m = folium.Map(location=[22.5, 79], zoom_start=5, tiles="OpenStreetMap")
    for _, r in hot.iterrows():
        if r.city not in CITIES:
            continue
        color = "red" if r.z >= 3 else "orange" if r.z >= 1.5 else "green"
        folium.CircleMarker(CITIES[r.city], radius=8 + min(max(r.z, 0), 6) * 3,
                            color=color, fill=True, fill_opacity=.6, tooltip=f"{r.city}: {r.disease}",
                            popup=f"<b>{r.city}</b><br>{r.disease}<br>Last 7 days: {r.recent} cases"
                                  f"<br>Normal week: ~{r.baseline}<br>Anomaly score: {r.z:.1f}").add_to(m)
    st_folium(m, height=520, width=1000, returned_objects=[])
    for _, r in hot.sort_values("z", ascending=False).head(3).iterrows():
        if r.z >= 1.5:
            st.warning(f"**{r.city}**: {r.disease} is running about "
                       f"{r.recent / max(r.baseline, 1):.1f}x its usual weekly level.")