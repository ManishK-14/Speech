import streamlit as st
import json
import pandas as pd

st.set_page_config(page_title="AI Speaking Coach", page_icon="🎙️", layout="wide")

st.title("🎙️ AI Public Speaking Coach Dashboard")
st.markdown("Analyze your speech pace, vocal tone, and clarity.")

try:
    with open("report.json", "r") as f:
        report = json.load(f)
except FileNotFoundError:
    st.error("Report not found! Please run your analyze script first to generate 'report.json'.")
    st.stop()

st.subheader("Key Performance Metrics")
col1, col2, col3, col4 = st.columns(4)

metrics = report["metrics"]
col1.metric("Pace (WPM)", f"{metrics['wpm']} wpm")
col2.metric("Filler Words", metrics["total_fillers"])
col3.metric("Pauses", metrics["total_pauses"])
col4.metric("Pitch Variation", f"{metrics['pitch_variation_cv']}%")

st.divider()

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("💡 Coaching Feedback")
    for tip in report["feedback"]:
        if "Excellent" in tip or "Good" in tip or "Expressive" in tip:
            st.success(tip)
        else:
            st.warning(tip)

    st.subheader("📝 Transcript Snippet")
    st.info(report["transcription"][:500] + "...")

with right_col:
    st.subheader("📊 Filler Word Breakdown")
    fillers = metrics.get("filler_breakdown", {})
    if fillers:
        df_fillers = pd.DataFrame(list(fillers.items()), columns=["Word", "Count"])
        st.bar_chart(df_fillers.set_index("Word"))
    else:
        st.write("Awesome! No major filler words detected.")

    st.subheader("🔊 Audio Analysis")
    st.image("analysis_plot.png", caption="Waveform, Pitch, and Energy", use_container_width=True)