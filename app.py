from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/win_predictor_v2.joblib")
BALLS_PER_INNINGS = 120

st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="wide")
st.title("IPL Win Predictor")

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

bundle = load_bundle()
model, features, venues = bundle["model"], bundle["features"], bundle["venues"]

with st.sidebar:
    st.header("Match Details")
    venue = st.selectbox("Select Venue", venues)
    target = st.number_input("Target Score", 1, 300, 180)
    runs_needed = st.number_input("Runs Needed", 1, 300, 60)
    overs_left = st.number_input("Overs Remaining", 1, 20, 6)
    wickets_in_hand = st.number_input("Wickets in Hand", 1, 10, 6)

st.write("Enter the match details in the sidebar and click Predict.")

if st.button("Predict Win Probability", type="primary"):
    balls_remaining = int(overs_left * 6)
    balls_bowled = BALLS_PER_INNINGS - balls_remaining
    runs_scored = target - runs_needed

    if runs_scored < 0:
        st.error("Runs Needed cannot exceed the Target."); st.stop()
    if balls_bowled > 0 and runs_scored > balls_bowled * 6 + 10:
        st.error(f"Impossible: {runs_scored} runs cannot come from {balls_bowled} balls."); st.stop()

    current_rr = runs_scored / (balls_bowled / 6) if balls_bowled else 0.0
    required_rr = runs_needed / (balls_remaining / 6)

    X = pd.DataFrame([{
        "venue": venue, "target": target, "runs_needed": runs_needed,
        "balls_remaining": balls_remaining, "wickets_in_hand": wickets_in_hand,
        "current_run_rate": current_rr, "required_run_rate": required_rr,
        "rr_pressure": required_rr - current_rr,
    }])[features]

    win_prob = model.predict_proba(X)[0, 1]

    st.subheader("Prediction Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Win Probability (Chasing Team)", f"{win_prob:.1%}")
    c2.metric("Current Run Rate", f"{current_rr:.2f}")
    c3.metric("Required Run Rate", f"{required_rr:.2f}")
    st.progress(min(max(win_prob, 0.0), 1.0))
    st.caption(f"{runs_needed} needed off {balls_remaining} balls at {venue} "
               f"({wickets_in_hand} wickets in hand)")
    if current_rr > 15:
        st.warning("Very high current run rate — rare in training data; treat with caution.")

st.markdown("---")
st.caption(f"Model: {bundle['name']} | 1,100+ IPL matches (Cricsheet) | match-level validation.")
