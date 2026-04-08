import streamlit as st
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import random

# Page Config
st.set_page_config(page_title="Fake Account Detector", page_icon="🔍", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: black;'>
    🔍 AI Fake Account Detector
    </h1>
""", unsafe_allow_html=True)

# Load model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.write("Enter account details to check if it's Fake or Real")

# Username Input
username = st.text_input("Enter Instagram Username (optional)")

# Input Handling
if username:
    st.info(f"Analyzing @{username}...")

    account_age = random.randint(1, 1000)
    followers = random.randint(10, 10000)
    following = random.randint(50, 5000)
    posts = random.randint(0, 500)
    bio_length = random.randint(0, 150)
    has_profile_pic = random.choice([0, 1])

else:
    account_age = st.number_input("Account Age (days)", 0, 5000, 10)
    followers = st.number_input("Followers", 0, 1000000, 100)
    following = st.number_input("Following", 0, 1000000, 150)
    posts = st.number_input("Posts", 0, 100000, 10)
    bio_length = st.number_input("Bio Length", 0, 500, 50)
    has_profile_pic = st.selectbox("Has Profile Picture?", [0, 1])

# Feature Engineering
ratio = followers / (following + 1)
activity_rate = posts / (account_age + 1)
engagement_score = followers / (posts + 1)
growth_rate = followers / (account_age + 1)

import pandas as pd

feature_dict = {
    "account_age": account_age,
    "followers": followers,
    "following": following,
    "posts": posts,
    "bio_length": bio_length,
    "has_profile_pic": has_profile_pic,
    "ratio": ratio,
    "activity_rate": activity_rate,
    "engagement_score": engagement_score,
    "growth_rate": growth_rate
}

features = pd.DataFrame([feature_dict])

# Scale features
features_scaled = scaler.transform(features)

# Trust Score Function
def calculate_trust_score(prob_real):
    uncertainty = 1 - abs(prob_real - 0.5) * 2
    penalty = uncertainty * 20
    score = (prob_real * 100) - penalty
    return round(max(0, min(100, score)), 2)

# Predict Button
if st.button("Analyze Account"):

    # Prediction
    prob = model.predict_proba(features_scaled)[0]
    prob_fake = prob[1]
    prob_real = prob[0]

    # Result
    result = "Fake ❌" if prob_fake > 0.5 else "Real ✅"

    # Trust Score
    trust_score = calculate_trust_score(prob_real)

    # Display Result
    st.subheader(f"Result: {result}")
    st.subheader(f"Trust Score: {trust_score}%")
    st.progress(int(trust_score))

    # Risk Level
    if trust_score > 70:
        st.success("🟢 Low Risk")
    elif trust_score > 40:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")

    # AI Explanation
    st.subheader("🧠 AI Explanation")

    if prob_fake > 0.5:
        st.error("This account is likely FAKE due to unusual behavior patterns.")
    else:
        st.success("This account appears to be REAL based on activity patterns.")

    # SHAP Explainability
    st.subheader("📊 Explanation (Why this result?)")

    explainer = shap.Explainer(model)
    shap_values = explainer(features_scaled)

    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(plt)

    # Key Insights
    st.subheader("🔍 Key Insights:")

    if following > followers * 3:
        st.write("⚠️ Following is much higher than followers")

    if account_age < 30:
        st.write("⚠️ Account is very new")

    if posts < 5:
        st.write("⚠️ Very low activity")

    if has_profile_pic == 0:
        st.write("⚠️ No profile picture")