import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Multi-Model Analytics Dashboard",
    layout="wide"
)

st.title("📊 Multi-Algorithm Prediction Dashboard")

# -----------------------------
# MODEL PATHS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

LOGISTIC_MODEL_PATH = BASE_DIR / "reg_model.pkl"
KMEANS_MODEL_PATH   = BASE_DIR / "classifier_model (2).pkl"
ARM_MODEL_PATH      = BASE_DIR / "ARM model.pkl"

# -----------------------------
# SAFE MODEL LOADER
# -----------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

# -----------------------------
# SIDEBAR MENU
# -----------------------------
st.sidebar.title("⚙️ Select Prediction Type")

option = st.sidebar.radio(
    "Choose Model",
    [
        "Logistic Regression Prediction",
        "K-Means Clustering Prediction",
        "Association Rule Mining Prediction"
    ]
)

# ======================================================
# 1️⃣ LOGISTIC REGRESSION PAGE
# ======================================================
if option == "Logistic Regression Prediction":

    st.subheader("✅ Logistic Regression – Cross Sell Prediction")

    model = load_model(LOGISTIC_MODEL_PATH)

    st.info("Enter only relevant customer attributes used in the final model.")

    # 👉 Adjust these if your real features differ
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education", ["Graduate", "Post Graduate", "Other"])
    occupation = st.selectbox("Occupation", ["Salaried", "Business", "Student", "Other"])

    # Simple encoding (change if your training encoding differs)
    gender_map = {"Male": 1, "Female": 0}
    edu_map = {"Graduate": 1, "Post Graduate": 2, "Other": 0}
    occ_map = {"Salaried": 1, "Business": 2, "Student": 3, "Other": 0}

    input_data = np.array([[
        gender_map[gender],
        edu_map[education],
        occ_map[occupation]
    ]])

    if st.button("Predict"):

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success("✅ Customer is likely to ACCEPT cross-sell offer.")
        else:
            st.warning("❌ Customer is NOT likely to accept cross-sell offer.")

# ======================================================
# 2️⃣ K-MEANS CLUSTERING PAGE
# ======================================================
elif option == "K-Means Clustering Prediction":

    st.subheader("📌 K-Means – Customer Cluster Identification")

    model = load_model(KMEANS_MODEL_PATH)

    st.info("Provide same features used while training clustering model.")

    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education", ["Graduate", "Post Graduate", "Other"])
    occupation = st.selectbox("Occupation", ["Salaried", "Business", "Student", "Other"])

    gender_map = {"Male": 1, "Female": 0}
    edu_map = {"Graduate": 1, "Post Graduate": 2, "Other": 0}
    occ_map = {"Salaried": 1, "Business": 2, "Student": 3, "Other": 0}

    input_data = np.array([[
        gender_map[gender],
        edu_map[education],
        occ_map[occupation]
    ]])

    if st.button("Predict Cluster"):

        cluster = model.predict(input_data)[0]

        st.success(f"🎯 Customer belongs to Cluster: **{cluster}**")

        st.caption("Clusters can be used for segmentation, targeting, and personalization strategies.")

# ======================================================
# 3️⃣ ASSOCIATION RULE MINING PAGE
# ======================================================
elif option == "Association Rule Mining Prediction":

    st.subheader("🔗 Association Rule Mining – Recommendation Engine")

    rules_df = load_model(ARM_MODEL_PATH)   # Usually ARM model is saved as DataFrame

    st.info("Select customer attributes to find matching association rules.")

    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education", ["Graduate", "Post Graduate", "Other"])
    occupation = st.selectbox("Occupation", ["Salaried", "Business", "Student", "Other"])

    if st.button("Generate Recommendation"):

        # Example filtering logic (adjust column names if needed)
        filtered_rules = rules_df[
            rules_df["antecedents"].astype(str).str.contains(gender, case=False) |
            rules_df["antecedents"].astype(str).str.contains(education, case=False) |
            rules_df["antecedents"].astype(str).str.contains(occupation, case=False)
        ]

        if len(filtered_rules) == 0:
            st.warning("⚠️ No matching rules found for selected inputs.")
        else:
            st.success("✅ Recommended Rules Found")
            st.dataframe(
                filtered_rules[["antecedents", "consequents", "confidence", "lift"]]
                .sort_values(by="confidence", ascending=False)
                .head(10)
            )

            st.caption("These rules help identify product bundling and cross-selling opportunities.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("🚀 Built for Analytics & Decision Intelligence")
