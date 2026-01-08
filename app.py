# app.py
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="FitPulse – Health Analytics",
    page_icon="❤️",
    layout="wide"
)

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.title("💓 FitPulse")
st.sidebar.caption("End-to-End Health Analytics Pipeline")

menu = st.sidebar.radio(
    "App Flow",
    [
        "Data Collection",
        "Feature Extraction",
        "Anomaly Detection",
        "Dashboard",
        "Reports",
        "About"
    ]
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "metrics" not in st.session_state:
    st.session_state.metrics = []

# ==================================================
# 1️⃣ DATA COLLECTION
# ==================================================
if menu == "Data Collection":
    st.title("📂 Data Collection")

    file = st.file_uploader("Upload CSV Dataset", type=["csv"])

    if file:
        df = pd.read_csv(file)

        if "timestamp" not in df.columns:
            st.error("CSV must contain a 'timestamp' column")
            st.stop()

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        metrics = df.select_dtypes(include="number").columns.tolist()

        if not metrics:
            st.error("No numeric health metrics found")
            st.stop()

        st.session_state.df = df
        st.session_state.metrics = metrics

        st.success("✅ Data loaded successfully")
        st.write("Detected Metrics:", metrics)
        st.dataframe(df.head())

# ==================================================
# 2️⃣ FEATURE EXTRACTION
# ==================================================
elif menu == "Feature Extraction":
    st.title("⚙️ Feature Extraction")

    if st.session_state.df is None:
        st.warning("Upload data first")
    else:
        df = st.session_state.df
        metric = st.selectbox("Select Metric", st.session_state.metrics)

        df["rolling_mean"] = df[metric].rolling(10, min_periods=1).mean()
        df["rolling_std"] = df[metric].rolling(10, min_periods=1).std()
        df["diff"] = df[metric].diff()

        st.session_state.df = df

        st.success("✅ Features generated")
        st.dataframe(df[[metric, "rolling_mean", "rolling_std", "diff"]].head())

# ==================================================
# 3️⃣ ANOMALY DETECTION (MAIN ML PIPELINE)
# ==================================================
elif menu == "Anomaly Detection":
    st.title("🚨 Anomaly Detection")

    if st.session_state.df is None:
        st.warning("Complete previous steps first")
    else:
        df = st.session_state.df.copy()
        metric = st.selectbox("Select Metric", st.session_state.metrics)

        method = st.radio(
            "Select Algorithm",
            ["Statistical", "K-Means", "DBSCAN", "Isolation Forest"]
        )

        X = df[[metric]].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ---- Statistical (Z-score)
        if method == "Statistical":
            mean = X[metric].mean()
            std = X[metric].std()
            X["anomaly"] = abs(X[metric] - mean) > 3 * std

        # ---- K-Means
        elif method == "K-Means":
            kmeans = KMeans(n_clusters=2, random_state=42)
            X["cluster"] = kmeans.fit_predict(X_scaled)
            anomaly_cluster = X["cluster"].value_counts().idxmin()
            X["anomaly"] = X["cluster"] == anomaly_cluster

        # ---- DBSCAN
        elif method == "DBSCAN":
            dbscan = DBSCAN(eps=0.8, min_samples=10)
            labels = dbscan.fit_predict(X_scaled)
            X["anomaly"] = labels == -1

        # ---- Isolation Forest (NEW & UNIQUE)
        elif method == "Isolation Forest":
            iso = IsolationForest(contamination=0.05, random_state=42)
            preds = iso.fit_predict(X_scaled)
            X["anomaly"] = preds == -1

        anomalies = X[X["anomaly"]]

        fig = px.line(df, x="timestamp", y=metric, title=f"{method} Detection")
        fig.add_scatter(
            x=df.loc[anomalies.index, "timestamp"],
            y=df.loc[anomalies.index, metric],
            mode="markers",
            marker=dict(color="red", size=8),
            name="Anomaly"
        )

        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Detected Anomalies")
        st.dataframe(df.loc[anomalies.index])

# ==================================================
# 4️⃣ DASHBOARD
# ==================================================
elif menu == "Dashboard":
    st.title("📊 Dashboard Overview")

    if st.session_state.df is None:
        st.warning("Upload data first")
    else:
        df = st.session_state.df
        metric = st.selectbox("Metric", st.session_state.metrics)

        c1, c2, c3 = st.columns(3)
        c1.metric("Records", len(df))
        c2.metric("Average", round(df[metric].mean(), 2))
        c3.metric("Max", round(df[metric].max(), 2))

        fig = px.line(df, x="timestamp", y=metric)
        st.plotly_chart(fig, use_container_width=True)

# ==================================================
# 5️⃣ REPORTS
# ==================================================
elif menu == "Reports":
    st.title("📑 Reports")

    if st.session_state.df is None:
        st.warning("No data available")
    else:
        st.download_button(
            "⬇ Download CSV Report",
            st.session_state.df.to_csv(index=False),
            "fitpulse_report.csv",
            "text/csv"
        )

# ==================================================
# 6️⃣ ABOUT
# ==================================================
elif menu == "About":
    st.title("ℹ️ About")

    st.markdown("""
    **FitPulse** is an end-to-end health analytics pipeline.

    **Pipeline Flow**
    - Data Collection
    - Feature Engineering
    - Multi-model Anomaly Detection
    - Interactive Dashboard

    **Algorithms Used**
    - Statistical (Z-score)
    - K-Means
    - DBSCAN
    - Isolation Forest

    **Tech Stack**
    - Python, Pandas
    - Streamlit
    - Plotly
    - Scikit-learn
    """)
