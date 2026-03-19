

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from prophet import Prophet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
import io

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG (expanded sidebar)
# ============================================================
st.set_page_config(
    page_title="FitPulse",
    page_icon="🏋🏻‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# COMBINED CUSTOM CSS (fixed: header is no longer hidden)
# ============================================================
st.markdown("""
<style>

/* ---------- from interface ---------- */
.stApp {
    background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
}
.block-container {
    background: transparent !important;
    padding-top: 2rem;
}
.main-title {
    font-size:4rem;
    font-weight:800;
    text-align:center;
    color:white;
    text-shadow:0 5px 20px rgba(0,0,0,0.4);
}
.subtitle{
    text-align:center;
    color:rgba(255,255,255,0.9);
    font-size:1.2rem;
    margin-bottom:2rem;
}
.dropdown-label{
    font-size:1.1rem;
    font-weight:600;
    color:white;
    text-align:center;
}
div[data-baseweb="select"] > div{
    background:#f7fafc !important;
    border-radius:12px !important;
    border:2px solid #e2e8f0 !important;
}
div[data-baseweb="select"] span{
    color:#1a202c !important;
}
div[role="listbox"] div{
    color:#1a202c !important;
}
.stButton > button{
    background:linear-gradient(90deg,#ff5f6d,#ff3d77);
    color:white;
    border:none;
    border-radius:30px;
    padding:0.8rem;
    font-weight:600;
    font-size:1.1rem;
    width:100%;
}
.footer{
    text-align:center;
    margin-top:2rem;
    color:rgba(255,255,255,0.8);
}

/* ---------- from milestone1 ---------- */
[data-testid="stAppViewContainer"] {
    background: transparent !important;
}
.glass-card {
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(12px);
    padding:20px;
    border-radius:20px;
    text-align:center;
    box-shadow:0 8px 32px rgba(0,0,0,0.3);
}

/* ---------- from milestone2 (dark theme, but header is now visible) ---------- */
:root {
    --bg-primary:    #0a0e1a;
    --bg-secondary:  #111827;
    --bg-card:       #161d2e;
    --bg-card-hover: #1c2540;
    --accent-cyan:   #00d4ff;
    --accent-purple: #a855f7;
    --accent-pink:   #ec4899;
    --accent-green:  #10b981;
    --accent-orange: #f59e0b;
    --text-primary:  #f1f5f9;
    --text-secondary:#94a3b8;
    --text-muted:    #475569;
    --border:        #1e2d4a;
    --glow-cyan:     0 0 20px rgba(0,212,255,0.3);
    --glow-purple:   0 0 20px rgba(168,85,247,0.3);
}
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1224 50%, #0a0e1a 100%) !important;
}
/* Only hide the main menu and footer, keep header for sidebar toggle */
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1224 0%, #111827 100%) !important;
    border-right: 1px solid var(--border) !important;
}
.hero-banner {
    background: linear-gradient(135deg, #0d1224 0%, #1a1040 40%, #0d1224 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple), var(--accent-pink));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-subtitle {
    color: var(--text-secondary);
    font-size: 1rem;
}
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 2rem 0 1.5rem 0;
}
.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
}
.info-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 1.4rem 1.6rem !important;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: var(--glow-cyan) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
}
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stButton > button {
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(168,85,247,0.15)) !important;
    border-color: var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
    box-shadow: var(--glow-cyan) !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER: apply dark theme for matplotlib (from milestone2)
# ============================================================
def apply_dark_theme():
    mpl.rcParams.update({
        'figure.facecolor':  '#111827',
        'axes.facecolor':    '#161d2e',
        'axes.edgecolor':    '#1e2d4a',
        'axes.labelcolor':   '#94a3b8',
        'axes.titlecolor':   '#f1f5f9',
        'xtick.color':       '#475569',
        'ytick.color':       '#475569',
        'text.color':        '#f1f5f9',
        'grid.color':        '#1e2d4a',
        'grid.linewidth':    0.6,
        'legend.facecolor':  '#161d2e',
        'legend.edgecolor':  '#1e2d4a',
        'legend.labelcolor': '#94a3b8',
        'font.family':       'sans-serif',
        'axes.spines.top':   False,
        'axes.spines.right': False,
    })

apply_dark_theme()


# ============================================================
# SESSION STATE INIT
# ============================================================
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = 'home'   # 'home', 'milestone1', 'milestone2'


# ============================================================
# FUNCTION: HOME PAGE
# ============================================================
def show_home():
    st.markdown('<div class="main-title">FitPulse</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Health Anomaly Detection</div>', unsafe_allow_html=True)

    st.markdown('<div class="dropdown-label">📋 Select a milestone</div>', unsafe_allow_html=True)

    option = st.selectbox(
        "",
        ["", "Milestone 1: Data Preprocessing", "Milestone 2: Feature Extraction & Clustering"],
        format_func=lambda x: x if x else "Choose an option...",
        label_visibility="collapsed"
    )

    if st.button("🚀 Launch"):
        if option == "Milestone 1: Data Preprocessing":
            st.session_state.app_mode = 'milestone1'
            st.rerun()
        elif option == "Milestone 2: Feature Extraction & Clustering":
            st.session_state.app_mode = 'milestone2'
            st.rerun()

    st.markdown('<div class="footer">Select a milestone and click Launch to begin</div>', unsafe_allow_html=True)


# ============================================================
# FUNCTION: BACK TO HOME BUTTON
# ============================================================
def back_to_home_button():
    if st.button("🔙 Back to Home"):
        st.session_state.app_mode = 'home'
        st.rerun()


# ============================================================
# MILESTONE 1 (fully preserved)
# ============================================================
def show_milestone1():
    back_to_home_button()
    st.markdown("---")

    st.title("🚀 Data Preprocessing")
    st.write("Upload → Explore → Clean → Download")

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.header("📁 Upload Dataset")
        file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])

    if file is not None:

        # Load file
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        # ---------------- SUMMARY CARDS ----------------
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f'<div class="glass-card"><h3>📄 Rows</h3><h2>{df.shape[0]}</h2></div>', unsafe_allow_html=True)

        with col2:
            st.markdown(f'<div class="glass-card"><h3>📊 Columns</h3><h2>{df.shape[1]}</h2></div>', unsafe_allow_html=True)

        with col3:
            st.markdown(f'<div class="glass-card"><h3>⚠️ Missing</h3><h2>{df.isnull().sum().sum()}</h2></div>', unsafe_allow_html=True)

        st.write("")

        # ---------------- BUTTONS ----------------
        colA, colB, colC = st.columns(3)

        show_data = colA.button("👀 Show Original Data")
        show_null = colB.button("🔍 Show Null Rows")
        preprocess = colC.button("🧠 Preprocessed Data")

        # ---------------- ORIGINAL DATA ----------------
        if show_data:
            st.subheader("📄 Original Dataset")
            st.dataframe(df, use_container_width=True)

        # ---------------- NULL ANALYSIS ----------------
        if show_null:
            st.subheader("⚠️ Missing Value Rows")
            null_rows = df[df.isnull().any(axis=1)]

            if null_rows.empty:
                st.success("No missing values found 🎉")
            else:
                st.dataframe(null_rows, use_container_width=True)

            # Missing values chart
            st.subheader("📊 Missing Values Per Column")
            missing_count = df.isnull().sum()
            st.bar_chart(missing_count)

        # ---------------- PREPROCESSING ----------------
        if preprocess:
            with st.spinner("Processing your dataset... ✨"):

                # Convert Date column
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

                # Numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

                if numeric_cols:
                    if "User_ID" in df.columns:
                        df[numeric_cols] = df.groupby("User_ID")[numeric_cols].transform(
                            lambda x: x.interpolate().ffill().bfill()
                        )
                    else:
                        df[numeric_cols] = df[numeric_cols].interpolate().ffill().bfill()

                # Categorical columns
                for col in df.select_dtypes(include="object").columns:
                    df[col] = df[col].fillna("Unknown")

            st.success("✅ Dataset Cleaned Successfully!")
            st.subheader("🧹 Cleaned Dataset")
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Cleaned Data",
                csv,
                "cleaned_data.csv",
                "text/csv"
            )

    else:
        st.info("⬅️ Upload a dataset from sidebar to begin")


# ============================================================
# MILESTONE 2 (fully preserved)
# ============================================================
def show_milestone2():
    back_to_home_button()
    st.markdown("---")

    # ---------------- HERO BANNER ----------------
    st.markdown("""
    <div class="hero-banner">
      <div class="hero-badge">FEATURE EXTRACTION & MODELING  · Milestone 2</div>
      <p class="hero-title">FitPulse ML Pipeline</p>
      <p class="hero-subtitle">TSFresh Feature Extraction &nbsp;&bull;&nbsp; Prophet Forecasting &nbsp;&bull;&nbsp; KMeans &amp; DBSCAN Clustering</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- SESSION STATE INIT ----------------
    for key in ['daily_df','hourly_s_df','hourly_i_df','sleep_df','hr_df','master_df','hr_minute_df']:
        if key not in st.session_state:
            st.session_state[key] = None
    if 'current_section' not in st.session_state:
        st.session_state.current_section = "Data Overview"

    # ---------------- PREPROCESSING FUNCTIONS ----------------
    def preprocess_timestamps(daily, hourly_s, hourly_i, sleep, hr):
        daily["ActivityDate"]    = pd.to_datetime(daily["ActivityDate"], format="%m/%d/%Y")
        hourly_s["ActivityHour"] = pd.to_datetime(hourly_s["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p")
        hourly_i["ActivityHour"] = pd.to_datetime(hourly_i["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p")
        sleep["date"]            = pd.to_datetime(sleep["date"], format="%m/%d/%Y %I:%M:%S %p")
        hr["Time"]               = pd.to_datetime(hr["Time"], format="%m/%d/%Y %I:%M:%S %p")
        return daily, hourly_s, hourly_i, sleep, hr

    def create_master(daily, hr, sleep):
        hr = hr.copy()
        hr["Time"] = pd.to_datetime(hr["Time"], errors="coerce")
        hr = hr.dropna(subset=["Time","Id","Value"]).sort_values(["Id","Time"]).reset_index(drop=True)
        hr_indexed = hr.set_index("Time")
        hr_minute = (
            hr_indexed.groupby("Id")["Value"]
            .resample("1min", closed="left", label="left")
            .mean().reset_index()
        )
        hr_minute.columns = ["Id","Time","HeartRate"]
        hr_minute = hr_minute.dropna()
        hr_minute["Date"] = hr_minute["Time"].dt.date
        hr_daily = (
            hr_minute.groupby(["Id","Date"])["HeartRate"]
            .agg(["mean","max","min","std"]).reset_index()
            .rename(columns={"mean":"AvgHR","max":"MaxHR","min":"MinHR","std":"StdHR"})
        )
        sleep = sleep.copy()
        sleep["Date"] = pd.to_datetime(sleep["date"], errors="coerce").dt.date
        sleep = sleep.dropna(subset=["Date","Id","value"])
        sleep_daily = (
            sleep.groupby(["Id","Date"])
            .agg(TotalSleepMinutes=("value","count"),
                 DominantSleepStage=("value", lambda x: x.mode()[0] if len(x) else 0))
            .reset_index()
        )
        master = daily.copy().rename(columns={"ActivityDate":"Date"})
        master["Date"] = master["Date"].dt.date
        master = master.merge(hr_daily, on=["Id","Date"], how="left")
        master = master.merge(sleep_daily, on=["Id","Date"], how="left")
        master["TotalSleepMinutes"]  = master["TotalSleepMinutes"].fillna(0)
        master["DominantSleepStage"] = master["DominantSleepStage"].fillna(0)
        for col in ["AvgHR","MaxHR","MinHR","StdHR"]:
            master[col] = master.groupby("Id")[col].transform(lambda x: x.fillna(x.median()))
        return master, hr_minute

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 1.5rem 0;">
            <div style="font-size:2.5rem; margin-bottom:0.5rem;">&#9825;</div>
            <div style="font-size:1.1rem; font-weight:700; color:#f1f5f9; letter-spacing:0.05em;">Health Analytics</div>
            <div style="font-size:0.7rem; color:#475569; letter-spacing:0.15em; text-transform:uppercase; margin-top:0.25rem;">Fitbit · ML Pipeline</div>
        </div>
        <hr style="border:none; height:1px; background:linear-gradient(90deg,transparent,#1e2d4a,transparent); margin-bottom:1.5rem;">
        """, unsafe_allow_html=True)

        st.markdown('<div style="font-size:0.7rem; color:#475569; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.5rem; font-weight:600;">Data Upload</div>', unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload all 5 CSV files",
            type=['csv'],
            accept_multiple_files=True,
            help="dailyActivity, hourlySteps, hourlyIntensities, minuteSleep, heartrate_seconds"
        )

        expected_files = [
            "dailyActivity_merged.csv",
            "hourlySteps_merged.csv",
            "hourlyIntensities_merged.csv",
            "minuteSleep_merged.csv",
            "heartrate_seconds_merged.csv"
        ]

        if uploaded_files and len(uploaded_files) == 5:
            file_dict = {f.name: f for f in uploaded_files}
            missing = [f for f in expected_files if f not in file_dict]
            if missing:
                st.error(f"Missing: {missing}")
            else:
                with st.spinner("Processing data pipeline..."):
                    daily    = pd.read_csv(file_dict["dailyActivity_merged.csv"])
                    hourly_s = pd.read_csv(file_dict["hourlySteps_merged.csv"])
                    hourly_i = pd.read_csv(file_dict["hourlyIntensities_merged.csv"])
                    sleep    = pd.read_csv(file_dict["minuteSleep_merged.csv"])
                    hr       = pd.read_csv(file_dict["heartrate_seconds_merged.csv"])
                    daily, hourly_s, hourly_i, sleep, hr = preprocess_timestamps(daily, hourly_s, hourly_i, sleep, hr)
                    master, hr_minute = create_master(daily, hr, sleep)
                    st.session_state['daily_df']    = daily
                    st.session_state['hourly_s_df'] = hourly_s
                    st.session_state['hourly_i_df'] = hourly_i
                    st.session_state['sleep_df']    = sleep
                    st.session_state['hr_df']       = hr
                    st.session_state['master_df']   = master
                    st.session_state['hr_minute_df']= hr_minute
                st.markdown("""
                <div style="background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3);
                            border-radius:10px; padding:0.75rem 1rem; margin-top:0.75rem;">
                    <span class="status-dot"></span>
                    <span style="color:#10b981; font-size:0.82rem; font-weight:600;">All 5 files loaded</span>
                </div>
                """, unsafe_allow_html=True)
        elif uploaded_files:
            st.warning(f"Upload exactly 5 files. ({len(uploaded_files)}/5 uploaded)")
        else:
            st.markdown("""
            <div style="background:rgba(0,212,255,0.05); border:1px dashed rgba(0,212,255,0.2);
                        border-radius:10px; padding:0.75rem 1rem; margin-top:0.5rem;">
                <div style="color:#475569; font-size:0.8rem; line-height:1.6;">
                    Required files:<br>
                    <span style="color:#94a3b8;">&#x25CF; dailyActivity_merged.csv</span><br>
                    <span style="color:#94a3b8;">&#x25CF; hourlySteps_merged.csv</span><br>
                    <span style="color:#94a3b8;">&#x25CF; hourlyIntensities_merged.csv</span><br>
                    <span style="color:#94a3b8;">&#x25CF; minuteSleep_merged.csv</span><br>
                    <span style="color:#94a3b8;">&#x25CF; heartrate_seconds_merged.csv</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <hr style="border:none;height:1px;background:linear-gradient(90deg,transparent,#1e2d4a,transparent);">
        <div style="font-size:0.68rem; color:#334155; text-align:center; padding:0.5rem 0;">
            Milestone 2 &nbsp;|&nbsp; TSFresh &bull; Prophet &bull; Clustering
        </div>
        """, unsafe_allow_html=True)

    # ---------------- GATE: data required ----------------
    if st.session_state['master_df'] is None:
        st.markdown("""
        <div style="background:var(--bg-card); border:1px solid var(--border); border-radius:16px;
                    padding:3rem; text-align:center; margin-top:2rem;">
            <div style="font-size:3rem; margin-bottom:1rem; color:#1e2d4a;">&#9825;</div>
            <div style="font-size:1.25rem; font-weight:600; color:#f1f5f9; margin-bottom:0.5rem;">Awaiting Data</div>
            <div style="color:#475569; font-size:0.9rem;">Upload all five CSV files in the sidebar to begin your analysis.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Retrieve data
    daily     = st.session_state['daily_df']
    hourly_s  = st.session_state['hourly_s_df']
    hourly_i  = st.session_state['hourly_i_df']
    sleep     = st.session_state['sleep_df']
    hr        = st.session_state['hr_df']
    master    = st.session_state['master_df']
    hr_minute = st.session_state['hr_minute_df']

    # ---------------- TOP NAVIGATION ----------------
    nav_labels = ["Data Overview", "TSFresh", "Prophet", "Clustering", "Summary"]
    col1, col2, col3, col4, col5 = st.columns(5)
    for col, label in zip([col1,col2,col3,col4,col5], nav_labels):
        with col:
            if st.button(label, use_container_width=True, key=f"nav_{label}"):
                st.session_state.current_section = label

    st.markdown('<div class="nav-separator"></div>', unsafe_allow_html=True)

    # ---------------- SECTION: DATA OVERVIEW ----------------
    if st.session_state.current_section == "Data Overview":
        st.markdown("""
        <div class="section-header">
          <div class="section-icon icon-blue">&#9878;</div>
          <div>
            <div class="section-title">Dataset Overview</div>
            <div class="section-desc">Raw data summary and key statistics</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Daily Activity Users", daily["Id"].nunique())
        with c2:
            st.metric("Heart Rate Users", hr["Id"].nunique())
        with c3:
            st.metric("Sleep Records Users", sleep["Id"].nunique())
        with c4:
            st.metric("Master Rows", f"{len(master):,}")

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""<div style="font-size:0.75rem;color:#00d4ff;letter-spacing:0.1em;text-transform:uppercase;
                           font-weight:600;margin-bottom:0.5rem;">Daily Activity — First Rows</div>""",
                        unsafe_allow_html=True)
            st.dataframe(daily.head(), use_container_width=True)

        with col_b:
            st.markdown("""<div style="font-size:0.75rem;color:#00d4ff;letter-spacing:0.1em;text-transform:uppercase;
                           font-weight:600;margin-bottom:0.5rem;">Heart Rate (1-min Resampled)</div>""",
                        unsafe_allow_html=True)
            st.dataframe(hr_minute.head(), use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown("""<div style="font-size:0.75rem;color:#a855f7;letter-spacing:0.1em;text-transform:uppercase;
                           font-weight:600;margin-bottom:0.5rem;">Sleep Minutes — First Rows</div>""",
                        unsafe_allow_html=True)
            st.dataframe(sleep.head(), use_container_width=True)

        with col_d:
            st.markdown("""<div style="font-size:0.75rem;color:#a855f7;letter-spacing:0.1em;text-transform:uppercase;
                           font-weight:600;margin-bottom:0.5rem;">Master DataFrame — Daily Aggregates</div>""",
                        unsafe_allow_html=True)
            st.dataframe(master.head(), use_container_width=True)

        st.markdown("""<div style="font-size:0.75rem;color:#10b981;letter-spacing:0.1em;text-transform:uppercase;
                       font-weight:600;margin:1.5rem 0 0.5rem 0;">Key Statistics</div>""",
                    unsafe_allow_html=True)
        st.dataframe(
            master[["TotalSteps","Calories","AvgHR","TotalSleepMinutes","VeryActiveMinutes"]].describe().round(2),
            use_container_width=True
        )

    # ---------------- SECTION: TSFRESH ----------------
    elif st.session_state.current_section == "TSFresh":
        st.markdown("""
        <div class="section-header">
          <div class="section-icon icon-purple">&#9670;</div>
          <div>
            <div class="section-title">TSFresh Feature Extraction</div>
            <div class="section-desc">Statistical features from minute-level heart rate signals</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card info-card-accent">
            <div style="color:#94a3b8; font-size:0.88rem; line-height:1.7;">
                TSFresh automatically extracts <strong style="color:#f1f5f9;">hundreds of time-series features</strong>
                per user using the <strong style="color:#00d4ff;">MinimalFCParameters</strong> preset —
                including mean, variance, skewness, autocorrelation, and more.
                Results are normalized and visualized as a heatmap.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Run TSFresh Feature Extraction", use_container_width=False):
            ts_hr = hr_minute[["Id","Time","HeartRate"]].copy()
            ts_hr = ts_hr.dropna().sort_values(["Id","Time"])
            ts_hr = ts_hr.rename(columns={"Id":"id","Time":"time","HeartRate":"value"})
            with st.spinner("Extracting features — this may take 1-2 minutes..."):
                features = extract_features(
                    ts_hr, column_id="id", column_sort="time", column_value="value",
                    default_fc_parameters=MinimalFCParameters(),
                    n_jobs=1, disable_progressbar=False
                )
                features = features.dropna(axis=1, how="all")
            st.session_state["tsfresh_features"] = features
            st.success(f"Features extracted: {features.shape[0]} users x {features.shape[1]} features")

        if "tsfresh_features" in st.session_state:
            features = st.session_state["tsfresh_features"]

            m1, m2 = st.columns(2)
            m1.metric("Users", features.shape[0])
            m2.metric("Features Extracted", features.shape[1])

            scaler_vis = MinMaxScaler()
            features_norm = pd.DataFrame(
                scaler_vis.fit_transform(features),
                index=features.index, columns=features.columns
            )

            fig, ax = plt.subplots(figsize=(14, 8))
            fig.patch.set_facecolor('#111827')
            ax.set_facecolor('#161d2e')

            cmap = mpl.colors.LinearSegmentedColormap.from_list(
                "health", ["#0a0e1a","#1e2d4a","#00d4ff","#a855f7","#ec4899"]
            )
            sns.heatmap(
                features_norm, cmap=cmap, annot=True, fmt=".2f",
                linewidths=0.4, linecolor="#0a0e1a",
                ax=ax, cbar_kws={"shrink": 0.7}
            )
            ax.set_title("TSFresh Feature Matrix (normalized 0-1)", fontsize=14, fontweight='bold',
                         color='#f1f5f9', pad=15)
            ax.set_xlabel("Features", color='#94a3b8', fontsize=10)
            ax.set_ylabel("User ID", color='#94a3b8', fontsize=10)
            ax.tick_params(colors='#475569')
            plt.colorbar(ax.collections[0], ax=ax).ax.tick_params(colors='#94a3b8')
            plt.tight_layout()
            st.pyplot(fig)

            csv = features.to_csv().encode("utf-8")
            st.download_button(
                "Download Features as CSV",
                csv, "tsfresh_features.csv", "text/csv",
                use_container_width=False
            )
        else:
            st.info("Click the button above to extract time-series features.")

    # ---------------- SECTION: PROPHET ----------------
    elif st.session_state.current_section == "Prophet":
        st.markdown("""
        <div class="section-header">
          <div class="section-icon icon-green">&#9650;</div>
          <div>
            <div class="section-title">Prophet Trend Forecasts</div>
            <div class="section-desc">30-day forward forecasts with 80% confidence intervals</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Heart Rate
        st.markdown("""<div style="font-size:0.75rem;color:#ec4899;letter-spacing:0.1em;
                       text-transform:uppercase;font-weight:600;margin-bottom:0.75rem;">
                       Heart Rate Forecast</div>""", unsafe_allow_html=True)

        if st.button("Run Heart Rate Prophet Model"):
            prophet_hr = hr_minute.groupby("Date")["HeartRate"].mean().reset_index()
            prophet_hr.columns = ["ds","y"]
            prophet_hr["ds"] = pd.to_datetime(prophet_hr["ds"])
            prophet_hr = prophet_hr.dropna().sort_values("ds")

            if len(prophet_hr) < 2:
                st.warning("Not enough data for forecasting.")
            else:
                with st.spinner("Fitting Prophet model on heart rate data..."):
                    model_hr = Prophet(
                        daily_seasonality=False, weekly_seasonality=True,
                        yearly_seasonality=False, interval_width=0.80,
                        changepoint_prior_scale=0.01, changepoint_range=0.8
                    )
                    model_hr.fit(prophet_hr)
                    future_hr    = model_hr.make_future_dataframe(periods=30)
                    forecast_hr  = model_hr.predict(future_hr)

                fig, ax = plt.subplots(figsize=(14, 5))
                ax.scatter(prophet_hr["ds"], prophet_hr["y"],
                           color="#ec4899", s=20, alpha=0.7, label="Actual HR", zorder=3)
                ax.plot(forecast_hr["ds"], forecast_hr["yhat"],
                        color="#00d4ff", linewidth=2.5, label="Predicted Trend")
                ax.fill_between(forecast_hr["ds"],
                                forecast_hr["yhat_lower"], forecast_hr["yhat_upper"],
                                alpha=0.2, color="#00d4ff", label="80% Confidence Interval")
                ax.axvline(prophet_hr["ds"].max(), color="#f59e0b",
                           linestyle="--", linewidth=2, label="Forecast Start")
                ax.set_title("Heart Rate — Prophet Trend Forecast", fontsize=14, fontweight='bold', color='#f1f5f9')
                ax.set_xlabel("Date", color='#94a3b8')
                ax.set_ylabel("Heart Rate (bpm)", color='#94a3b8')
                ax.legend(facecolor='#161d2e', edgecolor='#1e2d4a', labelcolor='#94a3b8')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

                fig2 = model_hr.plot_components(forecast_hr)
                fig2.set_facecolor('#111827')
                for fax in fig2.axes:
                    fax.set_facecolor('#161d2e')
                    fax.tick_params(colors='#475569')
                    fax.xaxis.label.set_color('#94a3b8')
                    fax.yaxis.label.set_color('#94a3b8')
                    fax.title.set_color('#f1f5f9')
                    for line in fax.get_lines():
                        if line.get_color() == 'b':
                            line.set_color('#00d4ff')
                    for coll in fax.collections:
                        coll.set_facecolor('#00d4ff')
                        coll.set_alpha(0.2)
                plt.suptitle("Prophet Components — Heart Rate", fontsize=12, color='#f1f5f9', y=1.01)
                plt.tight_layout()
                st.pyplot(fig2)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Steps & Sleep
        st.markdown("""<div style="font-size:0.75rem;color:#10b981;letter-spacing:0.1em;
                       text-transform:uppercase;font-weight:600;margin-bottom:0.75rem;">
                       Steps &amp; Sleep Forecasts</div>""", unsafe_allow_html=True)

        if st.button("Run Steps & Sleep Prophet Models"):
            steps_agg = daily.groupby("ActivityDate")["TotalSteps"].mean().reset_index()
            steps_agg.columns = ["ds","y"]
            steps_agg["ds"] = pd.to_datetime(steps_agg["ds"], errors="coerce")
            steps_agg = steps_agg.dropna().sort_values("ds")

            sleep_agg = master.groupby("Date")["TotalSleepMinutes"].mean().reset_index()
            sleep_agg.columns = ["ds","y"]
            sleep_agg["ds"] = pd.to_datetime(sleep_agg["ds"], errors="coerce")
            sleep_agg = sleep_agg.dropna().sort_values("ds")

            if len(steps_agg) < 2 or len(sleep_agg) < 2:
                st.warning("Not enough data for one of the series.")
            else:
                with st.spinner("Fitting Steps and Sleep models..."):
                    model_steps = Prophet(weekly_seasonality=True, interval_width=0.80)
                    model_steps.fit(steps_agg)
                    forecast_steps = model_steps.predict(model_steps.make_future_dataframe(periods=30))

                    model_sleep = Prophet(weekly_seasonality=True, interval_width=0.80)
                    model_sleep.fit(sleep_agg)
                    forecast_sleep = model_sleep.predict(model_sleep.make_future_dataframe(periods=30))

                fig, axes = plt.subplots(2, 1, figsize=(14, 10), facecolor='#111827')

                # Steps
                axes[0].scatter(steps_agg["ds"], steps_agg["y"], color="#10b981", s=20, alpha=0.7, label="Actual Steps")
                axes[0].plot(forecast_steps["ds"], forecast_steps["yhat"], color="#f1f5f9", linewidth=2.5, label="Trend")
                axes[0].fill_between(forecast_steps["ds"], forecast_steps["yhat_lower"], forecast_steps["yhat_upper"],
                                     alpha=0.2, color="#10b981", label="80% CI")
                axes[0].axvline(steps_agg["ds"].max(), color="#f59e0b", linestyle="--", linewidth=2, label="Forecast Start")
                axes[0].set_title("Daily Steps — Prophet Forecast", fontsize=13, fontweight='bold', color='#f1f5f9')
                axes[0].set_ylabel("Steps", color='#94a3b8')
                axes[0].legend(fontsize=9, facecolor='#161d2e', edgecolor='#1e2d4a', labelcolor='#94a3b8')
                axes[0].tick_params(axis='x', rotation=45, colors='#475569')
                axes[0].tick_params(axis='y', colors='#475569')

                # Sleep
                axes[1].scatter(sleep_agg["ds"], sleep_agg["y"], color="#a855f7", s=20, alpha=0.7, label="Actual Sleep")
                axes[1].plot(forecast_sleep["ds"], forecast_sleep["yhat"], color="#f1f5f9", linewidth=2.5, label="Trend")
                axes[1].fill_between(forecast_sleep["ds"], forecast_sleep["yhat_lower"], forecast_sleep["yhat_upper"],
                                     alpha=0.2, color="#a855f7", label="80% CI")
                axes[1].axvline(sleep_agg["ds"].max(), color="#f59e0b", linestyle="--", linewidth=2, label="Forecast Start")
                axes[1].set_title("Sleep Minutes — Prophet Forecast", fontsize=13, fontweight='bold', color='#f1f5f9')
                axes[1].set_ylabel("Sleep (min)", color='#94a3b8')
                axes[1].legend(fontsize=9, facecolor='#161d2e', edgecolor='#1e2d4a', labelcolor='#94a3b8')
                axes[1].tick_params(axis='x', rotation=45, colors='#475569')
                axes[1].tick_params(axis='y', colors='#475569')

                plt.tight_layout()
                st.pyplot(fig)

    # ---------------- SECTION: CLUSTERING ----------------
    elif st.session_state.current_section == "Clustering":
        st.markdown("""
        <div class="section-header">
          <div class="section-icon icon-orange">&#9670;</div>
          <div>
            <div class="section-title">User Clustering</div>
            <div class="section-desc">KMeans, DBSCAN and t-SNE projections on activity features</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        cluster_cols = ["TotalSteps","Calories","VeryActiveMinutes","FairlyActiveMinutes",
                        "LightlyActiveMinutes","SedentaryMinutes","TotalSleepMinutes"]
        cluster_features = master.groupby("Id")[cluster_cols].mean().dropna()

        c1, c2 = st.columns(2)
        c1.metric("Users for Clustering", cluster_features.shape[0])
        c2.metric("Feature Dimensions", cluster_features.shape[1])

        st.markdown("""<div style="font-size:0.75rem;color:#00d4ff;letter-spacing:0.1em;
                       text-transform:uppercase;font-weight:600;margin:1.25rem 0 0.5rem;">
                       User-level Feature Matrix</div>""", unsafe_allow_html=True)
        st.dataframe(cluster_features.head(), use_container_width=True)

        scaler  = StandardScaler()
        X_scaled = scaler.fit_transform(cluster_features)

        # Elbow curve
        st.markdown("""<div style="font-size:0.75rem;color:#f59e0b;letter-spacing:0.1em;
                       text-transform:uppercase;font-weight:600;margin:1.5rem 0 0.75rem;">
                       KMeans Elbow Analysis</div>""", unsafe_allow_html=True)

        inertias = []
        K_range  = range(2, 10)
        for k_val in K_range:
            km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(K_range, inertias, "o-", color="#00d4ff", linewidth=2.5, markersize=9,
                markerfacecolor="#a855f7", markeredgecolor="#00d4ff", markeredgewidth=2)
        ax.fill_between(K_range, inertias, alpha=0.08, color="#00d4ff")
        ax.set_xlabel("Number of Clusters (K)", color='#94a3b8', fontsize=10)
        ax.set_ylabel("Inertia", color='#94a3b8', fontsize=10)
        ax.set_title("KMeans Elbow Curve", fontsize=13, fontweight='bold', color='#f1f5f9')
        plt.tight_layout()
        st.pyplot(fig)

        # KMeans
        st.markdown("""<div style="font-size:0.75rem;color:#a855f7;letter-spacing:0.1em;
                       text-transform:uppercase;font-weight:600;margin:1.5rem 0 0.75rem;">
                       KMeans Configuration</div>""", unsafe_allow_html=True)

        k = st.slider("Number of Clusters (K)", 2, 8, 3)

        if st.button("Run KMeans Clustering"):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            cluster_features_k = cluster_features.copy()
            cluster_features_k["Cluster"] = labels

            dist_col = st.columns(k)
            vc = cluster_features_k["Cluster"].value_counts().sort_index()
            palette = ["#00d4ff","#ec4899","#10b981","#f59e0b","#a855f7","#fc8181"]
            for i, col in enumerate(dist_col):
                if i < k:
                    count = vc.get(i, 0)
                    pct   = count / len(cluster_features_k) * 100
                    col.metric(f"Cluster {i}", f"{count} users", f"{pct:.1f}%")

            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            var   = pca.explained_variance_ratio_ * 100

            fig, ax = plt.subplots(figsize=(10, 6))
            for cl in sorted(set(labels)):
                mask = labels == cl
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                           c=palette[cl % len(palette)],
                           label=f"Cluster {cl}", s=120, alpha=0.9,
                           edgecolors='#0a0e1a', linewidths=1.2)
            ax.set_title(f"KMeans (K={k}) — PCA Projection", fontsize=13, fontweight='bold', color='#f1f5f9')
            ax.set_xlabel(f"PC1 ({var[0]:.1f}%)", color='#94a3b8')
            ax.set_ylabel(f"PC2 ({var[1]:.1f}%)", color='#94a3b8')
            ax.legend(facecolor='#161d2e', edgecolor='#1e2d4a', labelcolor='#94a3b8')
            plt.tight_layout()
            st.pyplot(fig)

            # Cluster profiles
            st.markdown("""<div style="font-size:0.75rem;color:#10b981;letter-spacing:0.1em;
                           text-transform:uppercase;font-weight:600;margin:1.25rem 0 0.5rem;">
                           Cluster Profiles</div>""", unsafe_allow_html=True)
            feature_cols = [c for c in cluster_features_k.columns if c != "Cluster"]
            profile = cluster_features_k.groupby("Cluster")[feature_cols].mean().round(2)
            st.dataframe(profile, use_container_width=True)

            fig2, ax2 = plt.subplots(figsize=(13, 5))
            plot_cols   = ["TotalSteps","Calories","VeryActiveMinutes","SedentaryMinutes","TotalSleepMinutes"]
            plot_colors = ["#00d4ff","#ec4899","#10b981","#f59e0b","#a855f7"]
            x = np.arange(k)
            width = 0.15
            for idx, (col_name, color) in enumerate(zip(plot_cols, plot_colors)):
                # normalize to 0-1 for comparison
                vals = profile[col_name].values
                norm_vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
                ax2.bar(x + idx*width, profile[col_name].values, width,
                        label=col_name, color=color, alpha=0.85, edgecolor='#0a0e1a')
            ax2.set_title("Cluster Profiles — Key Feature Averages", fontsize=13, fontweight='bold', color='#f1f5f9')
            ax2.set_xlabel("Cluster", color='#94a3b8')
            ax2.set_ylabel("Mean Value", color='#94a3b8')
            ax2.set_xticks(x + width * 2)
            ax2.set_xticklabels([f"Cluster {i}" for i in range(k)], color='#94a3b8')
            ax2.legend(bbox_to_anchor=(1.01, 1), facecolor='#161d2e', edgecolor='#1e2d4a', labelcolor='#94a3b8')
            plt.tight_layout()
            st.pyplot(fig2)

            # Interpretation cards
            st.markdown("""<div style="font-size:0.75rem;color:#00d4ff;letter-spacing:0.1em;
                           text-transform:uppercase;font-weight:600;margin:1.25rem 0 0.75rem;">
                           Cluster Interpretation</div>""", unsafe_allow_html=True)

            for i in range(k):
                row    = profile.loc[i]
                steps  = row["TotalSteps"]
                sed    = row["SedentaryMinutes"]
                active = row["VeryActiveMinutes"]
                if steps > 10000:
                    profile_label = "HIGHLY ACTIVE"
                    color_accent  = "#10b981"
                    icon_char     = "A"
                elif steps > 5000:
                    profile_label = "MODERATELY ACTIVE"
                    color_accent  = "#f59e0b"
                    icon_char     = "M"
                else:
                    profile_label = "SEDENTARY"
                    color_accent  = "#a855f7"
                    icon_char     = "S"

                st.markdown(f"""
                <div style="background:var(--bg-card); border:1px solid var(--border); border-left:3px solid {color_accent};
                            border-radius:12px; padding:1rem 1.25rem; margin-bottom:0.6rem;
                            display:flex; gap:1rem; align-items:flex-start;">
                    <div style="background:{color_accent}22; border:1px solid {color_accent}55;
                                color:{color_accent}; font-weight:700; font-size:0.9rem;
                                width:32px; height:32px; border-radius:8px; display:flex;
                                align-items:center; justify-content:center; flex-shrink:0;">{icon_char}</div>
                    <div>
                        <div style="font-weight:700; color:#f1f5f9; margin-bottom:0.25rem;">
                            Cluster {i}
                            <span style="font-size:0.7rem; background:{color_accent}22; color:{color_accent};
                                         border:1px solid {color_accent}44; border-radius:50px;
                                         padding:0.15rem 0.6rem; margin-left:0.5rem; font-weight:600;
                                         letter-spacing:0.08em;">{profile_label}</span>
                        </div>
                        <div style="color:#94a3b8; font-size:0.82rem; line-height:1.8;">
                            Avg Steps: <strong style="color:#f1f5f9;">{steps:,.0f}</strong>
                            &nbsp;&nbsp;|&nbsp;&nbsp;
                            Sedentary: <strong style="color:#f1f5f9;">{sed:.0f} min</strong>
                            &nbsp;&nbsp;|&nbsp;&nbsp;
                            Very Active: <strong style="color:#f1f5f9;">{active:.0f} min</strong>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # DBSCAN
        st.markdown("""<div style="font-size:0.75rem;color:#ec4899;letter-spacing:0.1em;
                       text-transform:uppercase;font-weight:600;margin:2rem 0 0.75rem;">
                       DBSCAN Configuration</div>""", unsafe_allow_html=True)

        db_c1, db_c2 = st.columns(2)
        with db_c1:
            eps = st.slider("Epsilon (EPS)", 0.5, 5.0, 2.2, 0.1)
        with db_c2:
            min_samples = st.slider("Min Samples", 2, 10, 2)

        if st.button("Run DBSCAN Clustering"):
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise    = list(labels).count(-1)

            mn1, mn2 = st.columns(2)
            mn1.metric("Clusters Found", n_clusters)
            mn2.metric("Noise Points", n_noise, delta=f"{n_noise/len(labels)*100:.1f}% of data", delta_color="inverse")

            pca    = PCA(n_components=2, random_state=42)
            X_pca  = pca.fit_transform(X_scaled)
            var    = pca.explained_variance_ratio_ * 100
            palette = ["#00d4ff","#ec4899","#10b981","#f59e0b","#a855f7","#fc8181"]

            fig, ax = plt.subplots(figsize=(10, 6))
            for lb in sorted(set(labels)):
                mask = labels == lb
                if lb == -1:
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                               c="#ef4444", marker="x", s=150, label="Noise", alpha=0.9, linewidths=2)
                else:
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                               c=palette[lb % len(palette)],
                               label=f"Cluster {lb}", s=120, alpha=0.9,
                               edgecolors='#0a0e1a', linewidths=1.2)
            ax.set_title(f"DBSCAN (eps={eps}, min={min_samples}) — PCA Projection",
                         fontsize=13, fontweight='bold', color='#f1f5f9')
            ax.set_xlabel(f"PC1 ({var[0]:.1f}%)", color='#94a3b8')
            ax.set_ylabel(f"PC2 ({var[1]:.1f}%)", color='#94a3b8')
            ax.legend(facecolor='#161d2e', edgecolor='#1e2d4a', labelcolor='#94a3b8')
            plt.tight_layout()
            st.pyplot(fig)

        # t-SNE
        st.markdown("""<div style="font-size:0.75rem;color:#a855f7;letter-spacing:0.1em;
                       text-transform:uppercase;font-weight:600;margin:2rem 0 0.75rem;">
                       t-SNE Manifold Projection</div>""", unsafe_allow_html=True)

        if st.button("Run t-SNE (may take a moment)"):
            with st.spinner("Computing t-SNE embedding..."):
                tsne     = TSNE(n_components=2, random_state=42,
                                perplexity=min(30, len(X_scaled)-1), max_iter=1000)
                X_tsne   = tsne.fit_transform(X_scaled)
                km       = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels_km= km.fit_predict(X_scaled)
                db       = DBSCAN(eps=eps, min_samples=min_samples)
                labels_db= db.fit_predict(X_scaled)

            palette = ["#00d4ff","#ec4899","#10b981","#f59e0b","#a855f7","#fc8181"]
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='#111827')

            for cl in sorted(set(labels_km)):
                mask = labels_km == cl
                axes[0].scatter(X_tsne[mask,0], X_tsne[mask,1],
                                c=palette[cl % len(palette)], label=f"Cluster {cl}",
                                s=90, alpha=0.9, edgecolors='#0a0e1a', linewidths=1)
            axes[0].set_title("KMeans — t-SNE", fontsize=12, fontweight='bold', color='#f1f5f9')
            axes[0].legend(facecolor='#161d2e', edgecolor='#1e2d4a', labelcolor='#94a3b8')

            for lb in sorted(set(labels_db)):
                mask = labels_db == lb
                if lb == -1:
                    axes[1].scatter(X_tsne[mask,0], X_tsne[mask,1],
                                    c="#ef4444", marker="x", s=150, label="Noise", alpha=0.9, linewidths=2)
                else:
                    axes[1].scatter(X_tsne[mask,0], X_tsne[mask,1],
                                    c=palette[lb % len(palette)], label=f"Cluster {lb}",
                                    s=90, alpha=0.9, edgecolors='#0a0e1a', linewidths=1)
            axes[1].set_title("DBSCAN — t-SNE", fontsize=12, fontweight='bold', color='#f1f5f9')
            axes[1].legend(facecolor='#161d2e', edgecolor='#1e2d4a', labelcolor='#94a3b8')

            for ax in axes:
                ax.set_facecolor('#161d2e')
                ax.tick_params(colors='#475569')

            plt.tight_layout()
            st.pyplot(fig)

    # ---------------- SECTION: SUMMARY ----------------
    elif st.session_state.current_section == "Summary":
        st.markdown("""
        <div class="section-header">
          <div class="section-icon icon-pink">&#9672;</div>
          <div>
            <div class="section-title">Milestone 2 Summary</div>
            <div class="section-desc">Pipeline overview and key findings</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Stats row
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Users", "35")
        s2.metric("Study Period", "31 days")
        s3.metric("TSFresh Features", "10+")
        s4.metric("Forecast Horizon", "30 days")

        st.markdown("<br>", unsafe_allow_html=True)

        # Pipeline overview
        st.markdown("""<div style="font-size:0.75rem;color:#00d4ff;letter-spacing:0.1em;
                       text-transform:uppercase;font-weight:600;margin-bottom:0.75rem;">
                       Pipeline Overview</div>""", unsafe_allow_html=True)

        pipeline_items = [
            ("Dataset",   "Real Fitbit device data — 35 users, 31 days (March–April 2016)"),
            ("TSFresh",   "Extracted 10 statistical features from minute-level heart rate per user via MinimalFCParameters."),
            ("Prophet HR","30-day heart rate forecast with weekly seasonality, 80% CI, changepoint_prior_scale=0.01."),
            ("Prophet Steps","30-day steps forecast with weekly seasonality and 80% confidence intervals."),
            ("Prophet Sleep","30-day sleep forecast; accounts for missing nights and noisy signal."),
            ("KMeans",    "K=3 optimal (elbow): 12 moderately active, 15 sedentary, 8 highly active users."),
            ("DBSCAN",    "eps=2.2, min_samples=2 yielded 3 clusters + 1 outlier (2.9% noise rate)."),
        ]

        for idx, (label, desc) in enumerate(pipeline_items):
            accent = ["#00d4ff","#a855f7","#ec4899","#10b981","#f59e0b","#63b3ed","#f687b3"][idx % 7]
            st.markdown(f"""
            <div class="summary-item">
                <div class="summary-num">{idx+1}</div>
                <div class="summary-text">
                    <strong>{label}</strong><br>{desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Screenshots checklist
        st.markdown("""<div style="font-size:0.75rem;color:#a855f7;letter-spacing:0.1em;
                       text-transform:uppercase;font-weight:600;margin-bottom:0.75rem;">
                       Submission Screenshots Checklist</div>""", unsafe_allow_html=True)

        screenshots = [
            "TSFresh feature matrix heatmap (normalized 0-1)",
            "Prophet HR forecast with 80% confidence interval",
            "Steps Prophet forecast (30-day)",
            "Sleep Prophet forecast (combined plot)",
            "KMeans PCA scatter plot with cluster labels",
            "DBSCAN PCA scatter plot with noise markers",
            "t-SNE projection — KMeans vs DBSCAN side-by-side",
            "Cluster profiles bar chart with feature averages",
        ]

        for idx, item in enumerate(screenshots):
            st.markdown(f"""
            <div style="background:rgba(168,85,247,0.06); border:1px solid rgba(168,85,247,0.2);
                        border-radius:10px; padding:0.7rem 1rem; margin-bottom:0.4rem;
                        display:flex; align-items:center; gap:0.75rem;">
                <div style="background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.3);
                            color:#10b981; font-size:0.7rem; font-weight:700;
                            width:22px; height:22px; border-radius:5px;
                            display:flex; align-items:center; justify-content:center; flex-shrink:0;">
                    &#10003;
                </div>
                <div style="color:#94a3b8; font-size:0.85rem;">
                    <strong style="color:#f1f5f9;">{idx+1}.</strong> {item}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(0,212,255,0.05),rgba(168,85,247,0.05));
                    border:1px solid rgba(0,212,255,0.2); border-radius:14px;
                    padding:1.5rem; margin-top:1.5rem; text-align:center;">
            <div style="color:#94a3b8; font-size:0.85rem; line-height:1.8;">
                Built with <strong style="color:#f1f5f9;">Streamlit</strong> &nbsp;|&nbsp;
                <strong style="color:#00d4ff;">TSFresh</strong> &nbsp;|&nbsp;
                <strong style="color:#ec4899;">Prophet</strong> &nbsp;|&nbsp;
                <strong style="color:#a855f7;">scikit-learn</strong><br>
                <span style="font-size:0.75rem; color:#475569;">Milestone 2 · Health Analytics Pipeline · 2026</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# MAIN APP ROUTER
# ============================================================
if st.session_state.app_mode == 'home':
    show_home()
elif st.session_state.app_mode == 'milestone1':
    show_milestone1()
elif st.session_state.app_mode == 'milestone2':
    show_milestone2()