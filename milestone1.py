import streamlit as st
import pandas as pd
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🚀 Data Preprocessing",
    page_icon="✨",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#1f1c2c,#928dab);
    color:white;
}

.glass-card {
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(12px);
    padding:20px;
    border-radius:20px;
    text-align:center;
    box-shadow:0 8px 32px rgba(0,0,0,0.3);
}

.stButton>button {
    background: linear-gradient(90deg,#ff512f,#dd2476);
    color:white;
    border:none;
    border-radius:12px;
    padding:10px 20px;
    font-size:16px;
    transition:0.3s;
}

.stButton>button:hover {
    transform:scale(1.1);
    background: linear-gradient(90deg,#36d1dc,#5b86e5);
}

</style>
""", unsafe_allow_html=True)

st.title("🚀 Data Preprocessing")
st.write("Upload → Explore → Clean → Download")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📁 Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])

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