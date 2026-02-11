import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="QuoteGuard", page_icon="🛡️", layout="wide")

st.title("🛡️ QuoteGuard")
st.caption("Upload insurance quote data and instantly assign risk tiers using percentile thresholds (P60 / P85).")

# --- Helper functions ---
def find_numeric_cols(df: pd.DataFrame):
    numeric_cols = []
    for c in df.columns:
        try:
            pd.to_numeric(df[c], errors="raise")
            numeric_cols.append(c)
        except Exception:
            pass
    # Also include columns already numeric
    for c in df.select_dtypes(include=[np.number]).columns:
        if c not in numeric_cols:
            numeric_cols.append(c)
    return numeric_cols

def pick_default_amount_col(df: pd.DataFrame):
    """Try to guess the best 'amount' column (premium/quote/price)."""
    candidates = []
    for c in df.columns:
        name = str(c).lower()
        if any(k in name for k in ["premium", "quote", "price", "amount", "cost", "annual"]):
            candidates.append(c)
    numeric_cols = find_numeric_cols(df)
    # Prioritize candidates that are numeric
    for c in candidates:
        if c in numeric_cols:
            return c
    # Otherwise pick first numeric col
    if numeric_cols:
        return numeric_cols[0]
    return None

def compute_thresholds(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 5:
        return None
    p60 = float(np.percentile(s, 60))
    p85 = float(np.percentile(s, 85))
    return p60, p85

def assign_tier(x, p60, p85):
    if pd.isna(x):
        return "Unknown"
    if x <= p60:
        return "Low"
    elif x <= p85:
        return "Medium"
    else:
        return "High"

def tier_badge(tier: str):
    if tier == "Low":
        return "🟢 Low"
    if tier == "Medium":
        return "🟡 Medium"
    if tier == "High":
        return "🔴 High"
    return "⚪ Unknown"

# --- Sidebar ---
st.sidebar.header("Upload")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
use_manual_thresholds = st.sidebar.toggle("Use manual thresholds", value=False)

st.sidebar.header("Export")
export_filename = st.sidebar.text_input("Export filename", value="quoteguard_tiers.csv")

# --- Main ---
if not uploaded:
    st.info("Upload a CSV to begin. Recommended columns: premium/quote amount, optional customer fields.")
    st.stop()

# Read CSV
raw = uploaded.read()
df = pd.read_csv(io.BytesIO(raw))

st.subheader("Preview")
st.dataframe(df.head(50), use_container_width=True)

numeric_cols = find_numeric_cols(df)
if not numeric_cols:
    st.error("No numeric columns detected. QuoteGuard needs at least one numeric column (e.g., premium/quote amount).")
    st.stop()

default_amount_col = pick_default_amount_col(df)
amount_col = st.selectbox(
    "Select the quote/premium amount column",
    options=numeric_cols,
    index=(numeric_cols.index(default_amount_col) if default_amount_col in numeric_cols else 0),
)

amount_series = pd.to_numeric(df[amount_col], errors="coerce")
thresholds = compute_thresholds(amount_series)

colA, colB, colC = st.columns([1, 1, 2])

if use_manual_thresholds:
    with colA:
        p60 = st.number_input("Manual P60", value=11399.86, step=100.0, format="%.2f")
    with colB:
        p85 = st.number_input("Manual P85", value=24990.17, step=100.0, format="%.2f")
else:
    if thresholds is None:
        st.error("Not enough valid numeric values in the selected column to compute percentiles.")
        st.stop()
    p60, p85 = thresholds
    with colA:
        st.metric("Computed P60", f"{p60:,.2f}")
    with colB:
        st.metric("Computed P85", f"{p85:,.2f}")

with colC:
    st.caption("Tier logic:")
    st.write(f"- Low: ≤ P60 ({p60:,.2f})")
    st.write(f"- Medium: (P60, P85] ({p85:,.2f})")
    st.write(f"- High: > P85")

df_out = df.copy()
df_out["QuoteGuard_Tier"] = amount_series.apply(lambda x: assign_tier(x, p60, p85))
df_out["QuoteGuard_Tier_Badge"] = df_out["QuoteGuard_Tier"].apply(tier_badge)

st.subheader("Tier Results")
tier_counts = df_out["QuoteGuard_Tier"].value_counts(dropna=False)
st.write(tier_counts)

st.dataframe(df_out.head(200), use_container_width=True)

# Charts
st.subheader("Charts")
c1, c2 = st.columns(2)

with c1:
    fig = plt.figure()
    plt.hist(amount_series.dropna(), bins=30)
    plt.axvline(p60, linestyle="--")
    plt.axvline(p85, linestyle="--")
    plt.title(f"Distribution of {amount_col}")
    plt.xlabel(amount_col)
    plt.ylabel("Count")
    st.pyplot(fig, clear_figure=True)

with c2:
    fig2 = plt.figure()
    tier_counts_plot = df_out["QuoteGuard_Tier"].value_counts()
    plt.bar(tier_counts_plot.index.astype(str), tier_counts_plot.values)
    plt.title("Tier Counts")
    plt.xlabel("Tier")
    plt.ylabel("Count")
    st.pyplot(fig2, clear_figure=True)

# Export
st.subheader("Download")
csv_bytes = df_out.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download tiered CSV",
    data=csv_bytes,
    file_name=export_filename,
    mime="text/csv",
)

st.success("✅ QuoteGuard is ready. Next: deploy to Streamlit Cloud and link it in your portfolio.")
