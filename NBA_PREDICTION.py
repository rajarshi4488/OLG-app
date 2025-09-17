import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 New Daily Run Dashboard")

@st.cache_data
def load_dashboard_data():
    return pd.read_excel("dashboard_data_daily_run.xlsx", sheet_name=None)

data = load_dashboard_data()
available_sheets = set(data.keys())

# ---------- 1. Business KPIs ----------
st.subheader("1️⃣ Business KPIs (Latest Run)")
if "Business_KPIs" in available_sheets:
    kpi_df = data["Business_KPIs"]
    # nice metrics in one row
    cols = st.columns(4)
    for i, row in kpi_df.iterrows():
        cols[i % 4].metric(label=row["Metric"], value=f"{row['Value']:.3f}")
else:
    st.warning("Business_KPIs sheet not found.")

st.markdown("---")

# ---------- 2. Top-3 Recommendations ----------
st.subheader("2️⃣ Daily Player Top-3 Recommendations")
if "Top3_Games" in available_sheets:
    top3_df = data["Top3_Games"]
    st.dataframe(
        top3_df,
        use_container_width=True,
        hide_index=True
    )

    # Optional quick summary
    st.write(f"Total players with recommendations: {top3_df['mask_id'].nunique()}")
else:
    st.warning("Top3_Recommendations sheet not found.")

st.markdown("---")

# ---------- 3. Daily vs Training Feature Comparison ----------
st.subheader("3️⃣ Daily vs Training Feature Comparison")
if "Drift_Summary" in available_sheets:
    feat_df = data["Drift_Summary"]

    # Nice side-by-side bar plot
    fig, ax = plt.subplots(figsize=(7,4))
    width = 0.35
    x = range(len(feat_df))
    ax.bar(x, feat_df["Train Mean"], width, label="Train Mean", color="steelblue")
    ax.bar([p + width for p in x], feat_df["Day Mean"], width, label="Day Mean", color="orange")
    ax.set_xticks([p + width/2 for p in x])
    ax.set_xticklabels(feat_df["Feature"])
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)

    # Table for precise numbers
    st.dataframe(feat_df, use_container_width=True)
else:
    st.warning("Feature_Diff sheet not found.")


# -----------------------------------------------------------
# 8. ChatGPT Q & A on the Dashboard
# -----------------------------------------------------------
import os
from openai import OpenAI

st.markdown("---")
st.subheader("💬 Ask Questions about this Dashboard")

# One-time model and client initialization
api_key = "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"
if not api_key:
    st.warning("Set OPENAI_API_KEY as an environment variable to enable Q & A.")
else:
    client = OpenAI(api_key=api_key)

    # Text box for user question
    question = st.text_area("Type your question about the KPIs, trends, or players:")

    if st.button("Ask ChatGPT") and question:
        with st.spinner("Generating answer..."):
            # Build a context string from the dashboard’s current data
            context_parts = []
            if "Business_KPIs" in data:
                context_parts.append("Business KPIs:\n" + data["Business_KPIs"].to_csv(index=False))
            if "Top_Games" in data:
                context_parts.append("Top Games:\n" + data["Top_Games"].head(10).to_csv(index=False))
            if "Top_Teams" in data:
                context_parts.append("Top Teams:\n" + data["Top_Teams"].head(10).to_csv(index=False))
            if "Daily_Active_Users" in data:
                context_parts.append("Recent DAU summary:\n" + data["Daily_Active_Users"].tail(7).to_csv(index=False))
            # …add any other sheets you want ChatGPT to see

            context = "\n\n".join(context_parts)

            # Send prompt to OpenAI
            prompt = (
                "You are an assistant that answers questions about an NBA Top-3 Recommendation "
                "dashboard. Use the data below to answer the question factually and concisely.\n\n"
                f"DATA:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"
            )

            resp = client.chat.completions.create(
                model="gpt-4o-mini",  # or gpt-4-turbo/gpt-3.5-turbo if preferred
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.2
            )
            st.success(resp.choices[0].message.content)
