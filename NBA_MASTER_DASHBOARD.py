import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="NBA Top-3 Recommendation Dashboard", layout="wide")
st.title("NBA Top-3 Recommendation Dashboard")

@st.cache_data
def load_dashboard_data():
    return pd.read_excel("dashboard_data.xlsx", sheet_name=None)

data = load_dashboard_data()
available_sheets = set(data.keys())

# -----------------------------------------------------------
# 1. Business KPIs
# -----------------------------------------------------------
if "Business_KPIs" in available_sheets:
    k = data["Business_KPIs"].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision@3", f"{k.get('Precision@3', float('nan')):.3f}")
    c2.metric("MAP@3", f"{k.get('MAP@3', float('nan')):.3f}")
    c3.metric("Wager-capture %", f"{k.get('Wager-capture %', float('nan')):.1f}%")
    c4.metric("User coverage %", f"{k.get('User coverage %', float('nan')):.1f}%")
    st.markdown("---")

# -----------------------------------------------------------
# 2. DAU Trends
# -----------------------------------------------------------
if "Daily_Active_Users" in available_sheets:
    st.subheader("Daily Active Users (DAU)")
    dau = data["Daily_Active_Users"]
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(dau['day'], dau['unique_users'])
    ax.set_xlabel("Date"); ax.set_ylabel("Unique Users")
    st.pyplot(fig)

if "Monthly_DAU" in available_sheets:
    st.subheader("Monthly Active Users")
    dau_m = data["Monthly_DAU"]
    fig2, ax2 = plt.subplots(figsize=(6,3))
    ax2.bar(dau_m['month'], dau_m['unique_users'])
    ax2.set_ylabel("Unique Users"); ax2.set_xlabel("Month")
    st.pyplot(fig2)

if "Churn_Summary" in available_sheets:
    st.markdown("---")
    st.subheader("Churn Summary (Days Active per User)")
    st.dataframe(data["Churn_Summary"])

if "Inactivity_Bands" in available_sheets:
    st.subheader("Inactivity Bands (Users by Recency)")
    st.dataframe(data["Inactivity_Bands"])

st.markdown("---")

# -----------------------------------------------------------
# 3. Top Games & Teams
# -----------------------------------------------------------
if "Top_Games" in available_sheets or "Top_Teams" in available_sheets:
    col_left, col_right = st.columns(2)
    if "Top_Games" in available_sheets:
        with col_left:
            st.subheader("Top 10 Games by Unique Users")
            st.dataframe(data["Top_Games"])
    if "Top_Teams" in available_sheets:
        with col_right:
            st.subheader("Top 10 Teams by Unique Users")
            st.dataframe(data["Top_Teams"])
    st.markdown("---")

# -----------------------------------------------------------
# 4. Weekend Lift, Conversion, Peer Momentum
# -----------------------------------------------------------
if "Weekend_Lift" in available_sheets:
    st.subheader("Top 10 Games by Weekend Lift")
    st.dataframe(data["Weekend_Lift"])

if "Top_Conversion" in available_sheets:
    st.markdown("---")
    st.subheader("Top 10 Games by Conversion Rate")
    st.dataframe(data["Top_Conversion"])

if "Peer_Momentum" in available_sheets:
    st.markdown("---")
    st.subheader("Top 10 Games by Peer Momentum")
    st.dataframe(data["Peer_Momentum"])

st.markdown("---")

# -----------------------------------------------------------
# 5. Monitoring / Responsible Gaming
# -----------------------------------------------------------
if any(s in available_sheets for s in ["KPI_Anomalies", "Watch_QuietWhales", "Bursty_Games"]):
    st.subheader("Monitoring & Responsible Gaming")
    m1, m2, m3 = st.columns(3)
    if "KPI_Anomalies" in available_sheets:
        with m1:
            st.write("**KPI Anomalies**")
            st.dataframe(data["KPI_Anomalies"])
    if "Watch_QuietWhales" in available_sheets:
        with m2:
            st.write("**Watchlist: Quiet Whales**")
            st.dataframe(data["Watch_QuietWhales"])
    if "Bursty_Games" in available_sheets:
        with m3:
            st.write("**Bursty Games**")
            st.dataframe(data["Bursty_Games"])
    st.markdown("---")

# -----------------------------------------------------------
# 6. Player Drill-Down (optional, uses raw CSV if present)
# -----------------------------------------------------------
st.subheader("Player Drill-Down: Individual Betting History")

if st.checkbox("Enable player-level view (loads raw CSV)"):
    try:
        # You can rename to train_matrix_feature.csv if that’s your raw file
        full_df = pd.read_csv("df_train_clean.csv", parse_dates=['date'])
        st.write(full_df.head())

    except FileNotFoundError:
        st.error("Place df_train_clean.csv (or your raw detailed dataset) in this folder to enable drill-down.")
    else:
        needed = {'mask_id','date','event_description_norm','wager_amount'}
        if not needed.issubset(full_df.columns):
            st.error("Raw CSV must include mask_id, date, event_description_norm, wager_amount.")
        else:
            # Compute 30-day rolling avg wager if missing
            if 'avg_wager_30d' not in full_df.columns:
                full_df = full_df.sort_values(['mask_id','date'])
                full_df['avg_wager_30d'] = (
                    full_df
                    .groupby('mask_id', group_keys=False)
                    .apply(lambda x: x.set_index('date')['wager_amount']
                           .rolling('30D').mean())
                    .reset_index(level=0, drop=True)
                    .reindex(full_df.index)
                )

            mask_ids = sorted(full_df['mask_id'].unique())
            selected_id = st.selectbox("Select or enter a mask_id:", mask_ids)

            user_df = full_df[full_df['mask_id'] == selected_id].sort_values('date')

            if user_df.empty:
                st.warning("No records found for this mask_id.")
            else:
                st.write(f"### Betting History for mask_id: {selected_id}")
                st.dataframe(user_df[['date','event_description_norm','wager_amount','avg_wager_30d']])

                # Daily betting frequency
                st.write("**Bet Frequency Over Time**")
                daily_bets = user_df.groupby(user_df['date'].dt.date)['mask_id'].count()
                st.line_chart(daily_bets)

                # Rolling 30-day average wager
                st.write("**Average Wager (30-Day Rolling)**")
                st.line_chart(user_df.set_index('date')['avg_wager_30d'])



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
