import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Retail Analytics Dashboard", page_icon="📊", layout="wide")

# ------------------ BACKGROUND IMAGE ------------------
st.markdown("""
    <style>
        .main {
            background-image: url("https://images.unsplash.com/photo-1612832021046-0c5f7e3e5f4e?auto=format&fit=crop&w=1920&q=80");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }

        .block-container {
            background-color: rgba(10, 15, 28, 0.85);
            padding: 2rem;
            border-radius: 12px;
            color: #f1f5f9; /* Light text */
        }

        h1, h2, h3, h4, h5, h6, p, div, span {
            color: #f1f5f9 !important;
        }

        .stMetric label, .stMetric div {
            color: #f1f5f9 !important;
        }
    </style>
""", unsafe_allow_html=True)


# ------------------ HEADER ------------------
st.title("Retail Analytics Dashboard")
st.caption("Interactive sales insights with KPIs, trends, segments, funnel, and top products.")

# ------------------ DATA GENERATION ------------------
@st.cache_data
def load_data(n_rows=3000, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    regions = ["North", "South", "East", "West"]
    segments = ["Consumer", "Corporate", "Home Office"]
    products = ["Alpha", "Bravo", "Cobalt", "Delta", "Echo", "Flux", "Gamma", "Helix"]

    df = pd.DataFrame({
        "OrderDate": rng.choice(dates, size=n_rows),
        "Region": rng.choice(regions, size=n_rows),
        "Segment": rng.choice(segments, size=n_rows),
        "Product": rng.choice(products, size=n_rows),
        "Quantity": rng.integers(1, 6, size=n_rows),
        "Discount": np.round(rng.uniform(0, 0.35, size=n_rows), 2)
    })

    # Synthetic sales/profit logic
    base = rng.gamma(shape=2.1, scale=130, size=n_rows)
    df["Sales"] = np.round(base * (1 - df["Discount"]) + df["Quantity"] * 25, 2)
    margin = 0.24 - (df["Discount"] * 0.45)
    noise = rng.normal(0, 10, size=n_rows)
    df["Profit"] = np.round(df["Sales"] * margin + noise, 2)

    df["OrderDate"] = pd.to_datetime(df["OrderDate"])
    df["Month"] = df["OrderDate"].dt.to_period("M").dt.to_timestamp()
    return df

data = load_data()

# ------------------ SIDEBAR FILTERS ------------------
st.sidebar.header("Filters")
min_d, max_d = data["OrderDate"].min().date(), data["OrderDate"].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

regions = st.sidebar.multiselect("Region", sorted(data["Region"].unique()), default=sorted(data["Region"].unique()))
segments = st.sidebar.multiselect("Segment", sorted(data["Segment"].unique()), default=sorted(data["Segment"].unique()))

if isinstance(date_range, (date, pd.Timestamp)):
    date_start, date_end = min_d, max_d
else:
    date_start, date_end = date_range

df = data[
    (data["OrderDate"] >= pd.to_datetime(date_start)) &
    (data["OrderDate"] <= pd.to_datetime(date_end)) &
    (data["Region"].isin(regions)) &
    (data["Segment"].isin(segments))
].copy()

if df.empty:
    st.warning("No data for the selected filters. Expand your date range or selections.")
    st.stop()

# ------------------ KPIs ------------------
total_sales = float(df["Sales"].sum())
orders = len(df)
aov = total_sales / orders if orders else 0.0
monthly = df.groupby("Month", as_index=False)["Sales"].sum().sort_values("Month")
mom = (
    (monthly["Sales"].iloc[-1] - monthly["Sales"].iloc[-2]) / monthly["Sales"].iloc[-2]
    if len(monthly) >= 2 else np.nan
)

# Simple funnel from orders
purchases = max(orders, 1)
leads = int(purchases * 3.5)
visitors = int(leads * 3.0)

k1, k2, k3, k4 = st.columns(4)
k1.metric("💰 Total Sales", f"₹{total_sales:,.0f}")
k2.metric("📦 Orders", f"{orders:,}")
k3.metric("🧾 Avg Order Value", f"₹{aov:,.0f}")
k4.metric("📈 MoM Growth", "—" if np.isnan(mom) else f"{mom:.1%}")

# ------------------ CHARTS ------------------
st.markdown("---")
custom_colors = ['#60a5fa', '#38bdf8', '#0ea5e9', '#1e3a8a', '#64748b']

c1, c2 = st.columns((2, 1))

with c1:
    st.subheader("Sales Trend")
    fig_trend = px.line(monthly, x="Month", y="Sales", markers=True, color_discrete_sequence=[custom_colors[3]])
    fig_trend.update_traces(line=dict(width=3))
    fig_trend.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_trend, use_container_width=True)

with c2:
    st.subheader("Segment Share")
    seg = df.groupby("Segment", as_index=False)["Sales"].sum()
    fig_seg = px.pie(seg, names="Segment", values="Sales", color_discrete_sequence=custom_colors)
    fig_seg.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_seg, use_container_width=True)

st.subheader("Sales by Region")
reg = df.groupby("Region", as_index=False)["Sales"].sum()
fig_reg = px.bar(reg, x="Region", y="Sales", color="Region", text_auto=".2s", color_discrete_sequence=custom_colors)
fig_reg.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig_reg, use_container_width=True)

st.subheader("Conversion Funnel")
funnel_df = pd.DataFrame({"Stage": ["Visitors", "Leads", "Purchases"], "Count": [visitors, leads, purchases]})
fig_fun = px.funnel(funnel_df, x="Count", y="Stage", color_discrete_sequence=[custom_colors[0]])
fig_fun.update_layout(paper_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig_fun, use_container_width=True)

# ------------------ FOOTER ------------------
st.caption("Built with Streamlit + Plotly • Demo data, no external files required.")
