import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_lightweight_charts import st_lightweight_charts
from ipda_simulation_app import IPDASimulator  # your class file
import time

st.set_page_config(page_title="IPDA SOS-27-X Simulator", layout="wide", page_icon="📈")

st.title("📊 IPDA Simulation Dashboard — SOS-27-X Live Evaluation")
st.markdown("**High-fidelity replay with real-time SOS-27-X confidence & governance**")

# ====================== SESSION STATE ======================
if "sim" not in st.session_state:
    st.session_state.sim = None
if "logs" not in st.session_state:
    st.session_state.logs = None
if "df" not in st.session_state:
    st.session_state.df = None

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("🎛️ Simulation Controls")

    uploaded_file = st.file_uploader("Upload Dukascopy CSV (M1 or tick)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success(f"✅ Loaded {len(df):,} ticks")

    conf_threshold = st.slider("SOS-27-X Confidence Threshold", 0.50, 0.95, 0.72, 0.01)
    risk_pct = st.slider("Risk per Trade (%)", 0.1, 5.0, 1.0, 0.1) / 100
    max_steps = st.number_input("Max ticks to simulate", 1000, 500000, 50000, step=5000)

    run_button = st.button("🚀 RUN SIMULATION", type="primary", use_container_width=True)

# ====================== RUN SIMULATION ======================
if run_button and st.session_state.df is not None:
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(progress: float):
        progress_bar.progress(progress)
        status_text.text(f"Processing tick {(int(progress*max_steps)):,} / {max_steps:,}")

    with st.spinner("Running full IPDA pipeline..."):
        sim = IPDASimulator(
            df=st.session_state.df,
            confidence_threshold=conf_threshold,
            risk_pct=risk_pct,
        )
        sim.run_full_replay(
            start_idx=0,
            max_steps=max_steps,
            progress_callback=update_progress
        )

    # Save to session state (persistent across reruns)
    st.session_state.sim = sim
    st.session_state.logs = pd.DataFrame(sim.logs)

    progress_bar.progress(1.0)
    status_text.success("✅ Simulation Complete!")

# ====================== RESULTS (only if simulation ran) ======================
if st.session_state.sim and st.session_state.logs is not None:
    logs_df = st.session_state.logs
    sim = st.session_state.sim

    # Metrics cards
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Final Equity", f"${sim.equity:,.2f}", f"{(sim.equity/sim.initial_balance-1)*100:+.2f}%")
    col2.metric("Trades Taken", len(sim.trades))
    col3.metric("Avg Latency", f"{np.mean(sim.latency_stats):.1f} ms")
    col4.metric("Win Rate", f"{logs_df['trade_taken'].mean():.1%}")
    col5.metric("Max Drawdown", f"${sim.initial_balance - logs_df['equity'].min():,.2f}")

    tab1, tab2, tab3 = st.tabs(["📈 Live Chart (Lightweight)", "📉 Equity Curve", "🔍 SOS-27-X Insights"])

    with tab1:
        st.subheader("TradingView-style IPDA Chart")
        # Prepare data for lightweight-charts (fast & beautiful)
        candle_data = []
        for _, row in logs_df.iterrows():
            candle_data.append({
                "time": int(row['step']),  # or use real timestamp
                "open": row['price'],
                "high": row['price'] * 1.001,
                "low": row['price'] * 0.999,
                "close": row['price']
            })

        series = [
            {"type": "candlestick", "data": candle_data, "color": "green"},
            # Trade markers
            {"type": "markers", "data": [{"time": int(row['step']), "value": row['price']} 
                                         for _, row in logs_df[logs_df['direction'] > 0].iterrows()],
             "color": "lime", "shape": "triangle"},
            {"type": "markers", "data": [{"time": int(row['step']), "value": row['price']} 
                                         for _, row in logs_df[logs_df['direction'] < 0].iterrows()],
             "color": "red", "shape": "triangleDown"}
        ]

        st_lightweight_charts(
            series=series,
            chart_type="candlestick",
            width="100%",
            height=600,
            theme="dark",
            options={"timeScale": {"timeVisible": True}}
        )

    with tab2:
        st.line_chart(logs_df.set_index('step')['equity'], use_container_width=True)

    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Confidence Distribution")
            fig = px.histogram(logs_df, x='confidence', nbins=20, title="SOS-27-X Confidence")
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            st.subheader("Regime Breakdown")
            st.bar_chart(logs_df['regime'].value_counts())

        st.subheader("Raw Logs (last 100 ticks)")
        st.dataframe(logs_df.tail(100), use_container_width=True)

    # Export
    st.download_button(
        label="📥 Download Full Evaluation CSV",
        data=logs_df.to_csv(index=False),
        file_name=f"ipda_sos27x_evaluation_{int(time.time())}.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Upload your Dukascopy file and click **RUN SIMULATION** to begin.")

st.caption("Built for Roberto's SPECTRAL-OFI SENTINEL (SOS-27-X) — 2026 edition")
