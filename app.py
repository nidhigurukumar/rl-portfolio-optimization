
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from stable_baselines3 import DQN, A2C, PPO
from portfolio_env import PortfolioEnv, DQNPortfolioEnv

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data_prepared")
MODELS_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(page_title="RL Portfolio Dashboard", layout="wide")

st.title("📊 Deep RL for Dynamic Stock Portfolio Optimization")

st.markdown(
    """
    This dashboard uses **DQN (discrete)**, **A2C**, and **PPO** to optimize allocations
    over macro indices (S&P 500, DJIA, HSI) derived from your stock dataset.
    It includes EDA, model summaries, and evaluation metrics.
    """
)

# --------- Load data ---------
@st.cache_data
def load_splits():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    return train, val, test

train_df, val_df, test_df = load_splits()

# --------- Helper: risk metrics ---------
def sharpe_ratio(values, risk_free=0.0):
    v = pd.Series(values)
    rets = v.pct_change().dropna()
    if rets.std() == 0:
        return 0.0
    excess = rets - risk_free / 252.0
    return float(np.sqrt(252) * excess.mean() / excess.std())

def max_drawdown(values):
    v = pd.Series(values)
    cummax = v.cummax()
    dd = (v - cummax) / cummax
    return float(dd.min())

# --------- EDA Section ---------
st.header("🔍 Exploratory Data Analysis (EDA)")

eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Overview", "Distributions", "Correlations"])

with eda_tab1:
    st.subheader("Data Preview (Train/Test)")
    st.write("**Train Set (head):**")
    st.dataframe(train_df.head())
    st.write("**Test Set (head):**")
    st.dataframe(test_df.head())

    st.subheader("Basic Statistics (Test Set)")
    st.dataframe(test_df.describe())

    st.subheader("Index Time Series (Test Set)")
    fig, ax = plt.subplots()
    ax.plot(pd.to_datetime(test_df["dt"]), test_df["sp500"], label="S&P 500")
    ax.plot(pd.to_datetime(test_df["dt"]), test_df["djia"], label="DJIA")
    ax.plot(pd.to_datetime(test_df["dt"]), test_df["hsi"], label="HSI")
    ax.set_xlabel("Date")
    ax.set_ylabel("Index Level")
    ax.legend()
    st.pyplot(fig)

with eda_tab2:
    st.subheader("Return Distributions (Test Set)")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col in zip(axes, ["sp500_ret", "djia_ret", "hsi_ret"]):
        ax.hist(test_df[col], bins=50)
        ax.set_title(col)
    fig.tight_layout()
    st.pyplot(fig)

with eda_tab3:
    st.subheader("Feature Correlation Heatmap (Test Set)")
    numeric_cols = test_df.select_dtypes(include=[float, int]).columns
    corr = test_df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr, aspect="auto")
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=90, fontsize=6)
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_yticklabels(numeric_cols, fontsize=6)
    fig.colorbar(cax)
    st.pyplot(fig)

st.markdown("---")

# --------- Load models (if they exist) ---------
@st.cache_resource
def load_models():
    models = {}
    dqn_path = os.path.join(MODELS_DIR, "DQN_portfolio.zip")
    a2c_path = os.path.join(MODELS_DIR, "A2C_portfolio.zip")
    ppo_path = os.path.join(MODELS_DIR, "PPO_portfolio.zip")

    if os.path.exists(dqn_path):
        models["DQN"] = DQN.load(dqn_path, device="cpu")
    if os.path.exists(a2c_path):
        models["A2C"] = A2C.load(a2c_path, device="cpu")
    if os.path.exists(ppo_path):
        models["PPO"] = PPO.load(ppo_path, device="cpu")
    return models

models = load_models()
available_models = list(models.keys())

if not available_models:
    st.warning(
        "No trained models found in the 'models/' directory. "
        "Please run `python train_rl.py` first to train DQN, A2C, and PPO."
    )

# --------- Simulation & Metrics ---------
def simulate_model(model_name: str):
    model = models[model_name]
    if model_name == "DQN":
        env = DQNPortfolioEnv(test_df, tickers=["sp500", "djia", "hsi"], transaction_cost=0.001)
    else:
        env = PortfolioEnv(test_df, tickers=["sp500", "djia", "hsi"], transaction_cost=0.001)

    obs, info = env.reset()
    done = False
    portfolio_values = [info["portfolio_value"]]
    portfolio_returns = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        portfolio_values.append(info["portfolio_value"])
        portfolio_returns.append(info.get("portfolio_return", 0.0))
        if len(portfolio_values) > len(test_df) + 5:
            break

    return pd.Series(portfolio_values, name=model_name), pd.Series(portfolio_returns, name=f"{model_name}_ret")

# Baseline equal-weight portfolio for classification metrics
def baseline_equal_weight_returns(df):
    rets = df[["sp500_ret", "djia_ret", "hsi_ret"]].mean(axis=1)
    return pd.Series(rets, name="baseline_ret")

baseline_ret = baseline_equal_weight_returns(test_df)

st.header("🤖 Model Evaluation & Simulation")

mode = st.sidebar.radio("Mode", ["Single Model", "Compare All"])

def get_model_params_summary(model):
    summary = {
        "Algorithm": type(model).__name__,
        "Gamma": getattr(model, "gamma", None),
        "Learning Rate": getattr(model, "learning_rate", None),
    }
    for attr in ["n_steps", "batch_size", "buffer_size", "train_freq"]:
        if hasattr(model, attr):
            summary[attr] = getattr(model, attr)
    return summary

if available_models:
    if mode == "Single Model":
        choice = st.sidebar.selectbox("Select Model", available_models)
        st.subheader(f"📈 {choice} Portfolio Simulation (Test Set)")

        with st.spinner(f"Simulating {choice}..."):
            values, rets = simulate_model(choice)

        st.line_chart(values)

        sharpe = sharpe_ratio(values)
        mdd = max_drawdown(values)
        final_val = values.iloc[-1]

        aligned_len = min(len(rets), len(baseline_ret))
        y_true = (baseline_ret.iloc[:aligned_len] > 0).astype(int)
        y_pred = (rets.iloc[:aligned_len] > 0).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Portfolio Value", f"${final_val:,.2f}")
        with col2:
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        with col3:
            st.metric("Max Drawdown", f"{mdd:.2%}")

        st.subheader("📊 Classification-style Metrics (vs. Equal-Weight Baseline)")
        st.write(
            {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
            }
        )

        st.subheader("⚙️ Model Hyperparameters")
        st.json(get_model_params_summary(models[choice]))

    else:
        st.subheader("🔍 Model Comparison — Test Set Portfolio Values & Metrics")
        results_values = {}
        results_returns = {}
        metrics_summary = {}

        with st.spinner("Simulating all available models..."):
            for name in available_models:
                vals, rets = simulate_model(name)
                results_values[name] = vals
                results_returns[name] = rets

                sharpe = sharpe_ratio(vals)
                mdd = max_drawdown(vals)
                final_val = vals.iloc[-1]

                aligned_len = min(len(rets), len(baseline_ret))
                y_true = (baseline_ret.iloc[:aligned_len] > 0).astype(int)
                y_pred = (rets.iloc[:aligned_len] > 0).astype(int)
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                metrics_summary[name] = {
                    "Final Value": final_val,
                    "Sharpe": sharpe,
                    "Max Drawdown": mdd,
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1": f1,
                }

        aligned_df = pd.concat(results_values, axis=1)
        st.line_chart(aligned_df)

        st.subheader("📋 Metrics Summary")
        st.table(pd.DataFrame(metrics_summary).T)

        st.subheader("⚙️ Model Hyperparameters")
        for name in available_models:
            st.markdown(f"**{name}**")
            st.json(get_model_params_summary(models[name]))
