
# REINFORCEMENT LEARNING Portfolio Project: ACADEMIC PROJECT

This project uses **Deep Reinforcement Learning** (DQN, A2C, PPO)
for macro index portfolio optimization on your 34-year stock dataset.

## Features

- Uses your `stock_data.csv` (sp500, djia, hsi, macro factors) with engineered features
- Custom Gymnasium environments:
  - `PortfolioEnv` (continuous actions for PPO/A2C)
  - `DQNPortfolioEnv` (discrete actions for DQN via allocation patterns)
- GPU training (GTX 1650 Ti or other CUDA-capable device) via `device="cuda"`
- EDA inside Streamlit:
  - Data preview
  - Descriptive statistics
  - Time-series plots
  - Return distributions
  - Correlation heatmap
- RL evaluation:
  - Portfolio value curves
  - Sharpe Ratio
  - Max Drawdown
  - Classification-style metrics vs equal-weight baseline:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
  - Model hyperparameter summaries


## Structure

- `data_prepared/train.csv`, `val.csv`, `test.csv` — generated from your `stock_data.csv`
- `portfolio_env.py` — custom Gymnasium environments
- `train_rl.py` — trains DQN, A2C, PPO (using GPU if available)
- `models/` — trained models are saved here
- `app.py` — Streamlit dashboard for EDA and RL evaluation

## Setup

Create and activate a virtual environment (recommended), then:

```bash
pip install -r requirements.txt
```

**Important:** For GPU training, make sure you have a CUDA-enabled PyTorch installed,
e.g. (example for CUDA 11.x on Windows):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Check that PyTorch sees your GPU:

```python
import torch
print(torch.cuda.is_available())
```

## Train models (on GPU if available)

```bash
python train_rl.py
```

This will create:

- `models/DQN_portfolio.zip`
- `models/A2C_portfolio.zip`
- `models/PPO_portfolio.zip`

## Run dashboard

```bash
streamlit run app.py
```

Then open the URL shown in your terminal (usually http://localhost:8501).

You can:

- Explore the data through the EDA tabs
- Switch between **Single Model** and **Compare All** modes
- Inspect portfolio value trajectories, Sharpe ratio, max drawdown
- See accuracy, precision, recall, F1 score vs. an equal-weight baseline
- View key hyperparameters for each model
