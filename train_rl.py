
import os
import pandas as pd
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from portfolio_env import PortfolioEnv, DQNPortfolioEnv

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_prepared")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def make_env_continuous(split: str = "train"):
    csv_path = os.path.join(DATA_DIR, f"{split}.csv")
    df = pd.read_csv(csv_path)
    env = PortfolioEnv(df, tickers=["sp500", "djia", "hsi"], transaction_cost=0.001)
    return env


def make_env_discrete(split: str = "train"):
    csv_path = os.path.join(DATA_DIR, f"{split}.csv")
    df = pd.read_csv(csv_path)
    env = DQNPortfolioEnv(df, tickers=["sp500", "djia", "hsi"], transaction_cost=0.001)
    return env


def train_all(total_timesteps: int = 150_000):
    # DQN (discrete)
    print("Training DQN on GPU (if available)...")
    dqn_env = DummyVecEnv([lambda: make_env_discrete("train")])
    dqn_model = DQN(
        "MlpPolicy",
        dqn_env,
        verbose=1,
        device="cuda",   # uses GPU if available
        learning_rate=1e-4,
        buffer_size=50_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=10_000,
    )
    dqn_model.learn(total_timesteps=total_timesteps)
    dqn_model.save(os.path.join(MODELS_DIR, "DQN_portfolio"))

    # A2C (continuous)
    print("Training A2C on GPU (if available)...")
    a2c_env = DummyVecEnv([lambda: make_env_continuous("train")])
    a2c_model = A2C(
        "MlpPolicy",
        a2c_env,
        verbose=1,
        device="cuda",
        learning_rate=7e-4,
        gamma=0.99,
        n_steps=5,
    )
    a2c_model.learn(total_timesteps=total_timesteps)
    a2c_model.save(os.path.join(MODELS_DIR, "A2C_portfolio"))

    # PPO (continuous)
    print("Training PPO on GPU (if available)...")
    ppo_env = DummyVecEnv([lambda: make_env_continuous("train")])
    ppo_model = PPO(
        "MlpPolicy",
        ppo_env,
        verbose=1,
        device="cuda",
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=2048,
        batch_size=64,
    )
    ppo_model.learn(total_timesteps=total_timesteps)
    ppo_model.save(os.path.join(MODELS_DIR, "PPO_portfolio"))


if __name__ == "__main__":
    train_all(total_timesteps=150_000)
