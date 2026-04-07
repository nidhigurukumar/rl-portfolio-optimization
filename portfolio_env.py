
import numpy as np
import pandas as pd
import gymnasium as gym


class PortfolioEnv(gym.Env):
    """
    Continuous-action portfolio environment for PPO/A2C.

    - Observation: all numeric feature columns except the date column (dt)
    - Action: allocation weights over [sp500, djia, hsi] in [0, 1], normalized to sum to 1
    - Reward: log change in portfolio value minus simple transaction costs and a small risk penalty
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, data: pd.DataFrame, tickers=None, transaction_cost: float = 0.001):
        super().__init__()
        if tickers is None:
            tickers = ["sp500", "djia", "hsi"]
        self.tickers = list(tickers)
        self.transaction_cost = float(transaction_cost)

        self.data = data.copy().reset_index(drop=True)

        # numeric feature columns = everything except dt
        self.feature_cols = [c for c in self.data.columns if c != "dt"]
        self.obs_dim = len(self.feature_cols)

        self.initial_value = 100000.0
        self.portfolio_value = self.initial_value
        self.current_step = 1  # start from 1 so we have previous step for returns

        self.prev_action = np.ones(len(self.tickers), dtype=np.float32) / len(self.tickers)

        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(self.tickers),),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

    def _get_obs(self):
        row = self.data.iloc[self.current_step]
        obs = row[self.feature_cols].to_numpy(dtype=np.float32)
        return obs

    def _get_returns_vector(self):
        ret_cols = []
        for t in self.tickers:
            c = f"{t}_ret"
            if c in self.data.columns:
                ret_cols.append(c)
        if len(ret_cols) == len(self.tickers):
            rets = self.data[ret_cols].iloc[self.current_step].to_numpy(dtype=np.float32)
            return rets

        prices = self.data[self.tickers].iloc[self.current_step].to_numpy(dtype=np.float32)
        prev_prices = self.data[self.tickers].iloc[self.current_step - 1].to_numpy(dtype=np.float32)
        rets = (prices - prev_prices) / (prev_prices + 1e-8)
        return rets

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 1
        self.portfolio_value = self.initial_value
        self.prev_action = np.ones(len(self.tickers), dtype=np.float32) / len(self.tickers)
        obs = self._get_obs()
        info = {"portfolio_value": float(self.portfolio_value)}
        return obs, info

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, 0.0, 1.0)
        if action.sum() > 0:
            action = action / action.sum()
        else:
            action = np.ones_like(action) / len(self.tickers)

        turnover = np.abs(action - self.prev_action).sum()
        cost = self.transaction_cost * turnover

        prev_val = self.portfolio_value

        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        truncated = False

        rets = self._get_returns_vector()
        portfolio_ret = float(np.dot(action, rets))
        self.portfolio_value *= (1.0 + portfolio_ret)

        risk_penalty = 0.1 * (portfolio_ret ** 2)

        if self.portfolio_value <= 0:
            reward = -10.0
            terminated = True
        else:
            reward = float(np.log(self.portfolio_value / (prev_val + 1e-8)) - cost - risk_penalty)

        self.prev_action = action.copy()
        obs = self._get_obs()
        info = {
            "portfolio_value": float(self.portfolio_value),
            "portfolio_return": float(portfolio_ret),
            "transaction_cost": float(cost),
        }

        return obs, reward, terminated, truncated, info


class DQNPortfolioEnv(gym.Env):
    """
    Discrete-action portfolio environment for DQN.

    - Action: choose from a finite set of allocation patterns (27 combinations).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, data: pd.DataFrame, tickers=None, transaction_cost: float = 0.001):
        super().__init__()
        if tickers is None:
            tickers = ["sp500", "djia", "hsi"]
        self.tickers = list(tickers)
        self.transaction_cost = float(transaction_cost)

        self.data = data.copy().reset_index(drop=True)

        self.feature_cols = [c for c in self.data.columns if c != "dt"]
        self.obs_dim = len(self.feature_cols)

        self.initial_value = 100000.0
        self.portfolio_value = self.initial_value
        self.current_step = 1

        self.prev_action_vec = np.ones(len(self.tickers), dtype=np.float32) / len(self.tickers)

        # 27 combos: weights in {0, 0.5, 1} for each asset, normalized
        self.action_map = []
        for w1 in [0.0, 0.5, 1.0]:
            for w2 in [0.0, 0.5, 1.0]:
                for w3 in [0.0, 0.5, 1.0]:
                    weights = np.array([w1, w2, w3], dtype=np.float32)
                    if weights.sum() == 0:
                        continue
                    weights = weights / weights.sum()
                    self.action_map.append(weights)

        self.action_space = gym.spaces.Discrete(len(self.action_map))

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

    def _get_obs(self):
        row = self.data.iloc[self.current_step]
        obs = row[self.feature_cols].to_numpy(dtype=np.float32)
        return obs

    def _get_returns_vector(self):
        ret_cols = []
        for t in self.tickers:
            c = f"{t}_ret"
            if c in self.data.columns:
                ret_cols.append(c)
        if len(ret_cols) == len(self.tickers):
            rets = self.data[ret_cols].iloc[self.current_step].to_numpy(dtype=np.float32)
            return rets

        prices = self.data[self.tickers].iloc[self.current_step].to_numpy(dtype=np.float32)
        prev_prices = self.data[self.tickers].iloc[self.current_step - 1].to_numpy(dtype=np.float32)
        rets = (prices - prev_prices) / (prev_prices + 1e-8)
        return rets

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 1
        self.portfolio_value = self.initial_value
        self.prev_action_vec = np.ones(len(self.tickers), dtype=np.float32) / len(self.tickers)
        obs = self._get_obs()
        info = {"portfolio_value": float(self.portfolio_value)}
        return obs, info

    def step(self, action):
        action_idx = int(action)
        action_vec = self.action_map[action_idx]
        action_vec = np.clip(action_vec, 0.0, 1.0)
        if action_vec.sum() > 0:
            action_vec = action_vec / action_vec.sum()
        else:
            action_vec = np.ones_like(action_vec) / len(self.tickers)

        turnover = np.abs(action_vec - self.prev_action_vec).sum()
        cost = self.transaction_cost * turnover

        prev_val = self.portfolio_value

        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        truncated = False

        rets = self._get_returns_vector()
        portfolio_ret = float(np.dot(action_vec, rets))
        self.portfolio_value *= (1.0 + portfolio_ret)

        risk_penalty = 0.1 * (portfolio_ret ** 2)

        if self.portfolio_value <= 0:
            reward = -10.0
            terminated = True
        else:
            reward = float(np.log(self.portfolio_value / (prev_val + 1e-8)) - cost - risk_penalty)

        self.prev_action_vec = action_vec.copy()
        obs = self._get_obs()
        info = {
            "portfolio_value": float(self.portfolio_value),
            "portfolio_return": float(portfolio_ret),
            "transaction_cost": float(cost),
        }
        return obs, reward, terminated, truncated, info
