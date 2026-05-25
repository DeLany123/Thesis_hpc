"""
HPC Benchmark Script — Train & Evaluate a Single Agent With Observation/Reward Scaling
========================================================================================

Drop-in companion to benchmark_agent.py. Identical parameters, adds two new arguments:

    --scaling        : Which scaling to apply. One of {obs, rew, both}
                         obs  — Observation scaling only (RobustScalingWrapper)
                         rew  — Reward scaling only     (RunningRewardScaler)
                         both — Observation + reward scaling
    --scaling-method : Observation scaling method. One of {standard, robust, minmax_clipped}
                       (default: robust). Ignored when --scaling rew is used.

Usage examples::

    python benchmark_agent_scaled.py --agent SAC  --steps 500000  --scaling obs
    python benchmark_agent_scaled.py --agent PPO  --steps 1000000 --scaling rew  --iterations 4
    python benchmark_agent_scaled.py --agent DQN  --steps 750000  --scaling both --scaling-method standard
    python benchmark_agent_scaled.py --agent SAC  --steps 500000  --scaling obs  --sequential

Output naming (one CSV per fold + one combined summary):
    Per-fold  : {agent}_{steps}_{scaling}scaled_fold{fold}_{iter}iter_results.csv
    Summary   : {agent}_{steps}_{scaling}scaled_results.csv

All other arguments (--folds-path, --output-dir, --k-folds, --days-per-ep,
--battery-cap, --charge-rate, --cycle-cost, --sequential) behave identically
to benchmark_agent.py and are fully plug-and-play.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Literal
import torch

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN, A2C, SAC, DDPG
from stable_baselines3.common.base_class import BaseAlgorithm


# ═══════════════════════════════════════════════════════════════════════
#  AGENT REGISTRY  (identical to benchmark_agent.py)
# ═══════════════════════════════════════════════════════════════════════
AGENT_REGISTRY: Dict[str, Tuple[type, str, bool]] = {
    "PPO":  (PPO,  "MlpPolicy", False),
    "DQN":  (DQN,  "MlpPolicy", False),
    "A2C":  (A2C,  "MlpPolicy", False),
    "SAC":  (SAC,  "MlpPolicy", True),
    "DDPG": (DDPG, "MlpPolicy", True),
}


# ═══════════════════════════════════════════════════════════════════════
#  BASE ENVIRONMENT  (identical to benchmark_agent.py)
# ═══════════════════════════════════════════════════════════════════════
class BaseBatteryEnv(gym.Env):
    """
    Base battery trading environment.
    Shared simulation logic; subclasses implement _get_observation().
    """

    def __init__(
            self,
            battery_capacity_mwh: float,
            charge_discharge_rate_mw: float,
            all_data: pd.DataFrame,
            days_per_episode: int = 1,
            cycle_cost_eur: float = 6.25
    ):
        super().__init__()

        self.battery_capacity_mwh = battery_capacity_mwh
        self.charge_discharge_rate = charge_discharge_rate_mw
        self.all_data = all_data
        self.prices = all_data['Imbalance Price'].to_numpy()
        self.time_interval = 1 / 60
        self.max_steps = len(self.prices)
        self.days_per_episode = days_per_episode
        self.cycle_cost_eur = cycle_cost_eur

        throughput_per_cycle = 2 * self.battery_capacity_mwh
        self.marginal_cost_per_mwh = self.cycle_cost_eur / throughput_per_cycle

        self.daily_start_indices = self.all_data.groupby(
            self.all_data['Datetime'].dt.date
        ).head(1).index.tolist()
        self.start_day_counter = 0
        self.current_episode_end_step = 0

        self.action_space = gym.spaces.Discrete(3)  # 0: Idle, 1: Charge, 2: Discharge

        self.current_step = 0
        self.soc_mwh = 0.0
        self.total_energy_traded_per_quarter = 0.0
        self.total_charged_in_quarter = 0.0
        self.total_discharged_in_quarter = 0.0

    def _get_observation(self) -> np.ndarray:
        raise NotImplementedError

    def _get_power_rate_from_action(self, action: int) -> float:
        if action == 0:
            return 0.0
        elif action == 1:
            return self.charge_discharge_rate
        elif action == 2:
            return -self.charge_discharge_rate
        else:
            raise ValueError(f"Invalid action {action}")

    def _calculate_delayed_reward(self) -> float:
        if self.all_data['Datetime'].iloc[self.current_step].minute % 15 == 14:
            revenue = -self.prices[self.current_step] * self.total_energy_traded_per_quarter
            throughput = self.total_charged_in_quarter + self.total_discharged_in_quarter
            degradation_cost = throughput * self.marginal_cost_per_mwh
            return revenue - degradation_cost
        return 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.start_day_counter >= len(self.daily_start_indices):
            self.start_day_counter = 0
        self.current_step = self.daily_start_indices[self.start_day_counter]
        end_day_index = self.start_day_counter + self.days_per_episode
        if end_day_index >= len(self.daily_start_indices):
            self.current_episode_end_step = self.max_steps - 1
        else:
            self.current_episode_end_step = self.daily_start_indices[end_day_index] - 1
        self.start_day_counter += self.days_per_episode
        self.soc_mwh = 0.0
        self.total_energy_traded_per_quarter = 0.0
        self.total_charged_in_quarter = 0.0
        self.total_discharged_in_quarter = 0.0
        return self._get_observation(), {}

    def step(self, action: int):
        if self.all_data['Datetime'].iloc[self.current_step].minute % 15 == 0:
            self.total_energy_traded_per_quarter = 0.0
            self.total_charged_in_quarter = 0.0
            self.total_discharged_in_quarter = 0.0

        power_rate = self._get_power_rate_from_action(action)
        intended_energy_trade = power_rate * self.time_interval

        actual_energy_traded = 0.0
        if intended_energy_trade > 0:
            actual_energy_traded = min(intended_energy_trade, self.battery_capacity_mwh - self.soc_mwh)
        elif intended_energy_trade < 0:
            actual_energy_traded = max(intended_energy_trade, -self.soc_mwh)

        self.soc_mwh += actual_energy_traded
        self.total_energy_traded_per_quarter += actual_energy_traded
        if actual_energy_traded > 0:
            self.total_charged_in_quarter += actual_energy_traded
        elif actual_energy_traded < 0:
            self.total_discharged_in_quarter += abs(actual_energy_traded)

        reward = self._calculate_delayed_reward()

        episode_done = self.current_step >= self.current_episode_end_step
        data_done = self.current_step >= self.max_steps - 1
        terminated = episode_done or data_done
        obs = self._get_observation()

        self.current_step += 1
        info = {
            'energy_charged_discharged': actual_energy_traded,
            'real_reward': reward,   # always the unscaled financial reward
        }
        return obs, reward, terminated, False, info

    def action_masks(self) -> np.ndarray:
        mask = [1, 1, 1]
        epsilon = 1e-6
        if self.soc_mwh >= self.battery_capacity_mwh - epsilon: mask[1] = 0
        if self.soc_mwh <= epsilon: mask[2] = 0
        return np.array(mask, dtype=np.int8)

    def get_idle_action(self) -> int:
        return 0


class ExtendedBatteryEnv(BaseBatteryEnv):
    """
    Observation: [SoC, Current Price, Total Charged in Quarter, Total Discharged in Quarter]
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        low_bounds  = np.array([0.0, -np.inf, 0.0, 0.0], dtype=np.float32)
        high_bounds = np.array([self.battery_capacity_mwh, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=low_bounds, high=high_bounds, shape=(4,), dtype=np.float32
        )

    def _get_observation(self) -> np.ndarray:
        return np.array([
            self.soc_mwh,
            self.prices[self.current_step],
            self.total_charged_in_quarter,
            self.total_discharged_in_quarter,
        ], dtype=np.float32)


class ContinuousActionWrapper(gym.ActionWrapper):
    """Maps continuous scalar in [-1, 1] → discrete {0, 1, 2} for SAC/DDPG."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def action(self, continuous_action):
        if continuous_action < -0.33:
            return 2   # Discharge
        elif continuous_action > 0.33:
            return 1   # Charge
        else:
            return 0   # Idle


# ═══════════════════════════════════════════════════════════════════════
#  SCALING WRAPPERS
# ═══════════════════════════════════════════════════════════════════════

class RobustScalingWrapper(gym.ObservationWrapper):
    """
    Scales the observation vector before it is passed to the neural network.

    Features with fixed, known physical bounds (SoC, energy traded per quarter)
    are scaled via min-max normalisation to [0, 1]. The imbalance price feature
    has no fixed bounds; it is scaled using one of three methods that are
    fitted on the training data distribution passed through env.prices:

        'standard'       — zero-mean, unit-variance (z-score)
        'robust'         — median-centred, IQR-scaled (default; robust to spikes)
        'minmax_clipped' — 1st–99th percentile min-max with outlier clipping
    """

    def __init__(
            self,
            env: BaseBatteryEnv,
            method: Literal['standard', 'robust', 'minmax_clipped'] = 'robust',
    ):
        super().__init__(env)
        self.method = method

        self.soc_min   = 0.0
        self.soc_max   = env.battery_capacity_mwh
        self.trade_min = 0.0
        self.trade_max = env.charge_discharge_rate * (15 / 60)

        price_series = env.prices

        if self.method == 'standard':
            self.price_param1 = price_series.mean()
            self.price_param2 = price_series.std()
        elif self.method == 'robust':
            self.price_param1 = np.median(price_series)
            q1 = np.quantile(price_series, 0.25)
            q3 = np.quantile(price_series, 0.75)
            self.price_param2 = q3 - q1
        elif self.method == 'minmax_clipped':
            p01 = np.quantile(price_series, 0.01)
            p99 = np.quantile(price_series, 0.99)
            self.price_param1 = p01
            self.price_param2 = p99 - p01
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        if self.price_param2 == 0:
            self.price_param2 = 1e-8

        # Observation space bounds stay Box(-inf, inf) to avoid clamp issues
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        scaled = np.copy(obs)

        # Min-max scale bounded features to [0, 1]
        scaled[0] = (obs[0] - self.soc_min) / (self.soc_max - self.soc_min + 1e-8)
        scaled[2] = (obs[2] - self.trade_min) / (self.trade_max - self.trade_min + 1e-8)
        scaled[3] = (obs[3] - self.trade_min) / (self.trade_max - self.trade_min + 1e-8)
        scaled[[0, 2, 3]] = np.clip(scaled[[0, 2, 3]], 0.0, 1.0)

        # Scale price feature
        price = obs[1]
        if self.method == 'standard':
            scaled[1] = (price - self.price_param1) / self.price_param2
        elif self.method == 'robust':
            scaled[1] = (price - self.price_param1) / self.price_param2
        elif self.method == 'minmax_clipped':
            clipped = np.clip(price, self.price_param1, self.price_param1 + self.price_param2)
            scaled[1] = (clipped - self.price_param1) / self.price_param2

        return scaled.astype(np.float32)

    def action_masks(self):
        return self.env.action_masks()


class RunningRewardScaler(gym.RewardWrapper):
    """
    Normalises the reward signal online using a running estimate of its
    standard deviation (Welford's algorithm). Because the minimum and maximum
    imbalance reward are not known in advance, a fixed scaling factor cannot be
    pre-computed; the scaler instead adapts as training data accumulates.

    Only the standard deviation is tracked (the mean is not subtracted) to
    preserve the sign of the reward, which carries economically important
    information: positive rewards represent net profit, negative rewards losses.
    Rewards are additionally clipped to [-clip, +clip] after normalisation to
    prevent extreme outliers from destabilising the gradient update.

    The unscaled financial reward remains accessible via info['real_reward']
    (set by BaseBatteryEnv.step) and is therefore unaffected by this wrapper
    for evaluation purposes.
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8, clip: float = 10.0):
        super().__init__(env)
        self._count   = 0
        self._mean    = 0.0
        self._M2      = 0.0       # sum of squared deviations (Welford)
        self.epsilon  = epsilon
        self.clip     = clip

    def reward(self, reward: float) -> float:
        # Update running variance (Welford's online algorithm)
        self._count += 1
        delta = reward - self._mean
        self._mean += delta / self._count
        delta2 = reward - self._mean
        self._M2 += delta * delta2

        if self._count < 2:
            return float(np.clip(reward, -self.clip, self.clip))

        variance = self._M2 / (self._count - 1)
        std = np.sqrt(variance + self.epsilon)
        scaled = reward / std
        return float(np.clip(scaled, -self.clip, self.clip))

    def action_masks(self):
        return self.env.action_masks()


# ═══════════════════════════════════════════════════════════════════════
#  EVALUATION RESULT  (identical to benchmark_agent.py)
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class EvaluationResult:
    prices: List[float]
    soc: List[float]
    total_charged_per_quarter: List[float]
    total_discharged_per_quarter: List[float]
    actions: List[int]
    scaled_rewards: List[float]
    real_rewards: List[float]
    energy_charged_discharged: List[float]
    episodic_rewards: List[float]

    def to_pandas(self):
        return pd.DataFrame({
            "prices":                        self.prices,
            "soc":                           self.soc,
            "total_charged_per_quarter":     self.total_charged_per_quarter,
            "total_discharged_per_quarter":  self.total_discharged_per_quarter,
            "actions":                       self.actions,
            "scaled_rewards":                self.scaled_rewards,
            "real_rewards":                  self.real_rewards,
            "energy_charged_discharged":     self.energy_charged_discharged,
        })


# ═══════════════════════════════════════════════════════════════════════
#  EVALUATION LOOP  (identical to benchmark_agent.py)
# ═══════════════════════════════════════════════════════════════════════
def run_evaluation(
        scaled_env: gym.Env,
        model: BaseAlgorithm,
        is_masked: bool = True,
        number_of_episodes: int = 1,
) -> EvaluationResult:
    unwrapped_env: BaseBatteryEnv = scaled_env.unwrapped

    prices_history, soc_history = [], []
    total_charged_history, total_discharged_history = [], []
    action_history, scaled_reward_history = [], []
    real_reward_history, energy_history, episodic_rewards = [], [], []

    for episode_num in range(number_of_episodes):
        print(f"Starting episode {episode_num + 1}/{number_of_episodes}")
        obs, info = scaled_env.reset()

        start_time = unwrapped_env.all_data.iloc[unwrapped_env.current_step]['Datetime']
        end_time   = unwrapped_env.all_data.iloc[unwrapped_env.current_episode_end_step]['Datetime']
        print(f"From {start_time} to {end_time}")

        done = False
        reward_per_episode = 0

        while not done:
            action_mask = unwrapped_env.action_masks()
            if is_masked:
                action, _ = model.predict(obs, deterministic=True, action_masks=action_mask)
            else:
                action, _ = model.predict(obs)

            obs, reward, terminated, truncated, info = scaled_env.step(action)

            prices_history.append(unwrapped_env.prices[unwrapped_env.current_step - 1])
            soc_history.append(unwrapped_env.soc_mwh)
            total_charged_history.append(unwrapped_env.total_charged_in_quarter)
            total_discharged_history.append(unwrapped_env.total_discharged_in_quarter)
            action_history.append(action)
            scaled_reward_history.append(reward)
            real_reward_history.append(info.get('real_reward', 0))
            energy_history.append(info.get('energy_charged_discharged', 0))
            reward_per_episode += reward
            done = terminated or truncated

        episodic_rewards.append(reward_per_episode)
        print(f"Finished with total (scaled) reward: {reward_per_episode:.2f}")

    return EvaluationResult(
        prices=prices_history,
        soc=soc_history,
        total_charged_per_quarter=total_charged_history,
        total_discharged_per_quarter=total_discharged_history,
        actions=action_history,
        scaled_rewards=scaled_reward_history,
        real_rewards=real_reward_history,
        energy_charged_discharged=energy_history,
        episodic_rewards=episodic_rewards,
    )


# ═══════════════════════════════════════════════════════════════════════
#  FOLD LOADING  (identical to benchmark_agent.py)
# ═══════════════════════════════════════════════════════════════════════
def load_folds(
    folds_path: str,
    k: int,
) -> List[Tuple[pd.DataFrame, List[pd.DataFrame], List[pd.DataFrame]]]:
    folds = []
    for i in range(k):
        train_p = os.path.join(folds_path, f"fold_{i}_train.pkl")
        val_p   = os.path.join(folds_path, f"fold_{i}_val.pkl")
        test_p  = os.path.join(folds_path, f"fold_{i}_test.pkl")
        for p in (train_p, val_p, test_p):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing fold file: {p}")
        folds.append((
            pd.read_pickle(train_p),
            pd.read_pickle(val_p),
            pd.read_pickle(test_p),
        ))
        print(f"  Fold {i}: train={len(pd.read_pickle(train_p)):,} rows, "
              f"val={len(pd.read_pickle(val_p))} eps, test={len(pd.read_pickle(test_p))} eps")
    return folds


def _apply_scaling(
        env: gym.Env,
        scaling_mode: str,
        scaling_method: str,
) -> gym.Env:
    """Wraps *env* with the requested scaling wrapper(s)."""
    if scaling_mode in ('obs', 'both'):
        env = RobustScalingWrapper(env, method=scaling_method)
    if scaling_mode in ('rew', 'both'):
        env = RunningRewardScaler(env)
    return env


# ═══════════════════════════════════════════════════════════════════════
#  SINGLE-FOLD WORKER
# ═══════════════════════════════════════════════════════════════════════
def _run_fold(
    fold_idx: int,
    train_df: pd.DataFrame,
    val_episodes: List[pd.DataFrame],
    agent_class: type,
    policy_name: str,
    is_continuous: bool,
    total_steps: int,
    n_iterations: int,
    days_per_episode: int,
    battery_capacity: float,
    charge_rate: float,
    cycle_cost: float,
    scaling_mode: str,
    scaling_method: str,
    output_dir: str,
    agent_name: str,
) -> Dict[str, Any]:
    """
    Train an agent *n_iterations* times from scratch on one fold and
    evaluate on its validation episodes.  Identical to benchmark_agent.py
    except that scaling wrappers are applied around each environment.
    Writes a per-fold CSV immediately on completion.
    """

    target_threads = int(os.environ.get('OMP_NUM_THREADS', 6))
    torch.set_num_threads(target_threads)

    pid = os.getpid()
    torch_threads = torch.get_num_threads()

    try:
        allowed_cores     = sorted(list(os.sched_getaffinity(0)))
        current_core_start = os.sched_getcpu()
        print(f"👀 [OBSERVE START] Fold {fold_idx} | PID: {pid} | "
              f"PyTorch Threads: {torch_threads} | "
              f"Currently on Core: {current_core_start} | "
              f"Allowed Pool: {allowed_cores} | "
              f"Scaling: {scaling_mode} ({scaling_method})")
    except AttributeError:
        print(f"👀 [OBSERVE START] Fold {fold_idx} | PID: {pid} | (Not on Linux) | "
              f"Scaling: {scaling_mode} ({scaling_method})")

    val_df_combined = pd.concat(val_episodes, ignore_index=True)

    raw_revenues: List[float] = []
    train_times: List[float]  = []

    for run in range(1, n_iterations + 1):
        # ── Build training environment ───────────────────────────────
        train_env = ExtendedBatteryEnv(
            battery_capacity_mwh=battery_capacity,
            charge_discharge_rate_mw=charge_rate,
            all_data=train_df,
            days_per_episode=days_per_episode,
            cycle_cost_eur=cycle_cost,
        )
        if is_continuous:
            train_env = ContinuousActionWrapper(train_env)
        train_env = _apply_scaling(train_env, scaling_mode, scaling_method)

        # ── Train ────────────────────────────────────────────────────
        model = agent_class(policy_name, train_env, verbose=0)
        t0 = time.time()
        model.learn(total_timesteps=total_steps, reset_num_timesteps=True)
        t_train = time.time() - t0

        # ── Build validation environment ────────────────────────────
        val_env = ExtendedBatteryEnv(
            battery_capacity_mwh=battery_capacity,
            charge_discharge_rate_mw=charge_rate,
            all_data=val_df_combined,
            days_per_episode=days_per_episode,
            cycle_cost_eur=cycle_cost,
        )
        if is_continuous:
            val_env = ContinuousActionWrapper(val_env)
        val_env = _apply_scaling(val_env, scaling_mode, scaling_method)

        result: EvaluationResult = run_evaluation(
            val_env, model,
            number_of_episodes=len(val_episodes),
            is_masked=False,
        )

        # Revenue is always evaluated on the *unscaled* financial reward
        revenue = sum(result.real_rewards)
        raw_revenues.append(revenue)
        train_times.append(t_train)

        print(f"  [Fold {fold_idx}] Run {run}/{n_iterations} | "
              f"Scaling: {scaling_mode} | "
              f"Revenue: €{revenue:,.2f}  (train {t_train:.1f}s)")

    fold_dict = {
        "fold":                fold_idx,
        "n_iterations":        n_iterations,
        "mean_revenue":        float(np.mean(raw_revenues)),
        "std_revenue":         float(np.std(raw_revenues)),
        "min_revenue":         float(np.min(raw_revenues)),
        "max_revenue":         float(np.max(raw_revenues)),
        "all_revenues":        raw_revenues,
        "mean_train_time_sec": float(np.mean(train_times)),
    }

    # ── Write per-fold CSV immediately ──────────────────────────────
    per_fold_row = {
        "agent":               agent_name,
        "total_steps":         total_steps,
        "scaling":             scaling_mode,
        "scaling_method":      scaling_method,
        "fold":                fold_idx,
        "iterations":          n_iterations,
        "mean_revenue":        round(fold_dict["mean_revenue"], 4),
        "std_revenue":         round(fold_dict["std_revenue"],  4),
        "min_revenue":         round(fold_dict["min_revenue"],  4),
        "max_revenue":         round(fold_dict["max_revenue"],  4),
        "mean_train_time_sec": round(fold_dict["mean_train_time_sec"], 2),
    }
    per_fold_name = (
        f"{agent_name}_{total_steps}_{scaling_mode}scaled"
        f"_fold{fold_idx}_{n_iterations}iter_results.csv"
    )
    per_fold_path = os.path.join(output_dir, per_fold_name)
    pd.DataFrame([per_fold_row]).to_csv(per_fold_path, index=False)
    print(f"  [Fold {fold_idx}] Per-fold result saved → {per_fold_path}")

    return fold_dict


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description=(
            "HPC benchmark (scaled): train & evaluate one RL agent with "
            "observation/reward scaling across K folds."
        )
    )
    # ── Required ────────────────────────────────────────────────────
    parser.add_argument(
        "--agent", type=str, required=True,
        choices=list(AGENT_REGISTRY.keys()),
        help="Agent algorithm to benchmark.",
    )
    parser.add_argument(
        "--steps", type=int, required=True,
        help="Total training timesteps per run.",
    )
    parser.add_argument(
        "--scaling", type=str, required=True,
        choices=["obs", "rew", "both"],
        help=(
            "Which scaling to apply: "
            "'obs' = observation only, "
            "'rew' = reward only, "
            "'both' = observation + reward."
        ),
    )
    # ── Optional (identical defaults to benchmark_agent.py) ─────────
    parser.add_argument(
        "--scaling-method", type=str, default="robust",
        choices=["standard", "robust", "minmax_clipped"],
        help=(
            "Observation scaling method (ignored when --scaling rew). "
            "Default: robust (median-centred, IQR-scaled)."
        ),
    )
    parser.add_argument(
        "--iterations", type=int, default=3,
        help="Independent train-from-scratch runs per fold (default: 3).",
    )
    parser.add_argument(
        "--folds-path", type=str,
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Directory containing fold_*_{train,val,test}.pkl files.",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "results"),
        help="Directory to write result CSVs (default: ./results).",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Run folds one by one instead of in parallel.",
    )
    parser.add_argument("--k-folds",     type=int,   default=5)
    parser.add_argument("--days-per-ep", type=int,   default=4)
    parser.add_argument("--battery-cap", type=float, default=10.0)
    parser.add_argument("--charge-rate", type=float, default=5.0)
    parser.add_argument("--cycle-cost",  type=float, default=6.25)

    args = parser.parse_args()

    agent_class, policy_name, is_continuous = AGENT_REGISTRY[args.agent]
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load folds ────────────────────────────────────────────────────
    print(f"Loading {args.k_folds} folds from '{args.folds_path}' …")
    folds = load_folds(args.folds_path, args.k_folds)

    mode_str = "SEQUENTIALLY" if args.sequential else f"PARALLEL ({args.k_folds} workers)"
    print(
        f"\nBenchmarking {args.agent} | {args.steps:,} steps | "
        f"Scaling: {args.scaling} ({args.scaling_method}) | "
        f"{args.iterations} iterations/fold | {args.k_folds} folds | Mode: {mode_str}\n"
    )

    fold_results: List[Dict[str, Any]] = [None] * args.k_folds
    global_start_time = time.time()

    shared_kwargs = dict(
        agent_class=agent_class,
        policy_name=policy_name,
        is_continuous=is_continuous,
        total_steps=args.steps,
        n_iterations=args.iterations,
        days_per_episode=args.days_per_ep,
        battery_capacity=args.battery_cap,
        charge_rate=args.charge_rate,
        cycle_cost=args.cycle_cost,
        scaling_mode=args.scaling,
        scaling_method=args.scaling_method,
        output_dir=args.output_dir,
        agent_name=args.agent,
    )

    if args.sequential:
        for fold_idx, (train_df, val_eps, test_eps) in enumerate(folds):
            fold_results[fold_idx] = _run_fold(
                fold_idx=fold_idx,
                train_df=train_df,
                val_episodes=val_eps,
                **shared_kwargs,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.k_folds) as pool:
            future_to_fold = {}
            for fold_idx, (train_df, val_eps, test_eps) in enumerate(folds):
                future = pool.submit(
                    _run_fold,
                    fold_idx=fold_idx,
                    train_df=train_df,
                    val_episodes=val_eps,
                    **shared_kwargs,
                )
                future_to_fold[future] = fold_idx

            for future in as_completed(future_to_fold):
                idx = future_to_fold[future]
                try:
                    fold_results[idx] = future.result()
                except Exception as exc:
                    print(f"  *** Fold {idx} raised an exception: {exc}")
                    raise

    total_execution_time = time.time() - global_start_time

    # ── Aggregate & save combined summary ─────────────────────────────
    rows = []
    for r in fold_results:
        rows.append({
            "agent":               args.agent,
            "total_steps":         args.steps,
            "scaling":             args.scaling,
            "scaling_method":      args.scaling_method,
            "fold":                r["fold"],
            "iterations":          r["n_iterations"],
            "mean_revenue":        round(r["mean_revenue"], 4),
            "std_revenue":         round(r["std_revenue"],  4),
            "min_revenue":         round(r["min_revenue"],  4),
            "max_revenue":         round(r["max_revenue"],  4),
            "mean_train_time_sec": round(r["mean_train_time_sec"], 2),
        })

    df_results = pd.DataFrame(rows)

    summary_name = f"{args.agent}_{args.steps}_{args.scaling}scaled_results.csv"
    summary_path = os.path.join(args.output_dir, summary_name)
    df_results.to_csv(summary_path, index=False)

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 68}")
    print(f"  RESULTS — {args.agent} @ {args.steps:,} steps | "
          f"scaling={args.scaling} ({args.scaling_method})")
    print(f"{'=' * 68}")
    print(df_results.to_string(index=False))

    overall_mean = df_results["mean_revenue"].mean()
    overall_std  = df_results["mean_revenue"].std()
    print(f"\nCross-fold mean revenue : €{overall_mean:,.2f}  ± €{overall_std:,.2f}")
    print(f"Total wall time         : {total_execution_time:.1f}s")
    print(f"Summary results saved to: {summary_path}")


if __name__ == "__main__":
    main()