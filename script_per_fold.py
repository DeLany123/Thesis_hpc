"""
Single-Fold Benchmark Script — Train & Evaluate a Single Agent on One Fold
===========================================================================

This is a simplified version of benchmark_agent.py that runs exactly ONE fold
at a time (no parallelisation). Submit one PBS job per fold, or loop over folds
in a bash script.

Usage examples::

    python benchmark_single_fold.py --agent PPO --steps 500000 --fold 0
    python benchmark_single_fold.py --agent DQN --steps 1000000 --iterations 5 --fold 2
    python benchmark_single_fold.py --agent SAC --steps 250000 --fold 4 --folds-path /data/folds

Output CSV naming convention::

    <agent>_<steps>_fold<fold>_<iterations>iter_results.csv
    e.g.  PPO_500000_fold0_3iter_results.csv

Arguments:
    --agent         : One of {PPO, DQN, A2C, SAC, DDPG}
    --steps         : Total training timesteps per run
    --fold          : Which fold index to run (e.g. 0, 1, 2, 3, 4)
    --iterations    : Number of independent train-from-scratch runs (default: 3)
    --folds-path    : Directory containing cached fold pickles (fold_0_train.pkl, …)
    --output-dir    : Directory to write result CSVs into (default: ./results)
    --k-folds       : Total number of folds available on disk (default: 5, used for validation)
    --days-per-ep   : Days per episode for the environment (default: 4)
    --battery-cap   : Battery capacity in MWh (default: 10.0)
    --charge-rate   : Charge/discharge rate in MW (default: 5.0)
    --cycle-cost    : Cycle degradation cost in EUR (default: 6.25)
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN, A2C, SAC, DDPG
from stable_baselines3.common.base_class import BaseAlgorithm


# ═══════════════════════════════════════════════════════════════════════
#  AGENT REGISTRY
# ═══════════════════════════════════════════════════════════════════════
AGENT_REGISTRY: Dict[str, Tuple[type, str, bool]] = {
    "PPO":  (PPO,  "MlpPolicy", False),
    "DQN":  (DQN,  "MlpPolicy", False),
    "A2C":  (A2C,  "MlpPolicy", False),
    "SAC":  (SAC,  "MlpPolicy", True),
    "DDPG": (DDPG, "MlpPolicy", True),
}


# ═══════════════════════════════════════════════════════════════════════
#  ENVIRONMENT & WRAPPERS
# ═══════════════════════════════════════════════════════════════════════
class BaseBatteryEnv(gym.Env):
    """
    A BASE class for battery trading environments.
    Contains all the shared simulation logic for state transitions and rewards.
    Subclasses must implement _get_observation() and define observation_space.
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

        self.action_space = gym.spaces.Discrete(3)

        self.current_step = 0
        self.soc_mwh = 0.0
        self.total_energy_traded_per_quarter = 0.0
        self.total_charged_in_quarter = 0.0
        self.total_discharged_in_quarter = 0.0

    def _get_observation(self) -> np.ndarray:
        raise NotImplementedError("This method must be implemented by the subclass.")

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
            'real_reward': reward
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
    Observation space: [SoC, Current Price, Total Charged in Quarter, Total Discharged in Quarter]
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        low_bounds = np.array([0.0, -np.inf, 0.0, 0.0], dtype=np.float32)
        high_bounds = np.array([self.battery_capacity_mwh, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=low_bounds, high=high_bounds, shape=(4,), dtype=np.float32
        )

    def _get_observation(self) -> np.ndarray:
        return np.array([
            self.soc_mwh,
            self.prices[self.current_step],
            self.total_charged_in_quarter,
            self.total_discharged_in_quarter
        ], dtype=np.float32)


class ContinuousActionWrapper(gym.ActionWrapper):
    """Wraps Discrete(3) for SAC / DDPG: continuous [-1,1] → {0,1,2}."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def action(self, continuous_action):
        if continuous_action < -0.33:
            return 2
        elif continuous_action > 0.33:
            return 1
        else:
            return 0


# ═══════════════════════════════════════════════════════════════════════
#  EVALUATION RESULT
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
            "prices": self.prices,
            "soc": self.soc,
            "total_charged_per_quarter": self.total_charged_per_quarter,
            "total_discharged_per_quarter": self.total_discharged_per_quarter,
            "actions": self.actions,
            "scaled_rewards": self.scaled_rewards,
            "real_rewards": self.real_rewards,
            "energy_charged_discharged": self.energy_charged_discharged,
        })


# ═══════════════════════════════════════════════════════════════════════
#  EVALUATION LOOP
# ═══════════════════════════════════════════════════════════════════════
def run_evaluation(
        scaled_env: gym.Env,
        model: BaseAlgorithm,
        is_masked: bool = True,
        number_of_episodes: int = 1
) -> EvaluationResult:
    unwrapped_env: BaseBatteryEnv = scaled_env.unwrapped

    prices_history = []
    soc_history = []
    total_charged_per_quarter_history = []
    total_discharged_per_quarter_history = []
    action_history = []
    scaled_reward_history = []
    real_reward_history = []
    energy_charged_discharged_history = []
    episodic_rewards = []

    for episode_num in range(number_of_episodes):
        print(f"  Starting episode {episode_num + 1}/{number_of_episodes}")
        obs, info = scaled_env.reset()

        start_time = unwrapped_env.all_data.iloc[unwrapped_env.current_step]['Datetime']
        end_time = unwrapped_env.all_data.iloc[unwrapped_env.current_episode_end_step]['Datetime']
        print(f"  From {start_time} to {end_time}")

        done = False
        reward_per_episode = 0
        while not done:
            action_mask = unwrapped_env.action_masks()
            if is_masked:
                action, _states = model.predict(
                    obs, deterministic=True, action_masks=action_mask
                )
            else:
                action, _states = model.predict(obs)

            obs, reward, terminated, truncated, info = scaled_env.step(action)

            energy_charged_discharged = info.get('energy_charged_discharged', 0)
            prices_history.append(unwrapped_env.prices[unwrapped_env.current_step - 1])
            soc_history.append(unwrapped_env.soc_mwh)
            total_charged_per_quarter_history.append(unwrapped_env.total_charged_in_quarter)
            total_discharged_per_quarter_history.append(unwrapped_env.total_discharged_in_quarter)
            action_history.append(action)
            scaled_reward_history.append(reward)
            real_reward_history.append(info.get('real_reward', 0))
            reward_per_episode += reward
            energy_charged_discharged_history.append(energy_charged_discharged)

            done = terminated or truncated

        episodic_rewards.append(reward_per_episode)
        print(f"  Finished with total (scaled) reward: {reward_per_episode:.2f}")

    return EvaluationResult(
        prices=prices_history,
        soc=soc_history,
        total_charged_per_quarter=total_charged_per_quarter_history,
        total_discharged_per_quarter=total_discharged_per_quarter_history,
        actions=action_history,
        scaled_rewards=scaled_reward_history,
        real_rewards=real_reward_history,
        energy_charged_discharged=energy_charged_discharged_history,
        episodic_rewards=episodic_rewards
    )


# ═══════════════════════════════════════════════════════════════════════
#  FOLD LOADING  (single fold only)
# ═══════════════════════════════════════════════════════════════════════
def load_single_fold(
    folds_path: str,
    fold_idx: int,
) -> Tuple[pd.DataFrame, List[pd.DataFrame], List[pd.DataFrame]]:
    """Load one fold from disk and return (train_df, val_episodes, test_episodes)."""
    train_p = os.path.join(folds_path, f"fold_{fold_idx}_train.pkl")
    val_p   = os.path.join(folds_path, f"fold_{fold_idx}_val.pkl")
    test_p  = os.path.join(folds_path, f"fold_{fold_idx}_test.pkl")

    for p in (train_p, val_p, test_p):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing fold file: {p}")

    train_df  = pd.read_pickle(train_p)
    val_eps   = pd.read_pickle(val_p)
    test_eps  = pd.read_pickle(test_p)

    print(f"  Fold {fold_idx}: train={len(train_df):,} rows, "
          f"val={len(val_eps)} eps, test={len(test_eps)} eps")

    return train_df, val_eps, test_eps


# ═══════════════════════════════════════════════════════════════════════
#  SINGLE-FOLD TRAINING + EVALUATION
# ═══════════════════════════════════════════════════════════════════════
def run_fold(
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
) -> Dict[str, Any]:
    """
    Train an agent *n_iterations* times from scratch on one fold and
    evaluate on its validation episodes each time.
    """
    val_df_combined = pd.concat(val_episodes, ignore_index=True)

    raw_revenues: List[float] = []
    train_times: List[float] = []

    for run in range(1, n_iterations + 1):
        print(f"\n── Fold {fold_idx} | Iteration {run}/{n_iterations} ──")

        # ── Build training environment ────────────────────────────────
        train_env = ExtendedBatteryEnv(
            battery_capacity_mwh=battery_capacity,
            charge_discharge_rate_mw=charge_rate,
            all_data=train_df,
            days_per_episode=days_per_episode,
            cycle_cost_eur=cycle_cost,
        )
        if is_continuous:
            train_env = ContinuousActionWrapper(train_env)

        # ── Train ─────────────────────────────────────────────────────
        model = agent_class(policy_name, train_env, verbose=0)
        t0 = time.time()
        model.learn(total_timesteps=total_steps, reset_num_timesteps=True)
        t_train = time.time() - t0
        print(f"  Training finished in {t_train:.1f}s")

        # ── Evaluate on validation set ────────────────────────────────
        val_env = ExtendedBatteryEnv(
            battery_capacity_mwh=battery_capacity,
            charge_discharge_rate_mw=charge_rate,
            all_data=val_df_combined,
            days_per_episode=days_per_episode,
            cycle_cost_eur=cycle_cost,
        )
        if is_continuous:
            val_env = ContinuousActionWrapper(val_env)

        result: EvaluationResult = run_evaluation(
            val_env, model,
            number_of_episodes=len(val_episodes),
            is_masked=False,
        )

        revenue = sum(result.real_rewards)
        raw_revenues.append(revenue)
        train_times.append(t_train)
        print(f"  Validation revenue: €{revenue:,.2f}")

    return {
        "fold": fold_idx,
        "n_iterations": n_iterations,
        "mean_revenue": float(np.mean(raw_revenues)),
        "std_revenue": float(np.std(raw_revenues)),
        "min_revenue": float(np.min(raw_revenues)),
        "max_revenue": float(np.max(raw_revenues)),
        "all_revenues": raw_revenues,
        "mean_train_time_sec": float(np.mean(train_times)),
    }


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Single-fold benchmark: train & evaluate one RL agent on one fold."
    )
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
        "--fold", type=int, required=True,
        help="Fold index to run (e.g. 0, 1, 2, 3, 4).",
    )
    parser.add_argument(
        "--iterations", type=int, default=3,
        help="Independent train-from-scratch runs on this fold (default: 3).",
    )
    parser.add_argument(
        "--folds-path", type=str,
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Directory containing fold_*_{train,val,test}.pkl files.",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "results"),
        help="Directory to write result CSVs.",
    )
    parser.add_argument("--k-folds", type=int, default=5,
                        help="Total number of folds on disk (for validation only).")
    parser.add_argument("--days-per-ep", type=int, default=4)
    parser.add_argument("--battery-cap", type=float, default=10.0)
    parser.add_argument("--charge-rate", type=float, default=5.0)
    parser.add_argument("--cycle-cost", type=float, default=6.25)

    args = parser.parse_args()

    # ── Validate fold index ───────────────────────────────────────────
    if args.fold < 0 or args.fold >= args.k_folds:
        parser.error(f"--fold must be in [0, {args.k_folds - 1}], got {args.fold}")

    agent_class, policy_name, is_continuous = AGENT_REGISTRY[args.agent]
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load the single fold ──────────────────────────────────────────
    print(f"Loading fold {args.fold} from '{args.folds_path}' …")
    train_df, val_eps, test_eps = load_single_fold(args.folds_path, args.fold)

    print(f"\nBenchmarking {args.agent} | {args.steps:,} steps | "
          f"{args.iterations} iterations | Fold {args.fold}\n")

    global_start_time = time.time()

    # ── Run the single fold sequentially ──────────────────────────────
    result = run_fold(
        fold_idx=args.fold,
        train_df=train_df,
        val_episodes=val_eps,
        agent_class=agent_class,
        policy_name=policy_name,
        is_continuous=is_continuous,
        total_steps=args.steps,
        n_iterations=args.iterations,
        days_per_episode=args.days_per_ep,
        battery_capacity=args.battery_cap,
        charge_rate=args.charge_rate,
        cycle_cost=args.cycle_cost,
    )

    total_wall_time = time.time() - global_start_time

    # ── Build results DataFrame ───────────────────────────────────────
    row = {
        "agent": args.agent,
        "total_steps": args.steps,
        "fold": result["fold"],
        "iterations": result["n_iterations"],
        "mean_revenue": round(result["mean_revenue"], 4),
        "std_revenue": round(result["std_revenue"], 4),
        "min_revenue": round(result["min_revenue"], 4),
        "max_revenue": round(result["max_revenue"], 4),
        "mean_train_time_sec": round(result["mean_train_time_sec"], 2),
    }
    df_results = pd.DataFrame([row])

    # ── Save CSV with fold + iterations in the filename ───────────────
    csv_name = (
        f"{args.agent}_{args.steps}_fold{args.fold}_{args.iterations}iter_results.csv"
    )
    csv_path = os.path.join(args.output_dir, csv_name)
    df_results.to_csv(csv_path, index=False)

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print(f"  RESULTS — {args.agent} @ {args.steps:,} steps | Fold {args.fold}")
    print(f"{'=' * 64}")
    print(df_results.to_string(index=False))
    print(f"\nTotal wall time         : {total_wall_time:.1f}s")
    print(f"Results saved to        : {csv_path}")


if __name__ == "__main__":
    main()
