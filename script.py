"""
HPC Benchmark Script — Train & Evaluate a Single Agent Across K Folds
======================================================================

Usage examples::

    python benchmark_agent.py --agent PPO --steps 500000
    python benchmark_agent.py --agent DQN --steps 1000000 --iterations 5
    python benchmark_agent.py --agent SAC --steps 250000 --folds-path /data/folds

Arguments:
    --agent         : One of {PPO, DQN, A2C, SAC, DDPG}
    --steps         : Total training timesteps per run
    --iterations    : Number of independent train-from-scratch runs per fold (default: 3)
    --folds-path    : Directory containing cached fold pickles (fold_0_train.pkl, …)
    --output-dir    : Directory to write result CSVs into (default: ./results)
    --k-folds       : Number of folds to load (default: 5)
    --days-per-ep   : Days per episode for the environment (default: 3)
    --battery-cap   : Battery capacity in MWh (default: 10.0)
    --charge-rate   : Charge/discharge rate in MW (default: 5.0)
    --cycle-cost    : Cycle degradation cost in EUR (default: 6.25)
    --sequential    : Running the folds sequentially instead of in parallel (useful for debugging or limited resources)
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import torch

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN, A2C, SAC, DDPG
from stable_baselines3.common.base_class import BaseAlgorithm


# ═══════════════════════════════════════════════════════════════════════
#  AGENT REGISTRY
# ═══════════════════════════════════════════════════════════════════════
# Maps CLI name → (SB3 class, policy name, needs continuous wrapper)
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

        # --- Core Simulation Parameters ---
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

        # Pre-calculate the starting index of each day.
        self.daily_start_indices = self.all_data.groupby(
            self.all_data['Datetime'].dt.date
        ).head(1).index.tolist()
        # This counter now tracks which *day* we start the episode on.
        self.start_day_counter = 0
        # This will store the calculated end step for the current episode.
        self.current_episode_end_step = 0

        # --- Fixed Action Space ---
        self.action_space = gym.spaces.Discrete(3)  # 0: Idle, 1: Charge, 2: Discharge

        # --- Initialize all possible state variables ---
        # All variables subclasses need to define their state space.
        self.current_step = 0
        self.soc_mwh = 0.0
        self.total_energy_traded_per_quarter = 0.0
        self.total_charged_in_quarter = 0.0
        self.total_discharged_in_quarter = 0.0

    def _get_observation(self) -> np.ndarray:
        """Abstract method: Subclasses MUST implement this."""
        raise NotImplementedError("This method must be implemented by the subclass.")

    def _get_power_rate_from_action(self, action: int) -> float:
        """Translates a discrete action into a power rate in MW."""
        if action == 0:
            return 0.0
        elif action == 1:
            return self.charge_discharge_rate
        elif action == 2:
            return -self.charge_discharge_rate
        else:
            raise ValueError(f"Invalid action {action}")

    def _calculate_delayed_reward(self) -> float:
        """Calculates reward at the end of each 15-minute interval."""
        if self.all_data['Datetime'].iloc[self.current_step].minute % 15 == 14:
            revenue = -self.prices[self.current_step] * self.total_energy_traded_per_quarter

            # Degradation Cost
            throughput = self.total_charged_in_quarter + self.total_discharged_in_quarter
            degradation_cost = throughput * self.marginal_cost_per_mwh

            return revenue - degradation_cost

        return 0.0

    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state."""
        super().reset(seed=seed)

        # If the counter is at the end of the available days, wrap around to 0.
        if self.start_day_counter >= len(self.daily_start_indices):
            self.start_day_counter = 0

        # Start at the beginning of a day
        self.current_step = self.daily_start_indices[self.start_day_counter]
        # Determine the end step for this multi-day episode.
        end_day_index = self.start_day_counter + self.days_per_episode

        # Calculate the end step based on the start of the next day or the end of the data
        if end_day_index >= len(self.daily_start_indices):
            self.current_episode_end_step = self.max_steps - 1
        else:
            self.current_episode_end_step = self.daily_start_indices[end_day_index] - 1

        # Advance the counter for the NEXT time reset() is called
        self.start_day_counter += self.days_per_episode

        self.soc_mwh = 0.0
        self.total_energy_traded_per_quarter = 0.0
        self.total_charged_in_quarter = 0.0
        self.total_discharged_in_quarter = 0.0

        return self._get_observation(), {}

    def step(self, action: int):
        """Executes one time step within the environment. This logic is shared."""
        # Reset tracking variables at the beginning of a new quarter
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

        # Check if the episode is done
        episode_done = self.current_step >= self.current_episode_end_step
        # Check if we are at the end of the dataframe
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
        """Returns a binary mask of valid actions."""
        mask = [1, 1, 1]
        epsilon = 1e-6
        if self.soc_mwh >= self.battery_capacity_mwh - epsilon: mask[1] = 0
        if self.soc_mwh <= epsilon: mask[2] = 0
        return np.array(mask, dtype=np.int8)

    def get_idle_action(self) -> int:
        """Returns the action index for 'Idle'."""
        return 0

class ExtendedBatteryEnv(BaseBatteryEnv):
    """
    An extended battery environment.
    Observation space: [SoC, Current Price, Total Charged in Quarter, Total Discharged in Quarter]
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # --- EXTENDED OBSERVATION SPACE DEFINITION ---
        low_bounds = np.array([0.0, -np.inf, 0.0, 0.0], dtype=np.float32)
        high_bounds = np.array([self.battery_capacity_mwh, np.inf, np.inf, np.inf], dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=low_bounds, high=high_bounds, shape=(4,), dtype=np.float32
        )

    def _get_observation(self) -> np.ndarray:
        """Constructs the extended observation array."""
        return np.array([
            self.soc_mwh,
            self.prices[self.current_step],
            self.total_charged_in_quarter,
            self.total_discharged_in_quarter
        ], dtype=np.float32)

class ContinuousActionWrapper(gym.ActionWrapper):
    """
    Wraps a Discrete(3) environment so that SAC / DDPG can interact with it.
    Maps a continuous scalar in [-1, 1] → {0, 1, 2}.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def action(self, continuous_action):
        if continuous_action < -0.33:
            return 2   # Discharge
        elif continuous_action > 0.33:
            return 1   # Charge
        else:
            return 0   # Idle


# ═══════════════════════════════════════════════════════════════════════
#  EVALUATION RESULT
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class EvaluationResult:
    """
    Structured output for simulation results.
    Provides type hinting and autocompletion.
    """
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
        """Helper to convert the core history to a DataFrame."""
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
    """
    Runs an evaluation using a fully wrapped environment and a trained SB3 model.
    """
    # Use the .unwrapped attribute to get the original BaseBatteryEnv for logging
    unwrapped_env: BaseBatteryEnv = scaled_env.unwrapped

    # Initialize lists to store the history of UN-SCALED, human-readable data
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
        print(f"Starting episode {episode_num + 1}/{number_of_episodes}")

        # --- Interact with the SCALED environment ---
        obs, info = scaled_env.reset()

        # --- Use the unwrapped environment for logging ---
        start_time = unwrapped_env.all_data.iloc[unwrapped_env.current_step]['Datetime']
        end_time = unwrapped_env.all_data.iloc[unwrapped_env.current_episode_end_step]['Datetime']
        print(f"From {start_time} to {end_time}")

        done = False
        reward_per_episode = 0
        while not done:
            # --- Get the action directly from the MODEL ---
            # We get the action mask from the unwrapped env because wrappers hide custom methods
            action_mask = unwrapped_env.action_masks()

            # The model predicts based on the SCALED observation
            if is_masked:
                action, _states = model.predict(
                    obs,
                    deterministic=True,  # Use deterministic mode for evaluation
                    action_masks=action_mask
                )
            else:
                action, _states = model.predict(obs)

            # --- Step the SCALED environment ---
            obs, reward, terminated, truncated, info = scaled_env.step(action)

            # --- Log the UN-SCALED data from the unwrapped environment ---
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
        print(f"Finished with total (scaled) reward: {reward_per_episode:.2f}")

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
#  FOLD LOADING
# ═══════════════════════════════════════════════════════════════════════
def load_folds(
    folds_path: str,
    k: int,
) -> List[Tuple[pd.DataFrame, List[pd.DataFrame], List[pd.DataFrame]]]:
    """
    Load K pre-computed folds from *folds_path*.

    Expected files per fold::

        fold_0_train.pkl   fold_0_val.pkl   fold_0_test.pkl
        fold_1_train.pkl   …
    """
    folds = []
    for i in range(k):
        train_p = os.path.join(folds_path, f"fold_{i}_train.pkl")
        val_p   = os.path.join(folds_path, f"fold_{i}_val.pkl")
        test_p  = os.path.join(folds_path, f"fold_{i}_test.pkl")

        for p in (train_p, val_p, test_p):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing fold file: {p}")

        train_df  = pd.read_pickle(train_p)
        val_eps   = pd.read_pickle(val_p)
        test_eps  = pd.read_pickle(test_p)

        folds.append((train_df, val_eps, test_eps))
        print(f"  Fold {i}: train={len(train_df):,} rows, "
              f"val={len(val_eps)} eps, test={len(test_eps)} eps")

    return folds


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
) -> Dict[str, Any]:
    """
    Train an agent *n_iterations* times from scratch on one fold and
    evaluate on its validation episodes each time.
    """
    pid = os.getpid()
    target_threads = int(os.environ.get('OMP_NUM_THREADS', 4))
    print(f"target_threads: {target_threads}")
    # --- ENFORCE CPU AFFINITY ---
    try:
        start_core = fold_idx * target_threads
        target_cores = set(range(start_core, start_core + target_threads))
        os.sched_setaffinity(0, target_cores)
    except Exception as e:
        print(f"Could not set affinity: {e}")

    torch.set_num_threads(target_threads)

    # --- PURE OBSERVATION CODE ---
    def get_current_core():
        try:
            with open('/proc/self/stat') as f:
                return int(f.read().split()[38])
        except Exception:
            return getattr(os, "sched_getcpu", lambda: None)()

    try:
        allowed_cores = sorted(list(os.sched_getaffinity(0)))
    except AttributeError:
        allowed_cores = "Platform does not support sched_getaffinity"

    print(f"👀 [OBSERVE START] Fold {fold_idx} | PID: {pid} | Current Core: {get_current_core()} | Allowed Pool: {allowed_cores}")

    val_df_combined = pd.concat(val_episodes, ignore_index=True)

    raw_revenues: List[float] = []
    train_times: List[float] = []

    for run in range(1, n_iterations + 1):
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

        core_before_train = get_current_core()

        # ── Train ─────────────────────────────────────────────────────
        model = agent_class(policy_name, train_env, verbose=0)
        t0 = time.time()
        model.learn(total_timesteps=total_steps, reset_num_timesteps=True)
        t_train = time.time() - t0

        core_after_train = get_current_core()

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

        core_after_eval = get_current_core()

        revenue = sum(result.real_rewards)
        raw_revenues.append(revenue)
        train_times.append(t_train)

        print(f"  [Fold {fold_idx}] Run {run}/{n_iterations} — "
              f"Revenue: €{revenue:,.2f}  (train {t_train:.1f}s) | Cores used: start={core_before_train}, post-train={core_after_train}, post-eval={core_after_eval}")

    print(f"🏁 [OBSERVE END] Fold {fold_idx} | PID: {pid} | Current Core: {get_current_core()}")

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
        description="HPC benchmark: train & evaluate one RL agent across K folds."
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
        help="Directory to write result CSVs.",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="If set, runs folds one by one instead of in parallel."
    )
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--days-per-ep", type=int, default=4)
    parser.add_argument("--battery-cap", type=float, default=10.0)
    parser.add_argument("--charge-rate", type=float, default=5.0)
    parser.add_argument("--cycle-cost", type=float, default=6.25)

    args = parser.parse_args()

    agent_class, policy_name, is_continuous = AGENT_REGISTRY[args.agent]
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load folds ────────────────────────────────────────────────────
    print(f"Loading {args.k_folds} folds from '{args.folds_path}' …")
    folds = load_folds(args.folds_path, args.k_folds)

    mode_str = "SEQUENTIALLY" if args.sequential else f"PARALLEL ({args.k_folds} workers)"
    print(f"\nBenchmarking {args.agent} | {args.steps:,} steps | "
          f"{args.iterations} iterations/fold | {args.k_folds} folds | Mode: {mode_str}\n")

    fold_results: List[Dict[str, Any]] = [None] * args.k_folds

    global_start_time = time.time()

    if args.sequential:
        # ── RUN SEQUENTIALLY (Standard For-Loop) ──
        for fold_idx, (train_df, val_eps, test_eps) in enumerate(folds):
            fold_results[fold_idx] = _run_fold(
                fold_idx=fold_idx,
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
    else:
        # ── RUN IN PARALLEL (ProcessPoolExecutor) ──
        with ProcessPoolExecutor(max_workers=args.k_folds) as pool:
            future_to_fold = {}
            for fold_idx, (train_df, val_eps, test_eps) in enumerate(folds):
                future = pool.submit(
                    _run_fold,
                    fold_idx=fold_idx,
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
                future_to_fold[future] = fold_idx

            for future in as_completed(future_to_fold):
                idx = future_to_fold[future]
                try:
                    fold_results[idx] = future.result()
                except Exception as exc:
                    print(f"  *** Fold {idx} raised an exception: {exc}")
                    raise

        # --- END GLOBAL TIMER ---
    total_execution_time = time.time() - global_start_time

    # ── Aggregate & save ──────────────────────────────────────────────
    rows = []
    for r in fold_results:
        rows.append({
            "agent": args.agent,
            "total_steps": args.steps,
            "fold": r["fold"],
            "iterations": r["n_iterations"],
            "mean_revenue": round(r["mean_revenue"], 4),
            "std_revenue": round(r["std_revenue"], 4),
            "min_revenue": round(r["min_revenue"], 4),
            "max_revenue": round(r["max_revenue"], 4),
            "mean_train_time_sec": round(r["mean_train_time_sec"], 2),
        })

    df_results = pd.DataFrame(rows)

    csv_name = f"{args.agent}_{args.steps}_results.csv"
    csv_path = os.path.join(args.output_dir, csv_name)
    df_results.to_csv(csv_path, index=False)

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print(f"  RESULTS — {args.agent} @ {args.steps:,} steps")
    print(f"{'=' * 64}")
    print(df_results.to_string(index=False))

    overall_mean = df_results["mean_revenue"].mean()
    overall_std  = df_results["mean_revenue"].std()
    print(f"\nCross-fold mean revenue : €{overall_mean:,.2f}  ± €{overall_std:,.2f}")
    print(f"Results saved to        : {csv_path}")


if __name__ == "__main__":
    main()

