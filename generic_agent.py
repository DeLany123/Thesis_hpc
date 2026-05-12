"""
Generic Battery Agent — Train & Evaluate a Domain-Randomised SAC Policy
========================================================================

Trains a single SAC policy on a distribution of battery configurations
(capacity and charge rate) using domain randomisation, then evaluates
zero-shot on a set of fixed target configurations.

Usage examples::

    python generic_agent.py --steps 500000
    python generic_agent.py --steps 500000 --folds-path /data/folds --sequential
    python generic_agent.py --steps 500000 --cap-min 5 --cap-max 20

Arguments:
    --steps         : Total SAC training timesteps per run (default: 500000)
    --iterations    : Independent train-from-scratch runs per fold (default: 3)
    --folds-path    : Directory containing fold pickle files
    --output-dir    : Directory to write result CSVs (default: ./results)
    --k-folds       : Number of folds (default: 5)
    --days-per-ep   : Days per episode (default: 4)
    --cap-min       : Minimum battery capacity in MWh for training distribution (default: 5.0)
    --cap-max       : Maximum battery capacity in MWh for training distribution (default: 20.0)
    --rate-min      : Minimum charge/discharge rate in MW (default: 2.5)
    --rate-max      : Maximum charge/discharge rate in MW (default: 10.0)
    --marginal-cost : Degradation cost in EUR/MWh traded (default: 0.3125)
    --sequential    : Run folds sequentially instead of in parallel
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.base_class import BaseAlgorithm

# Allow importing from the repository root (benchmark_agent.py lives there).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from script import (
    BaseBatteryEnv,
    ContinuousActionWrapper,
    EvaluationResult,
    load_folds,
    run_evaluation,
)


# ═══════════════════════════════════════════════════════════════════════
#  EVALUATION CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════

# Fixed battery assets used for zero-shot evaluation after generic training.
EVAL_CONFIGS: List[Dict[str, Any]] = [
    {"name": "A_small",       "cap": 5.0,  "rate": 2.5, "label": "Small (5 MWh / 2.5 MW)"},
    {"name": "B_baseline",    "cap": 10.0, "rate": 5.0, "label": "Baseline (10 MWh / 5 MW)"},
    {"name": "C_large",       "cap": 20.0, "rate": 10.0, "label": "Large (20 MWh / 10 MW)"},
    {"name": "D_high_c_rate", "cap": 10.0, "rate": 7.5, "label": "High C-rate (10 MWh / 7.5 MW)"},
]


# ═══════════════════════════════════════════════════════════════════════
#  GENERIC BATTERY ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════

class GenericBatteryEnv(BaseBatteryEnv):
    """
    A battery environment that randomises capacity and charge rate at every
    episode reset, producing a configuration-invariant normalised observation.

    Observation vector: [SoC_norm, price, charged_norm, discharged_norm]

      SoC_norm      = soc_mwh / battery_capacity_mwh           in [0, 1]
      price         = imbalance price (EUR/MWh)                 unbounded
      charged_norm  = total_charged_in_quarter  / E_max_quarter  in [0, 1]
      discharged_norm = total_discharged_in_quarter / E_max_quarter in [0, 1]

    where E_max_quarter = charge_rate * (15 / 60) MWh is the maximum energy
    that can be traded in a single 15-minute quarter at full power.

    The observation bounds are therefore fixed regardless of which (cap, rate)
    pair is active, enabling the SAC policy to transfer zero-shot to any
    configuration within (or near) the training distribution.

    Reward is normalised by battery capacity during training so that the
    gradient signal remains on a comparable scale across episodes that differ
    substantially in absolute EUR reward.  Raw EUR revenue is recovered by
    multiplying the cumulative normalised reward by the capacity at evaluation
    time.
    """

    def __init__(
        self,
        cap_range: Tuple[float, float],
        rate_range: Tuple[float, float],
        marginal_cost_per_mwh: float,
        all_data: pd.DataFrame,
        days_per_episode: int = 1,
        normalise_reward: bool = True,
    ):
        """
        Parameters
        ----------
        cap_range : (min_capacity_mwh, max_capacity_mwh)
            Uniform sampling bounds for battery capacity.  Pass identical
            values to disable randomisation and fix the capacity.
        rate_range : (min_rate_mw, max_rate_mw)
            Uniform sampling bounds for the charge/discharge rate.
        marginal_cost_per_mwh : float
            Fixed degradation cost in EUR per MWh of energy traded,
            independent of battery size.
        all_data : pd.DataFrame
            Market data with columns 'Datetime' and 'Imbalance Price'.
        days_per_episode : int
            Number of consecutive days in each episode.
        normalise_reward : bool
            If True, divide the EUR reward by the current battery capacity
            before returning it from step().  Disable for final evaluation.
        """
        mid_cap = (cap_range[0] + cap_range[1]) / 2.0
        mid_rate = (rate_range[0] + rate_range[1]) / 2.0

        # Derive cycle_cost_eur from the desired marginal cost and the midpoint
        # capacity so the base class __init__ is satisfied.  The actual value
        # of self.marginal_cost_per_mwh is overwritten immediately after.
        init_cycle_cost = marginal_cost_per_mwh * 2.0 * mid_cap

        super().__init__(
            battery_capacity_mwh=mid_cap,
            charge_discharge_rate_mw=mid_rate,
            all_data=all_data,
            days_per_episode=days_per_episode,
            cycle_cost_eur=init_cycle_cost,
        )

        self.cap_range = cap_range
        self.rate_range = rate_range
        self.fixed_marginal_cost_per_mwh = marginal_cost_per_mwh
        self.normalise_reward = normalise_reward

        # Override marginal cost with the fixed value (not derived from capacity).
        self.marginal_cost_per_mwh = marginal_cost_per_mwh

        # Fixed observation space: bounds do not depend on (cap, rate).
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -np.inf, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0,  np.inf, 1.0, 1.0], dtype=np.float32),
            shape=(4,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        """Sample a new battery configuration then reset the episode."""
        rng = np.random.default_rng(seed)
        self.battery_capacity_mwh = float(rng.uniform(*self.cap_range))
        self.charge_discharge_rate = float(rng.uniform(*self.rate_range))
        # Marginal cost is configuration-independent.
        self.marginal_cost_per_mwh = self.fixed_marginal_cost_per_mwh
        return super().reset(seed=seed, options=options)

    # ------------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        """Return the normalised, configuration-invariant observation vector."""
        energy_per_quarter = self.charge_discharge_rate * (15.0 / 60.0)  # MWh

        soc_norm = self.soc_mwh / self.battery_capacity_mwh

        if energy_per_quarter > 0:
            charged_norm = self.total_charged_in_quarter / energy_per_quarter
            discharged_norm = self.total_discharged_in_quarter / energy_per_quarter
        else:
            charged_norm = 0.0
            discharged_norm = 0.0

        return np.array(
            [
                np.clip(soc_norm, 0.0, 1.0),
                self.prices[self.current_step],
                np.clip(charged_norm, 0.0, 1.0),
                np.clip(discharged_norm, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    def step(self, action):
        """Execute one time step and optionally normalise the reward by capacity."""
        obs, reward, terminated, truncated, info = super().step(action)
        if self.normalise_reward and self.battery_capacity_mwh > 0:
            reward = reward / self.battery_capacity_mwh
            info["real_reward"] = info.get("real_reward", 0.0) / self.battery_capacity_mwh
        return obs, reward, terminated, truncated, info


# ═══════════════════════════════════════════════════════════════════════
#  SINGLE-FOLD WORKER
# ═══════════════════════════════════════════════════════════════════════

def _run_fold(
    fold_idx: int,
    train_df: pd.DataFrame,
    val_episodes: List[pd.DataFrame],
    total_steps: int,
    n_iterations: int,
    days_per_episode: int,
    cap_range: Tuple[float, float],
    rate_range: Tuple[float, float],
    marginal_cost_per_mwh: float,
) -> Dict[str, Any]:
    """
    Train the generic SAC agent n_iterations times from scratch on one fold and
    evaluate on validation episodes for each target configuration.

    Returns a dict with aggregated statistics for this fold.
    """
    target_threads = int(os.environ.get("OMP_NUM_THREADS", 6))
    torch.set_num_threads(target_threads)

    val_df_combined = pd.concat(val_episodes, ignore_index=True)

    # Results keyed by config name, then list of raw EUR revenues.
    config_revenues: Dict[str, List[float]] = {c["name"]: [] for c in EVAL_CONFIGS}
    train_times: List[float] = []

    for run in range(1, n_iterations + 1):
        # ── Build generic training environment ───────────────────────────
        train_env = GenericBatteryEnv(
            cap_range=cap_range,
            rate_range=rate_range,
            marginal_cost_per_mwh=marginal_cost_per_mwh,
            all_data=train_df,
            days_per_episode=days_per_episode,
            normalise_reward=True,
        )
        train_env = ContinuousActionWrapper(train_env)

        # ── Train SAC ─────────────────────────────────────────────────────
        model = SAC("MlpPolicy", train_env, verbose=0)
        t0 = time.time()
        model.learn(total_timesteps=total_steps, reset_num_timesteps=True)
        t_train = time.time() - t0
        train_times.append(t_train)

        print(
            f"  [Fold {fold_idx}] Run {run}/{n_iterations} — "
            f"training finished in {t_train:.1f}s"
        )

        # ── Evaluate on each fixed configuration ─────────────────────────
        for cfg in EVAL_CONFIGS:
            val_env = GenericBatteryEnv(
                cap_range=(cfg["cap"], cfg["cap"]),
                rate_range=(cfg["rate"], cfg["rate"]),
                marginal_cost_per_mwh=marginal_cost_per_mwh,
                all_data=val_df_combined,
                days_per_episode=days_per_episode,
                normalise_reward=False,  # report raw EUR at evaluation
            )
            val_env = ContinuousActionWrapper(val_env)

            result: EvaluationResult = run_evaluation(
                val_env,
                model,
                number_of_episodes=len(val_episodes),
                is_masked=False,
            )
            revenue = sum(result.real_rewards)
            config_revenues[cfg["name"]].append(revenue)

            print(
                f"    Config {cfg['name']}: €{revenue:,.2f}"
            )

    # ── Aggregate statistics per configuration ────────────────────────────
    aggregated: Dict[str, Any] = {
        "fold": fold_idx,
        "n_iterations": n_iterations,
        "mean_train_time_sec": float(np.mean(train_times)),
    }
    for cfg in EVAL_CONFIGS:
        revs = config_revenues[cfg["name"]]
        prefix = cfg["name"]
        aggregated[f"{prefix}_mean_revenue"] = float(np.mean(revs))
        aggregated[f"{prefix}_std_revenue"] = float(np.std(revs))
        aggregated[f"{prefix}_all_revenues"] = revs

    return aggregated


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train a domain-randomised generic SAC battery agent and "
                    "evaluate zero-shot on fixed target configurations."
    )
    parser.add_argument("--steps",         type=int,   default=500_000)
    parser.add_argument("--iterations",    type=int,   default=3)
    parser.add_argument(
        "--folds-path", type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
    )
    parser.add_argument("--k-folds",       type=int,   default=5)
    parser.add_argument("--days-per-ep",   type=int,   default=4)
    parser.add_argument("--cap-min",       type=float, default=5.0)
    parser.add_argument("--cap-max",       type=float, default=20.0)
    parser.add_argument("--rate-min",      type=float, default=2.5)
    parser.add_argument("--rate-max",      type=float, default=10.0)
    parser.add_argument("--marginal-cost", type=float, default=0.3125,
                        help="Degradation cost in EUR/MWh (default matches 6.25 EUR/cycle on 10 MWh).")
    parser.add_argument("--sequential",    action="store_true")
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Run exactly this one fold index (0-based). "
             "Omit to run all folds (sequential or parallel).",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cap_range = (args.cap_min, args.cap_max)
    rate_range = (args.rate_min, args.rate_max)

    print(f"Loading {args.k_folds} folds from '{args.folds_path}' …")
    folds = load_folds(args.folds_path, args.k_folds)

    worker_kwargs = dict(
        total_steps=args.steps,
        n_iterations=args.iterations,
        days_per_episode=args.days_per_ep,
        cap_range=cap_range,
        rate_range=rate_range,
        marginal_cost_per_mwh=args.marginal_cost,
    )

    # ── Single-fold mode (used by HPC per-fold jobs) ──────────────────
    if args.fold is not None:
        if args.fold < 0 or args.fold >= args.k_folds:
            raise ValueError(
                f"--fold {args.fold} is out of range for {args.k_folds} folds."
            )
        train_df, val_eps, _test_eps = folds[args.fold]
        print(
            f"\nGeneric SAC | fold {args.fold} only | {args.steps:,} steps | "
            f"{args.iterations} iters | cap∈{cap_range} MWh | rate∈{rate_range} MW\n"
        )
        global_start = time.time()
        result = _run_fold(
            fold_idx=args.fold,
            train_df=train_df,
            val_episodes=val_eps,
            **worker_kwargs,
        )
        total_elapsed = time.time() - global_start
        print(f"\nFold {args.fold} finished in {total_elapsed:.1f}s")

        row = {"fold": result["fold"], "mean_train_time_sec": result["mean_train_time_sec"]}
        for cfg in EVAL_CONFIGS:
            prefix = cfg["name"]
            row[f"{prefix}_mean_revenue"] = result[f"{prefix}_mean_revenue"]
            row[f"{prefix}_std_revenue"]  = result[f"{prefix}_std_revenue"]

        summary_df = pd.DataFrame([row])
        output_path = os.path.join(
            args.output_dir, f"generic_model_fold{args.fold}_{args.steps}_results.csv"
        )
        summary_df.to_csv(output_path, index=False)
        print(f"\nResults written to: {output_path}")
        print(summary_df.to_string(index=False))
        return

    # ── All-folds mode ────────────────────────────────────────────────
    mode_str = "SEQUENTIALLY" if args.sequential else f"PARALLEL ({args.k_folds} workers)"
    print(
        f"\nGeneric SAC | {args.steps:,} steps | {args.iterations} iters/fold | "
        f"{args.k_folds} folds | cap∈{cap_range} MWh | rate∈{rate_range} MW | "
        f"Mode: {mode_str}\n"
    )

    fold_results: List[Dict[str, Any]] = [None] * args.k_folds
    global_start = time.time()

    if args.sequential:
        for fold_idx, (train_df, val_eps, _test_eps) in enumerate(folds):
            fold_results[fold_idx] = _run_fold(
                fold_idx=fold_idx,
                train_df=train_df,
                val_episodes=val_eps,
                **worker_kwargs,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.k_folds) as pool:
            futures = {}
            for fold_idx, (train_df, val_eps, _test_eps) in enumerate(folds):
                fut = pool.submit(
                    _run_fold,
                    fold_idx=fold_idx,
                    train_df=train_df,
                    val_episodes=val_eps,
                    **worker_kwargs,
                )
                futures[fut] = fold_idx

            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    fold_results[idx] = fut.result()
                    print(f"Fold {idx} complete.")
                except Exception as exc:
                    print(f"Fold {idx} raised: {exc}")
                    raise

    total_elapsed = time.time() - global_start
    print(f"\nAll folds finished in {total_elapsed:.1f}s")

    # ── Build summary CSV ─────────────────────────────────────────────
    rows = []
    for res in fold_results:
        if res is None:
            continue
        row = {
            "fold": res["fold"],
            "mean_train_time_sec": res["mean_train_time_sec"],
        }
        for cfg in EVAL_CONFIGS:
            prefix = cfg["name"]
            row[f"{prefix}_mean_revenue"] = res[f"{prefix}_mean_revenue"]
            row[f"{prefix}_std_revenue"]  = res[f"{prefix}_std_revenue"]
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    # Append overall mean row.
    numeric_cols = [c for c in summary_df.columns if c != "fold"]
    mean_row = summary_df[numeric_cols].mean().to_dict()
    mean_row["fold"] = "mean"
    summary_df = pd.concat(
        [summary_df, pd.DataFrame([mean_row])], ignore_index=True
    )

    output_path = os.path.join(args.output_dir, "generic_model_results.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"\nResults written to: {output_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
