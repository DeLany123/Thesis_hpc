"""
Generic Battery Agent v2 — Domain-Randomised SAC with Config-Conditioned Observations
======================================================================================

Builds on generic_agent_scaled.py with two improvements:

  1. Config features added to the observation vector.
     The policy now receives normalised battery capacity and C-rate as explicit
     features, allowing it to condition its trading strategy on the hardware it
     is operating.  This addresses the core limitation of the v1 generic model,
     which was blind to the current configuration and had to learn a single
     averaged strategy across all configs.

  2. Reward double-normalisation removed.
     v1 divided the reward by battery capacity before applying the Welford
     running scaler, effectively applying two normalisations.  v2 relies solely
     on the Welford wrapper, which already adapts to the reward scale of whatever
     configuration is active.

Observation vector (6 features):

    [SoC_norm, price_scaled, A^c_norm, A^d_norm, cap_norm, c_rate_norm]

      SoC_norm     = SoC_t / C                          in [0, 1]
      price_scaled = (p_t - median) / IQR               unbounded
      A^c_norm     = A^c_t / E_max_quarter              in [0, 1]
      A^d_norm     = A^d_t / E_max_quarter              in [0, 1]
      cap_norm     = C / cap_max                         in [0, 1]
      c_rate_norm  = (r / C) / c_rate_max               in [0, 1]

    cap_max and c_rate_max are the global maxima of the training distribution,
    kept fixed at evaluation time so the feature scale is consistent.

Usage examples::

    python generic_agent_v2.py --steps 500000
    python generic_agent_v2.py --steps 500000 --folds-path /data/folds --sequential
    python generic_agent_v2.py --steps 500000 --cap-min 5 --cap-max 20
    python generic_agent_v2.py --steps 500000 --price-scale-method standard

Arguments:
    --steps              : Total SAC training timesteps per run (default: 500000)
    --iterations         : Independent train-from-scratch runs per fold (default: 3)
    --folds-path         : Directory containing fold pickle files
    --output-dir         : Directory to write result CSVs (default: ./results)
    --k-folds            : Number of folds (default: 5)
    --days-per-ep        : Days per episode (default: 4)
    --cap-min            : Minimum battery capacity in MWh for training distribution (default: 5.0)
    --cap-max            : Maximum battery capacity in MWh for training distribution (default: 20.0)
    --rate-min           : Minimum charge/discharge rate in MW (default: 2.5)
    --rate-max           : Maximum charge/discharge rate in MW (default: 10.0)
    --marginal-cost      : Degradation cost in EUR/MWh traded (default: 0.3125)
    --price-scale-method : Price scaling method: standard | robust | minmax_clipped (default: robust)
    --sequential         : Run folds sequentially instead of in parallel
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any, Literal

import pandas as pd
import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import SAC

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
    {"name": "A_small",       "cap": 5.0,  "rate": 2.5,  "label": "Small (5 MWh / 2.5 MW)"},
    {"name": "B_baseline",    "cap": 10.0, "rate": 5.0,  "label": "Baseline (10 MWh / 5 MW)"},
    {"name": "C_large",       "cap": 20.0, "rate": 10.0, "label": "Large (20 MWh / 10 MW)"},
    {"name": "D_high_c_rate", "cap": 10.0, "rate": 7.5,  "label": "High C-rate (10 MWh / 7.5 MW)"},
]


# ═══════════════════════════════════════════════════════════════════════
#  PRICE SCALING HELPERS
# ═══════════════════════════════════════════════════════════════════════

def fit_price_scaler(
    prices: np.ndarray,
    method: Literal["standard", "robust", "minmax_clipped"] = "robust",
) -> Tuple[float, float]:
    """
    Fit price scaling parameters exclusively on a price series.

    Returns
    -------
    (param1, param2) where the scaled price is (p - param1) / param2.

    Methods
    -------
    robust         : param1 = median, param2 = IQR  (default; resistant to spikes)
    standard       : param1 = mean,   param2 = std
    minmax_clipped : param1 = 1st-percentile value, param2 = (99th - 1st) range
    """
    if method == "robust":
        param1 = float(np.median(prices))
        param2 = float(np.quantile(prices, 0.75) - np.quantile(prices, 0.25))
    elif method == "standard":
        param1 = float(prices.mean())
        param2 = float(prices.std())
    elif method == "minmax_clipped":
        param1 = float(np.quantile(prices, 0.01))
        param2 = float(np.quantile(prices, 0.99) - param1)
    else:
        raise ValueError(f"Unknown price scaling method: {method!r}")

    # Guard against zero-range distributions.
    if param2 == 0.0:
        param2 = 1e-8

    return param1, param2


# ═══════════════════════════════════════════════════════════════════════
#  WELFORD RUNNING REWARD SCALER
# ═══════════════════════════════════════════════════════════════════════

class RunningRewardScaler(gym.RewardWrapper):
    """
    Normalises the reward signal online using a running estimate of its
    standard deviation (Welford's algorithm).

    The mean is *not* subtracted so the sign of the reward is preserved:
    positive rewards represent profit, negative rewards represent losses.
    Normalised rewards are clipped to [-clip, +clip] to prevent extreme
    outliers from destabilising the gradient update.

    The raw EUR financial reward is accessible via info['real_reward'] and
    is unaffected by this wrapper for evaluation purposes.
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8, clip: float = 10.0):
        super().__init__(env)
        self._count  = 0
        self._mean   = 0.0
        self._M2     = 0.0
        self.epsilon = epsilon
        self.clip    = clip

    def reward(self, reward: float) -> float:
        self._count += 1
        delta        = reward - self._mean
        self._mean  += delta / self._count
        self._M2    += delta * (reward - self._mean)

        if self._count < 2:
            return float(np.clip(reward, -self.clip, self.clip))

        variance = self._M2 / (self._count - 1)
        std      = np.sqrt(variance + self.epsilon)
        return float(np.clip(reward / std, -self.clip, self.clip))

    def action_masks(self):
        return self.env.action_masks()


# ═══════════════════════════════════════════════════════════════════════
#  GENERIC BATTERY ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════

class GenericBatteryEnv(BaseBatteryEnv):
    """
    A battery environment that randomises capacity and charge rate at every
    episode reset, producing a configuration-conditioned normalised observation.

    Observation vector (notation follows the mathematical model in Chapter 3):

        [SoC_norm, price_scaled, A^c_norm, A^d_norm, cap_norm, c_rate_norm]

      SoC_norm     = SoC_t / C                          in [0, 1]
      price_scaled = (p_t - price_median) / price_iqr   unbounded
      A^c_norm     = A^c_t / E_max_quarter               in [0, 1]
      A^d_norm     = A^d_t / E_max_quarter               in [0, 1]
      cap_norm     = C / cap_max                          in [0, 1]
      c_rate_norm  = (r / C) / c_rate_max                in [0, 1]

    where E_max_quarter = r * (15 / 60) MWh, price_median / price_iqr are
    fitted on the training fold, and cap_max / c_rate_max are the global maxima
    of the training distribution (fixed at both train and eval time).

    The two config features (cap_norm, c_rate_norm) let the policy condition
    its trading strategy on the hardware it is currently operating, closing
    the information gap that limited the v1 generic model.

    Reward handling
    ---------------
    The raw EUR reward is returned unchanged from step().  A RunningRewardScaler
    wrapper is applied externally during training to normalise by the running
    reward standard deviation (Welford's algorithm).  No capacity division is
    applied, removing the double-normalisation present in generic_agent_scaled.py.
    """

    def __init__(
        self,
        cap_range: Tuple[float, float],
        rate_range: Tuple[float, float],
        marginal_cost_per_mwh: float,
        all_data: pd.DataFrame,
        price_median: float,
        price_iqr: float,
        cap_max: float,
        c_rate_max: float,
        days_per_episode: int = 1,
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
            Fixed degradation cost in EUR per MWh of energy traded.
        all_data : pd.DataFrame
            Market data with columns 'Datetime' and 'Imbalance Price'.
        price_median : float
            Median of the training-fold imbalance price series.
        price_iqr : float
            IQR of the training-fold imbalance price series.
        cap_max : float
            Global maximum battery capacity (MWh) of the training distribution.
            Used to normalise the cap_norm feature consistently at eval time.
        c_rate_max : float
            Global maximum C-rate of the training distribution (rate_max / cap_min).
            Used to normalise the c_rate_norm feature consistently at eval time.
        days_per_episode : int
            Number of consecutive days in each episode.
        """
        mid_cap  = (cap_range[0]  + cap_range[1])  / 2.0
        mid_rate = (rate_range[0] + rate_range[1]) / 2.0

        init_cycle_cost = marginal_cost_per_mwh * 2.0 * mid_cap

        super().__init__(
            battery_capacity_mwh=mid_cap,
            charge_discharge_rate_mw=mid_rate,
            all_data=all_data,
            days_per_episode=days_per_episode,
            cycle_cost_eur=init_cycle_cost,
        )

        self.cap_range  = cap_range
        self.rate_range = rate_range
        self.fixed_marginal_cost_per_mwh = marginal_cost_per_mwh

        self.price_median = price_median
        self.price_iqr    = price_iqr
        self.cap_max      = cap_max
        self.c_rate_max   = c_rate_max

        self.marginal_cost_per_mwh = marginal_cost_per_mwh

        # Observation space: 6 features.
        # SoC, A^c, A^d, cap_norm in [0,1]; scaled price unbounded; c_rate_norm in [0,1].
        self.observation_space = gym.spaces.Box(
            low =np.array([0.0, -np.inf, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0,  np.inf, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(6,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        """Sample a new battery configuration then reset the episode."""
        rng = np.random.default_rng(seed)
        self.battery_capacity_mwh  = float(rng.uniform(*self.cap_range))
        self.charge_discharge_rate = float(rng.uniform(*self.rate_range))
        self.marginal_cost_per_mwh = self.fixed_marginal_cost_per_mwh
        return super().reset(seed=seed, options=options)

    # ------------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        """
        Return the config-conditioned normalised observation vector.

        Notation follows the mathematical model:
          index 0 — SoC_norm:     SoC_t / C                   in [0, 1]
          index 1 — price_scaled: (p_t - median) / IQR        unbounded
          index 2 — A^c_norm:     A^c_t / E_max_quarter        in [0, 1]
          index 3 — A^d_norm:     A^d_t / E_max_quarter        in [0, 1]
          index 4 — cap_norm:     C / cap_max                  in [0, 1]
          index 5 — c_rate_norm:  (r / C) / c_rate_max         in [0, 1]
        """
        e_max_quarter = self.charge_discharge_rate * (15.0 / 60.0)  # MWh

        soc_norm = self.soc_mwh / self.battery_capacity_mwh

        if e_max_quarter > 0:
            a_c_norm = self.total_charged_in_quarter    / e_max_quarter
            a_d_norm = self.total_discharged_in_quarter / e_max_quarter
        else:
            a_c_norm = 0.0
            a_d_norm = 0.0

        price_scaled = (self.prices[self.current_step] - self.price_median) / self.price_iqr

        # Configuration features: normalised by training-distribution maxima.
        cap_norm    = self.battery_capacity_mwh / self.cap_max
        c_rate      = self.charge_discharge_rate / self.battery_capacity_mwh
        c_rate_norm = c_rate / self.c_rate_max

        return np.array(
            [
                np.clip(soc_norm,    0.0, 1.0),
                price_scaled,
                np.clip(a_c_norm,   0.0, 1.0),
                np.clip(a_d_norm,   0.0, 1.0),
                np.clip(cap_norm,   0.0, 1.0),
                np.clip(c_rate_norm, 0.0, 1.0),
            ],
            dtype=np.float32,
        )


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
    price_scale_method: str = "robust",
) -> Dict[str, Any]:
    """
    Train the generic SAC agent n_iterations times from scratch on one fold and
    evaluate zero-shot on validation episodes for each target configuration.

    Price scaling parameters are fitted exclusively on the training fold price
    series to prevent any look-ahead leakage into validation or test data.

    Returns a dict with aggregated statistics for this fold.
    """
    target_threads = int(os.environ.get("OMP_NUM_THREADS", 6))
    torch.set_num_threads(target_threads)

    val_df_combined = pd.concat(val_episodes, ignore_index=True)

    # Fit price scaler on training data only.
    train_prices   = train_df["Imbalance Price"].to_numpy()
    price_median, price_iqr = fit_price_scaler(train_prices, method=price_scale_method)
    print(
        f"  [Fold {fold_idx}] Price scaler ({price_scale_method}): "
        f"median={price_median:.2f}, IQR/scale={price_iqr:.2f}"
    )

    # Global maxima used to normalise the config features consistently.
    cap_max    = cap_range[1]
    c_rate_max = rate_range[1] / cap_range[0]  # max possible C-rate in training dist

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
            price_median=price_median,
            price_iqr=price_iqr,
            cap_max=cap_max,
            c_rate_max=c_rate_max,
            days_per_episode=days_per_episode,
        )
        train_env = ContinuousActionWrapper(train_env)
        # Sole reward normalisation: Welford running std (no capacity division).
        train_env = RunningRewardScaler(train_env)

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
                price_median=price_median,
                price_iqr=price_iqr,
                cap_max=cap_max,
                c_rate_max=c_rate_max,
                days_per_episode=days_per_episode,
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

            print(f"    Config {cfg['name']}: €{revenue:,.2f}")

    # ── Aggregate statistics per configuration ────────────────────────────
    aggregated: Dict[str, Any] = {
        "fold": fold_idx,
        "n_iterations": n_iterations,
        "mean_train_time_sec": float(np.mean(train_times)),
        "price_median": price_median,
        "price_iqr":    price_iqr,
    }
    for cfg in EVAL_CONFIGS:
        revs   = config_revenues[cfg["name"]]
        prefix = cfg["name"]
        aggregated[f"{prefix}_mean_revenue"] = float(np.mean(revs))
        aggregated[f"{prefix}_std_revenue"]  = float(np.std(revs))
        aggregated[f"{prefix}_all_revenues"] = revs

    return aggregated


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train a domain-randomised, price-scaled generic SAC battery agent "
                    "and evaluate zero-shot on fixed target configurations."
    )
    parser.add_argument("--steps",         type=int,   default=10_000)
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
    parser.add_argument(
        "--price-scale-method",
        type=str,
        default="robust",
        choices=["standard", "robust", "minmax_clipped"],
        help="Price scaling method fitted on the training fold (default: robust).",
    )
    parser.add_argument("--sequential",    action="store_true")
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Run exactly this one fold index (0-based). "
             "Omit to run all folds (sequential or parallel).",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cap_range  = (args.cap_min,  args.cap_max)
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
        price_scale_method=args.price_scale_method,
    )

    # ── Single-fold mode (used by HPC per-fold jobs) ──────────────────
    if args.fold is not None:
        if args.fold < 0 or args.fold >= args.k_folds:
            raise ValueError(
                f"--fold {args.fold} is out of range for {args.k_folds} folds."
            )
        train_df, val_eps, _test_eps = folds[args.fold]
        print(
            f"\nGeneric SAC (scaled) | fold {args.fold} only | {args.steps:,} steps | "
            f"{args.iterations} iters | cap∈{cap_range} MWh | rate∈{rate_range} MW | "
            f"price-scale={args.price_scale_method}\n"
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

        row = {
            "fold":               result["fold"],
            "mean_train_time_sec": result["mean_train_time_sec"],
            "price_median":        result["price_median"],
            "price_iqr":           result["price_iqr"],
        }
        for cfg in EVAL_CONFIGS:
            prefix = cfg["name"]
            row[f"{prefix}_mean_revenue"] = result[f"{prefix}_mean_revenue"]
            row[f"{prefix}_std_revenue"]  = result[f"{prefix}_std_revenue"]

        summary_df = pd.DataFrame([row])
        output_path = os.path.join(
            args.output_dir, f"generic_v2_fold{args.fold}_results.csv"
        )
        summary_df.to_csv(output_path, index=False)
        print(f"\nResults written to: {output_path}")
        print(summary_df.to_string(index=False))
        return

    # ── All-folds mode ────────────────────────────────────────────────
    mode_str = "SEQUENTIALLY" if args.sequential else f"PARALLEL ({args.k_folds} workers)"
    print(
        f"\nGeneric SAC (scaled) | {args.steps:,} steps | {args.iterations} iters/fold | "
        f"{args.k_folds} folds | cap∈{cap_range} MWh | rate∈{rate_range} MW | "
        f"price-scale={args.price_scale_method} | Mode: {mode_str}\n"
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
            "fold":                res["fold"],
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
    mean_row     = summary_df[numeric_cols].mean().to_dict()
    mean_row["fold"] = "mean"
    summary_df = pd.concat(
        [summary_df, pd.DataFrame([mean_row])], ignore_index=True
    )

    output_path = os.path.join(
        args.output_dir, f"generic_v2_results_{args.steps}.csv"
    )
    summary_df.to_csv(output_path, index=False)
    print(f"\nResults written to: {output_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

