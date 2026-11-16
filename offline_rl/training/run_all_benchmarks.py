"""
Orchestrator to run offline RL training and evaluation across all benchmark tiers:
- Tier 1 (First): Hopper (medium, medium-replay, medium-expert)
- Tier 2 (Second): HalfCheetah (medium, medium-expert)
- Tier 3 (Third): Walker2d (medium, medium-expert)
- Tier 4 (Optional Extension): Ant (medium)

Computes normalized D4RL scores and generates a consolidated benchmark report.
"""

import argparse
import os
import sys
import time
from typing import Dict, List

from offline_rl.training.train_offline import train_offline

TIERS = {
    "first": [
        ("Hopper", "medium", "offline_rl/configs/hopper_medium.yaml"),
        ("Hopper", "medium-replay", "offline_rl/configs/hopper_medium_replay.yaml"),
        ("Hopper", "medium-expert", "offline_rl/configs/hopper_medium_expert.yaml"),
    ],
    "second": [
        ("HalfCheetah", "medium", "offline_rl/configs/halfcheetah_medium.yaml"),
        ("HalfCheetah", "medium-expert", "offline_rl/configs/halfcheetah_medium_expert.yaml"),
    ],
    "third": [
        ("Walker2d", "medium", "offline_rl/configs/walker2d_medium.yaml"),
        ("Walker2d", "medium-expert", "offline_rl/configs/walker2d_medium_expert.yaml"),
    ],
    "optional": [
        ("Ant", "medium", "offline_rl/configs/ant_medium.yaml"),
    ],
    "phase8": [
        ("Hopper", "medium", "offline_rl/configs/hopper_medium.yaml"),
        ("Hopper", "medium-replay", "offline_rl/configs/hopper_medium_replay.yaml"),
        ("Hopper", "medium-expert", "offline_rl/configs/hopper_medium_expert.yaml"),
        ("HalfCheetah", "medium", "offline_rl/configs/halfcheetah_medium.yaml"),
        ("Walker2d", "medium", "offline_rl/configs/walker2d_medium.yaml"),
        ("Ant", "medium", "offline_rl/configs/ant_medium.yaml"),
    ],
}


def run_benchmarks(
    tier_choice: str = "first",
    max_iters: int = 5,
    steps_per_iter: int = 200,
    do_eval: bool = True,
    output_dir: str = "checkpoints/offline_rl",
):
    selected_benchmarks = []
    if tier_choice == "all":
        for t in ["first", "second", "third", "optional"]:
            selected_benchmarks.extend(TIERS[t])
    elif tier_choice in TIERS:
        selected_benchmarks = TIERS[tier_choice]
    else:
        raise ValueError(f"Unknown tier: {tier_choice}. Choose from 'first', 'second', 'third', 'optional', or 'all'.")

    results_table = []
    total_start = time.time()

    print("=" * 80)
    print(f"Starting SNN-DT Benchmark Suite | Selected Tier: {tier_choice.upper()}")
    print(f"Total Environments to train: {len(selected_benchmarks)}")
    print(f"Iterations per env: {max_iters} | Steps per iter: {steps_per_iter}")
    print("=" * 80)

    for env_family, dataset_type, config_path in selected_benchmarks:
        print("\n" + "#" * 70)
        print(f"Running Environment: {env_family} [{dataset_type}] | Config: {config_path}")
        print("#" * 70)

        t0 = time.time()
        try:
            model, dataset, eval_res = train_offline(
                config_path=config_path,
                output_dir=output_dir,
                override_iters=max_iters,
                override_steps=steps_per_iter,
                do_eval=do_eval,
            )
            elapsed = time.time() - t0

            mean_return = eval_res.get("eval/mean_return", 0.0) if eval_res else 0.0
            norm_score = eval_res.get("eval/normalized_score", 0.0) if eval_res else 0.0
            spike_rate = eval_res.get("eval/mean_spike_rate", 0.0) if eval_res else 0.0

            results_table.append(
                {
                    "env": f"{env_family} ({dataset_type})",
                    "status": "SUCCESS",
                    "time_s": f"{elapsed:.1f}",
                    "eval_return": f"{mean_return:.1f}",
                    "norm_score": f"{norm_score:.1f}",
                    "spike_rate": f"{spike_rate:.4f}",
                }
            )
        except Exception as e:
            print(f"[ERROR] Failed to run benchmark {env_family}-{dataset_type}: {e}")
            results_table.append(
                {
                    "env": f"{env_family} ({dataset_type})",
                    "status": f"FAILED: {str(e)[:25]}",
                    "time_s": "N/A",
                    "eval_return": "N/A",
                    "norm_score": "N/A",
                    "spike_rate": "N/A",
                }
            )

    total_time = time.time() - total_start

    print("\n" + "=" * 80)
    print("BENCHMARK EXECUTION SUMMARY")
    print("=" * 80)
    header = f"{'Environment':<30} | {'Status':<10} | {'Time (s)':<8} | {'Return':<8} | {'D4RL Score':<10} | {'Spike Rate':<10}"
    print(header)
    print("-" * len(header))
    for r in results_table:
        print(
            f"{r['env']:<30} | {r['status']:<10} | {r['time_s']:<8} | {r['eval_return']:<8} | {r['norm_score']:<10} | {r['spike_rate']:<10}"
        )
    print("=" * 80)
    print(f"Total Elapsed Time: {total_time:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SNN-DT MuJoCo Benchmark Suite")
    parser.add_argument(
        "--tier",
        type=str,
        default="first",
        choices=["first", "second", "third", "optional", "phase8", "all"],
        help="Benchmark tier to execute: 'first' (Hopper), 'second' (HalfCheetah), 'third' (Walker2d), 'optional' (Ant), 'phase8' (Hopper->Replay->Expert->Cheetah->Walker->Ant), or 'all'",
    )
    parser.add_argument("--max_iters", type=int, default=5, help="Number of training iterations per environment")
    parser.add_argument("--steps_per_iter", type=int, default=100, help="Steps per iteration")
    parser.add_argument("--no_eval", action="store_true", help="Skip online MuJoCo evaluation")
    parser.add_argument("--output_dir", type=str, default="checkpoints/offline_rl")
    args = parser.parse_args()

    run_benchmarks(
        tier_choice=args.tier,
        max_iters=args.max_iters,
        steps_per_iter=args.steps_per_iter,
        do_eval=not args.no_eval,
        output_dir=args.output_dir,
    )
