#!/usr/bin/env python3
"""
baselines/compare.py
Head-to-head: DT vs DSFormer vs SDT on classic control.
"""
import os, sys
DSF_GYM = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "external", "DecisionSpikeFormer", "gym"))
sys.path.insert(0, DSF_GYM)

from decision_spikeformer_pssa import DecisionSpikeFormer

from baselines.dsf_runner import evaluate as eval_dsf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--dsf_ckpt", default="external/DecisionSpikeFormer/logs/hopper-medium.pt")
    parser.add_argument("--sdt_ckpt", default="logs/sdt_cartpole.pt")
    args = parser.parse_args()

    results = {}

    # 1. DSFormer
    with TegraMeter("/tmp/dsf_power.log"):
        t0 = time.time()
        dsf_ret, dsf_len = eval_dsf(args.dsf_ckpt, args.env, args.episodes)
        dsf_latency = (time.time() - t0) / (args.episodes * dsf_len)
    dsf_power = parse_tegra_log("/tmp/dsf_power.log")
    dsf_energy = dsf_power * dsf_latency * 1e-3
    results["DSFormer"] = {"return": dsf_ret, "energy_mJ": dsf_energy}

    # 2. SDT (placeholder – replace with your own evaluate call)
    # with TegraMeter("/tmp/sdt_power.log"):
    #     ...
    results["SDT"] = {"return": 500.0, "energy_mJ": 0.051}  # dummy

    with open(f"results_{args.env}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
