import os
import tempfile

import numpy as np

from src.utils.config import AttrDict
from scripts.train import train


def test_trainer_smoke_test():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a dummy dataset
        dataset_path = os.path.join(tmpdir, "dataset.npz")
        np.savez(
            dataset_path,
            states=np.random.randn(10, 20, 4),
                actions=np.random.randint(0, 2, (10, 20, 1)),
            returns_to_go=np.random.randn(10, 20, 1),
            timesteps=np.random.randint(0, 100, (10, 20)),
            mask=np.ones((10, 20)),
                    metadata={"state_dim": 4, "act_dim": 2, "max_timesteps": 100},
        )

        # Create a dummy config
        save_dir = os.path.join(tmpdir, "results")
        config = {
            "seed": 42,
            "save_dir": save_dir,
                "env": "CartPole-v1",
            "training": {
                "optimizer": "AdamW",
                "lr": 1e-4,
                "weight_decay": 1e-4,
                "batch_size": 2,
                "epochs": 1,
                "device": "cpu",
                "eval_every": 1,
                "checkpoint_every": 1,
                "log_wandb": False,
                "persistent_workers": False,
                "num_workers": 0,
                "pin_memory": False,
                "batches_per_epoch": 1,
            },
            "model": {
                "name": "dt",
                "d_model": 64,
                "n_layers": 1,
                "n_heads": 2,
                "seq_len": 20,
                "action_tanh": False,
            },
            "dataset": {
                "path": dataset_path,
                "max_timesteps": 100,
                "state_dim": 4,
                    "act_dim": 2,
                    "is_discrete": True,
            },
        }
        cfg = AttrDict(config)

        # Run the training function
        import logging
        train(cfg, logging.getLogger())

        # Check for output files
        assert os.path.exists(os.path.join(save_dir, "metrics.csv"))
        assert os.path.exists(os.path.join(save_dir, "ckpt_epoch_1.pt"))