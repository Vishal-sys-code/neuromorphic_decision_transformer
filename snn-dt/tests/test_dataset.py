import json
import os
import subprocess
import tempfile

import numpy as np


def test_make_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = "CartPole-v1"
        seed = 42
        num_steps = 1000
        clip_len = 20
        out_file = os.path.join(tmpdir, "dataset.npz")

        # Run the script
        subprocess.run(
            [
                "python",
                "scripts/make_dataset.py",
                "--env",
                env,
                "--seed",
                str(seed),
                "--num_steps",
                str(num_steps),
                "--clip_len",
                str(clip_len),
                "--out",
                out_file,
            ],
            check=True,
        )

        # Check the output file
        assert os.path.exists(out_file)
        with np.load(out_file, allow_pickle=True) as data:

            # Check keys
            expected_keys = ["states", "actions", "returns_to_go", "timesteps", "mask", "metadata"]
            assert all(key in data for key in expected_keys)

            # Check metadata
            metadata = json.loads(data["metadata"].item())
            assert metadata["env"] == env
            assert metadata["seed"] == seed
            assert metadata["num_steps"] == num_steps
            assert metadata["clip_len"] == clip_len

            # Check shapes
            num_clips = data["states"].shape[0]
            assert data["states"].shape == (num_clips, clip_len, 4)  # CartPole state dim is 4
            assert data["actions"].shape == (num_clips, clip_len, 1)
            assert data["returns_to_go"].shape == (num_clips, clip_len, 1)
            assert data["timesteps"].shape == (num_clips, clip_len)
            assert data["mask"].shape == (num_clips, clip_len)

            # Check total steps
            assert np.sum(data["mask"]) >= num_steps