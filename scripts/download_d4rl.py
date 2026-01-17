import os
import argparse
import urllib.request
from tqdm import tqdm

# D4RL Dataset URLs (v2 for Mujoco)
# Source: https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/gym_mujoco/env_dict.py or similar
DATASET_URLS = {
    "hopper-medium-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium-v2.hdf5",
    "hopper-medium-expert-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium_expert-v2.hdf5",
    "walker2d-medium-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium-v2.hdf5",
    "walker2d-medium-expert-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium_expert-v2.hdf5",
    "halfcheetah-medium-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium-v2.hdf5",
    "halfcheetah-medium-expert-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium_expert-v2.hdf5",
}

def download_file(url, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    if os.path.exists(out_path):
        print(f"File already exists: {out_path}")
        return

    print(f"Downloading {url} to {out_path}...")
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=out_path, reporthook=t.update_to)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="data/d4rl_raw", help="Directory to save HDF5 files.")
    args = parser.parse_args()

    for name, url in DATASET_URLS.items():
        # Filename construction: match what convert_d4rl.py expects
        # convert logic: filename.replace('.hdf5', '') -> env_name
        # If we save as 'hopper_medium-v2.hdf5', convert expects 'hopper_medium-v2'
        # convert logic: name_stem.replace('_', '-')
        # So 'hopper_medium-v2' -> 'hopper-medium-v2'. This works.
        
        filename = url.split('/')[-1]
        out_path = os.path.join(args.out_dir, filename)
        download_file(url, out_path)

if __name__ == "__main__":
    main()
