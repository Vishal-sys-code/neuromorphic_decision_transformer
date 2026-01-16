import os
import argparse
import requests
from tqdm import tqdm

DATASET_URLS = {}

# Replicate D4RL URL construction logic
envs = ['halfcheetah', 'hopper', 'walker2d']
datasets = ['medium', 'medium-expert']

for env in envs:
    for dset in datasets:
        dset_suffix = dset.replace('-', '_')
        # dset_name is the filename stem on the server
        filename = f"{env}_{dset_suffix}-v2.hdf5"
        # env_name is the D4RL environment ID
        env_name = f"{env}-{dset}-v2"
        
        url = f"http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/{filename}"
        DATASET_URLS[env_name] = url

def download_dataset(env_name, download_dir):
    if env_name not in DATASET_URLS:
        print(f"Error: Dataset {env_name} not found in supported list.")
        return False

    url = DATASET_URLS[env_name]
    filename = url.split('/')[-1]
    save_path = os.path.join(download_dir, filename)

    if os.path.exists(save_path):
        print(f"Dataset {env_name} already exists at {save_path}. Skipping.")
        return True

    print(f"Downloading {env_name} from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        block_size = 1024 * 1024 # 1MB
        with open(save_path, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                f.write(data)
        print(f"Downloaded to {save_path}")
        return True
    except Exception as e:
        print(f"Failed to download {env_name}: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, help="Specific environment to download (e.g. hopper-medium-v2). If not provided, downloads all.")
    parser.add_argument("--download-dir", type=str, default="data/d4rl_raw", help="Directory to save HDF5 files.")
    args = parser.parse_args()

    os.makedirs(args.download_dir, exist_ok=True)

    if args.env:
        download_dataset(args.env, args.download_dir)
    else:
        for env_name in DATASET_URLS.keys():
            download_dataset(env_name, args.download_dir)
