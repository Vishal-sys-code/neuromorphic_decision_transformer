import pytest
import torch
import numpy as np
import argparse
import sys
import os

# Add project root to sys.path to allow importing novel_phases
# Assuming tests are run from the project root or that PYTHONPATH is set appropriately
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# Import functions and classes from the training script
# This might be long, consider importing specific modules or functions as needed

from training_phase1 import (
    get_args,
    set_seed,
    compute_returns_to_go,
    TrajectoryBuffer,
    collect_trajectories,
    TrajectoryDataset,
    DEVICE, # Use the device from the script
    # Constants for default args if needed, though args fixture is better
    DEFAULT_ENV_NAME, DEFAULT_OFFLINE_STEPS, DEFAULT_MAX_EPISODE_LENGTH,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LR, DEFAULT_GAMMA,
    DEFAULT_MAX_LENGTH, DEFAULT_SEED, DEFAULT_CHECKPOINT_DIR
)
from phase1_integration_with_sdt import (
    SpikingDecisionTransformer,
    get_default_config as get_model_default_config,
    compute_spiking_loss # Added this import
)

# Basic test to ensure the file is picked up by pytest
def test_pytest_setup():
    assert True

@pytest.fixture
def default_args():
    """Provides a default set of arguments for testing."""
    model_config = get_model_default_config()
    args = argparse.Namespace(
        env_name="CartPole-v1", # Default test environment
        offline_steps=100,      # Minimal steps for testing
        max_episode_length=50, # Max episode length for collection and model
        epochs=1,               # Minimal epochs
        batch_size=2,           # Small batch size
        lr=DEFAULT_LR,
        gamma=DEFAULT_GAMMA,
        max_length=5,           # Small context window K
        seed=DEFAULT_SEED,
        checkpoint_dir="test_checkpoints", # Use a test-specific dir
        log_interval=1,
        embedding_dim=model_config['embedding_dim'], # Keep model defaults for arch
        num_layers=1,           # Minimal layers for speed
        num_heads=1,            # Minimal heads
        T_max=model_config['T_max'],
        dropout=model_config['dropout'],
        lambda_reg=model_config['lambda_reg']
    )
    return args

@pytest.fixture
def continuous_env_args(default_args):
    """Provides arguments for a continuous environment like Pendulum-v1."""
    args = default_args
    args.env_name = "Pendulum-v1"
    # Pendulum specific details if any, for now, defaults are mostly fine
    # For Pendulum, offline_steps might need to be higher if using random data for training step test
    args.offline_steps = 200 # Pendulum episodes are often 200 steps
    args.max_episode_length = 200
    return args

def test_get_args_defaults(mocker):
    """Test get_args with default (empty) command line and mocked input."""
    # Mock sys.argv to simulate no command-line arguments
    mocker.patch.object(sys, 'argv', ['training_phase1.py'])
    
    # Mock input() to provide a default choice when prompted
    mocker.patch('builtins.input', return_value='1') # Simulate user choosing 'CartPole-v1'
    
    parsed_args = get_args()
    
    assert parsed_args.env_name == "CartPole-v1" # Default from ENV_MAP["1"]
    assert parsed_args.epochs == DEFAULT_EPOCHS # Check a few default values
    assert parsed_args.lr == DEFAULT_LR
    assert parsed_args.T_max == get_model_default_config()['T_max']

def test_get_args_specific_env(mocker):
    """Test get_args with a specific environment provided via command line."""
    test_env = "Pendulum-v1"
    mocker.patch.object(sys, 'argv', ['training_phase1.py', '--env_name', test_env, '--epochs', '5'])
    
    # input should not be called if env_name is provided
    mocked_input = mocker.patch('builtins.input')
    
    parsed_args = get_args()
    
    assert parsed_args.env_name == test_env
    assert parsed_args.epochs == 5
    mocked_input.assert_not_called()

def test_set_seed():
    """Test that set_seed runs without error."""
    try:
        set_seed(123)
        # Further checks could involve sampling from random, np.random, torch.rand
        # and ensuring they are the same after re-seeding, but this is often overkill.
        # For now, just ensuring it runs is a basic check.
        assert True 
    except Exception as e:
        pytest.fail(f"set_seed raised an exception: {e}")

def test_compute_returns_to_go():
    """Test the compute_returns_to_go function."""
    rewards = np.array([1.0, 1.0, 1.0, 1.0])
    gamma = 0.9
    # Expected RTGs:
    # rtg[3] = 1.0
    # rtg[2] = 1.0 + 0.9 * 1.0 = 1.9
    # rtg[1] = 1.0 + 0.9 * 1.9 = 1.0 + 1.71 = 2.71
    # rtg[0] = 1.0 + 0.9 * 2.71 = 1.0 + 2.439 = 3.439
    expected_rtgs = np.array([3.439, 2.71, 1.9, 1.0])
    
    calculated_rtgs = compute_returns_to_go(rewards, gamma)
    
    assert np.allclose(calculated_rtgs, expected_rtgs, atol=1e-3), \
        f"Expected {expected_rtgs}, but got {calculated_rtgs}"

    rewards_2 = np.array([0.0, 0.0, 1.0])
    gamma_2 = 1.0
    # Expected RTGs (gamma=1):
    # rtg[2] = 1.0
    # rtg[1] = 0.0 + 1.0 * 1.0 = 1.0
    # rtg[0] = 0.0 + 1.0 * 1.0 = 1.0
    expected_rtgs_2 = np.array([1.0, 1.0, 1.0])
    calculated_rtgs_2 = compute_returns_to_go(rewards_2, gamma_2)
    assert np.allclose(calculated_rtgs_2, expected_rtgs_2, atol=1e-3), \
        f"Expected {expected_rtgs_2}, but got {calculated_rtgs_2}"
        
    rewards_empty = np.array([])
    gamma_empty = 0.9
    expected_rtgs_empty = np.array([])
    calculated_rtgs_empty = compute_returns_to_go(rewards_empty, gamma_empty)
    assert np.array_equal(calculated_rtgs_empty, expected_rtgs_empty), \
        f"Expected {expected_rtgs_empty}, but got {calculated_rtgs_empty}"

def test_trajectory_buffer():
    """Test the TrajectoryBuffer class."""
    max_len, state_dim, act_dim_continuous = 5, 3, 2
    act_type_continuous = "continuous"
    
    # Test with continuous actions
    buffer_cont = TrajectoryBuffer(max_len, state_dim, act_dim_continuous, act_type_continuous)
    assert len(buffer_cont) == 0
    
    s1 = np.array([0.1, 0.2, 0.3])
    a1_cont = np.array([0.5, -0.5])
    r1 = 1.0
    
    s2 = np.array([0.4, 0.5, 0.6])
    a2_cont = np.array([-0.2, 0.8])
    r2 = 0.0

    buffer_cont.add(s1, a1_cont, r1)
    assert len(buffer_cont) == 1
    assert buffer_cont.states[0] is s1 # Check if it's storing the reference (or copy)
    
    buffer_cont.add(s2, a2_cont, r2)
    assert len(buffer_cont) == 2
    
    traj_cont = buffer_cont.get_trajectory()
    assert isinstance(traj_cont, dict)
    assert np.array_equal(traj_cont["states"], np.array([s1, s2], dtype=np.float32))
    assert np.array_equal(traj_cont["actions"], np.array([a1_cont, a2_cont], dtype=np.float32))
    assert np.array_equal(traj_cont["rewards"], np.array([r1, r2], dtype=np.float32))
    
    buffer_cont.reset()
    assert len(buffer_cont) == 0
    assert len(buffer_cont.states) == 0

    # Test with discrete actions
    act_dim_discrete = 1 # For discrete, this is often just a placeholder, actual value is action index
    act_type_discrete = "discrete"
    buffer_disc = TrajectoryBuffer(max_len, state_dim, act_dim_discrete, act_type_discrete)
    
    a1_disc = 1 # Action index
    a2_disc = 0
    
    buffer_disc.add(s1, a1_disc, r1)
    buffer_disc.add(s2, a2_disc, r2)
    
    traj_disc = buffer_disc.get_trajectory()
    assert np.array_equal(traj_disc["actions"], np.array([a1_disc, a2_disc], dtype=np.int64))

    # Test max_len constraint
    buffer_max = TrajectoryBuffer(2, state_dim, act_dim_continuous, act_type_continuous)
    buffer_max.add(s1, a1_cont, r1)
    buffer_max.add(s2, a2_cont, r2)
    s3 = np.array([0.7,0.8,0.9])
    a3_cont = np.array([0.1,0.1])
    r3 = -1.0
    buffer_max.add(s3, a3_cont, r3) # This should not be added
    
    assert len(buffer_max) == 2
    traj_max = buffer_max.get_trajectory()
    assert len(traj_max["states"]) == 2

# To run tests involving gym environments, gym needs to be installed.
# These tests will be skipped if gym is not available.
gym_available = False
try:
    import gym
    gym_available = True
except ImportError:
    pass

@pytest.mark.skipif(not gym_available, reason="gym package not found")
def test_collect_trajectories_cartpole(default_args):
    """Test collect_trajectories with CartPole-v1 for a few steps."""
    args = default_args
    args.offline_steps = 10 # Collect very few steps
    args.max_episode_length = 5 # Short episodes for quick termination
    
    # Make a temporary env to get dims (as in main script)
    # Ensure GYM_DISABLE_ENV_CHECKER is set for some envs if needed
    os.environ["GYM_DISABLE_ENV_CHECKER"] = "true"
    if not hasattr(np, "bool8"): np.bool8 = np.bool_ # Patch for gym
    if not hasattr(np, "float_"): np.float_ = np.float64


    env = gym.make(args.env_name) # CartPole-v1 from default_args
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n # CartPole has discrete actions
    act_type = "discrete"
    env.close()

    trajectories = collect_trajectories(
        env_name=args.env_name,
        state_dim=state_dim,
        act_dim=act_dim,
        act_type=act_type,
        offline_steps=args.offline_steps,
        max_episode_len=args.max_episode_length,
        gamma=args.gamma,
        seed=args.seed
    )

    assert isinstance(trajectories, list)
    assert len(trajectories) > 0 # Should collect at least one trajectory fragment
    
    total_collected_steps = 0
    for traj in trajectories:
        assert "states" in traj
        assert "actions" in traj
        assert "rewards" in traj
        assert len(traj["states"]) == len(traj["actions"]) == len(traj["rewards"])
        assert len(traj["states"]) <= args.max_episode_length
        total_collected_steps += len(traj["states"])
        
        assert traj["states"].dtype == np.float32
        assert traj["actions"].dtype == np.int64 # For discrete CartPole
        assert traj["rewards"].dtype == np.float32

    # Depending on episode termination, total_collected_steps might be slightly more
    # than args.offline_steps if the last episode goes over.
    # For such small numbers, it's often exactly offline_steps or a bit more.
    assert total_collected_steps >= args.offline_steps 

def test_trajectory_dataset_discrete(default_args):
    """Test TrajectoryDataset with a sample discrete action trajectory."""
    args = default_args
    args.max_length = 3 # K (context window for DT)
    args.gamma = 0.9

    # Sample trajectory (CartPole-like)
    # Episode length = 4, state_dim = 2, act_type = "discrete"
    sample_traj = {
        "states": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
        "actions": np.array([0, 1, 0, 1], dtype=np.int64), # Action indices
        "rewards": np.array([1.0, 1.0, 0.0, -1.0], dtype=np.float32)
    }
    trajectories = [sample_traj]
    act_type = "discrete"
    state_dim = sample_traj["states"].shape[1]

    dataset = TrajectoryDataset(trajectories, args, act_type)
    
    # Episode length is 4. K is 3.
    # Expected number of subsequences: 4
    # Seq 0 (len 1): S[0], A[0], Rtg[0], T[0] -> padded to 3
    # Seq 1 (len 2): S[0:1], A[0:1], Rtg[0:1], T[0:1] -> padded to 3
    # Seq 2 (len 3): S[0:2], A[0:2], Rtg[0:2], T[0:2] -> padded to 3
    # Seq 3 (len 3): S[1:3], A[1:3], Rtg[1:3], T[1:3] -> padded to 3
    assert len(dataset) == 4 

    # Check a specific item (e.g., the last one, which should be full length K)
    item = dataset[3] # Corresponds to original states [0.5,0.6], [0.7,0.8] (last two of S[1:3])
                      # and the one before it [0.3,0.4]

    assert item["states"].shape == (args.max_length, state_dim)
    assert item["actions"].shape == (args.max_length, 1) # Discrete actions are [K, 1]
    assert item["returns_to_go"].shape == (args.max_length, 1)
    assert item["timesteps"].shape == (args.max_length,) # Timesteps are [K]
    assert item["mask"].shape == (args.max_length,)

    assert item["states"].dtype == torch.float32
    assert item["actions"].dtype == torch.int64
    assert item["returns_to_go"].dtype == torch.float32
    assert item["timesteps"].dtype == torch.int64
    assert item["mask"].dtype == torch.float32 # Usually float for multiplying

    # Check padding and content for the last item (idx=3)
    # Original states for this subsequence: [[0.3,0.4], [0.5,0.6], [0.7,0.8]]
    # Original actions: [1, 0, 1]
    # Original timesteps (relative to subsequence): [0, 1, 2]
    # Original rewards for RTG: [1.0, 0.0, -1.0] (from sample_traj[1:])
    # RTGs for these rewards (gamma=0.9):
    # rtg_idx2 (orig_rwd_idx3=-1.0): -1.0
    # rtg_idx1 (orig_rwd_idx2=0.0):  0.0 + 0.9*(-1.0) = -0.9
    # rtg_idx0 (orig_rwd_idx1=1.0):  1.0 + 0.9*(-0.9) = 1.0 - 0.81 = 0.19
    # Expected RTG for item[3]: [0.19, -0.9, -1.0] (approx)

    expected_states_item3 = torch.tensor(sample_traj["states"][1:], device=DEVICE) # S[1:4]
    assert torch.allclose(item["states"], expected_states_item3)
    
    expected_actions_item3 = torch.tensor(sample_traj["actions"][1:].reshape(-1,1), device=DEVICE, dtype=torch.int64)
    assert torch.allclose(item["actions"], expected_actions_item3)

    # Timesteps are absolute from the episode start for the given subsequence
    # Subsequence for item[3] uses original states/actions/rewards at indices [1, 2, 3]
    # So, original timesteps are [1, 2, 3]
    expected_timesteps_item3 = torch.tensor(np.arange(1, 4), device=DEVICE, dtype=torch.int64)
    assert torch.allclose(item["timesteps"], expected_timesteps_item3)
    
    assert torch.allclose(item["mask"], torch.ones(args.max_length, device=DEVICE))


def test_trajectory_dataset_continuous(continuous_env_args):
    """Test TrajectoryDataset with a sample continuous action trajectory."""
    args = continuous_env_args
    args.max_length = 2 # K
    args.gamma = 1.0

    # Sample trajectory (Pendulum-like)
    # Episode length = 3, state_dim = 2, act_dim = 1, act_type = "continuous"
    sample_traj = {
        "states": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
        "actions": np.array([[-0.5], [0.0], [0.5]], dtype=np.float32), # Action vectors
        "rewards": np.array([-2.0, -1.0, 0.0], dtype=np.float32)
    }
    trajectories = [sample_traj]
    act_type = "continuous"
    state_dim = sample_traj["states"].shape[1]
    act_dim = sample_traj["actions"].shape[1]

    dataset = TrajectoryDataset(trajectories, args, act_type)
    assert len(dataset) == 3 # Episode len 3, K=2. Subsequences of len 1, 2, 2.

    item = dataset[2] # Last item, should be S[1:2], A[1:2], Rtg[1:2], T[0:1]
    assert item["states"].shape == (args.max_length, state_dim)
    assert item["actions"].shape == (args.max_length, act_dim) # Continuous actions [K, Adim]
    
    # Check padding for the first item (idx=0), actual length 1, padded to K=2
    item0 = dataset[0]
    assert item0["states"].shape == (args.max_length, state_dim)
    assert torch.allclose(item0["mask"], torch.tensor([0.0, 1.0], device=DEVICE)) # Padded at start
    assert torch.allclose(item0["states"][0], torch.zeros(state_dim, device=DEVICE)) # Padded part
    assert torch.allclose(item0["states"][1], torch.tensor(sample_traj["states"][0], device=DEVICE)) # Actual data

@pytest.mark.skipif(not gym_available, reason="gym package not found")
def test_model_initialization_discrete(default_args):
    """Test SpikingDecisionTransformer initialization for a discrete environment."""
    args = default_args # Uses CartPole-v1
    os.environ["GYM_DISABLE_ENV_CHECKER"] = "true"
    if not hasattr(np, "bool8"): np.bool8 = np.bool_
    if not hasattr(np, "float_"): np.float_ = np.float64

    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    env.close()

    model_config_args = {
        'state_dim': state_dim,
        'action_dim': act_dim,
        'embedding_dim': args.embedding_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'T_max': args.T_max,
        'max_length': args.max_length,
        'max_episode_len': args.max_episode_length,
        'dropout': args.dropout,
    }
    try:
        model = SpikingDecisionTransformer(**model_config_args).to(DEVICE)
        assert model is not None
        # Check a few params to ensure they are set
        assert model.state_dim == state_dim
        assert model.action_dim == act_dim
        assert model.max_length == args.max_length
        assert model.max_episode_len == args.max_episode_length
        assert len(model.layers) == args.num_layers
    except Exception as e:
        pytest.fail(f"Model initialization failed for discrete env: {e}")

@pytest.mark.skipif(not gym_available, reason="gym package not found")
def test_model_initialization_continuous(continuous_env_args):
    """Test SpikingDecisionTransformer initialization for a continuous environment."""
    args = continuous_env_args # Uses Pendulum-v1
    os.environ["GYM_DISABLE_ENV_CHECKER"] = "true"
    if not hasattr(np, "bool8"): np.bool8 = np.bool_
    if not hasattr(np, "float_"): np.float_ = np.float64
    
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0] # Continuous action dim
    env.close()

    model_config_args = {
        'state_dim': state_dim,
        'action_dim': act_dim,
        'embedding_dim': args.embedding_dim, # Use from fixture
        'num_layers': args.num_layers, # Use from fixture
        'num_heads': args.num_heads,   # Use from fixture
        'T_max': args.T_max,
        'max_length': args.max_length,
        'max_episode_len': args.max_episode_length,
        'dropout': args.dropout,
    }
    try:
        model = SpikingDecisionTransformer(**model_config_args).to(DEVICE)
        assert model is not None
        assert model.state_dim == state_dim
        assert model.action_dim == act_dim
        assert model.max_length == args.max_length
        assert model.max_episode_len == args.max_episode_length
    except Exception as e:
        pytest.fail(f"Model initialization failed for continuous env: {e}")

@pytest.mark.skipif(not gym_available, reason="gym package not found")
def test_training_step_discrete(default_args):
    """Test a single training step for a discrete action environment."""
    args = default_args # CartPole-v1
    args.offline_steps = args.batch_size * args.max_length # Ensure enough data for one batch
    args.log_interval = 1 # To avoid issues if checkpointing is tied to log interval
    
    os.environ["GYM_DISABLE_ENV_CHECKER"] = "true"
    if not hasattr(np, "bool8"): np.bool8 = np.bool_
    if not hasattr(np, "float_"): np.float_ = np.float64
    
    set_seed(args.seed)
    
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    act_type = "discrete"
    env.close()

    trajectories = collect_trajectories(args.env_name, state_dim, act_dim, act_type, 
                                        args.offline_steps, args.max_episode_length, 
                                        args.gamma, args.seed)
    if not trajectories: pytest.fail("No trajectories collected for discrete training step test.")

    dataset = TrajectoryDataset(trajectories, args, act_type)
    if len(dataset) < args.batch_size: pytest.fail(f"Dataset too small for batch size {args.batch_size}")
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model_config = {
        'state_dim': state_dim, 'action_dim': act_dim, 'embedding_dim': args.embedding_dim,
        'num_layers': args.num_layers, 'num_heads': args.num_heads, 'T_max': args.T_max,
        'max_length': args.max_length, 'max_episode_len': args.max_episode_length,
        'dropout': args.dropout
    }
    model = SpikingDecisionTransformer(**model_config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn_action = torch.nn.CrossEntropyLoss()

    try:
        batch = next(iter(dataloader))
        states, actions_target = batch["states"], batch["actions"]
        returns_to_go, timesteps = batch["returns_to_go"], batch["timesteps"]

        action_input_tensor = torch.zeros(args.batch_size, args.max_length, act_dim, dtype=torch.float, device=DEVICE)
        if args.max_length > 1:
            one_hot_prev_actions = torch.nn.functional.one_hot(
                actions_target[:, :-1].squeeze(-1), num_classes=act_dim).float()
            action_input_tensor[:, 1:] = one_hot_prev_actions
        
        actions_target_for_loss = actions_target.squeeze(-1)

        model_output = model(states, action_input_tensor, returns_to_go, timesteps)
        action_predictions = model_output['action_predictions']
        model_metrics = model_output['metrics']
        
        action_loss = loss_fn_action(action_predictions.reshape(-1, act_dim), actions_target_for_loss.reshape(-1))
        
        # compute_spiking_loss is imported from phase1_integration_with_sdt at the top of the file
        total_loss = compute_spiking_loss(action_loss, model_metrics, args.lambda_reg)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        assert total_loss.item() is not None
        assert not torch.isnan(total_loss)
    except Exception as e:
        pytest.fail(f"Training step failed for discrete env: {e}")


@pytest.mark.skipif(not gym_available, reason="gym package not found")
def test_training_step_continuous(continuous_env_args):
    """Test a single training step for a continuous action environment."""
    args = continuous_env_args # Pendulum-v1
    args.offline_steps = args.batch_size * args.max_length * 2 # Collect a bit more for continuous
    args.log_interval = 1

    os.environ["GYM_DISABLE_ENV_CHECKER"] = "true"
    if not hasattr(np, "bool8"): np.bool8 = np.bool_
    if not hasattr(np, "float_"): np.float_ = np.float64

    set_seed(args.seed)

    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0] # Continuous
    act_type = "continuous"
    env.close()

    trajectories = collect_trajectories(args.env_name, state_dim, act_dim, act_type, 
                                        args.offline_steps, args.max_episode_length, 
                                        args.gamma, args.seed)
    if not trajectories: pytest.fail("No trajectories collected for continuous training step test.")
    
    dataset = TrajectoryDataset(trajectories, args, act_type)
    if len(dataset) < args.batch_size: pytest.fail(f"Dataset too small for batch size {args.batch_size}")
        
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model_config = {
        'state_dim': state_dim, 'action_dim': act_dim, 'embedding_dim': args.embedding_dim,
        'num_layers': args.num_layers, 'num_heads': args.num_heads, 'T_max': args.T_max,
        'max_length': args.max_length, 'max_episode_len': args.max_episode_length,
        'dropout': args.dropout
    }
    model = SpikingDecisionTransformer(**model_config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn_action = torch.nn.MSELoss()

    try:
        batch = next(iter(dataloader))
        states, actions_target = batch["states"], batch["actions"]
        returns_to_go, timesteps = batch["returns_to_go"], batch["timesteps"]

        action_input_tensor = torch.zeros_like(actions_target, dtype=torch.float, device=DEVICE)
        if args.max_length > 1:
            action_input_tensor[:, 1:] = actions_target[:, :-1]
        
        actions_target_for_loss = actions_target

        model_output = model(states, action_input_tensor, returns_to_go, timesteps)
        action_predictions = model_output['action_predictions']
        model_metrics = model_output['metrics']
        
        action_loss = loss_fn_action(action_predictions, actions_target_for_loss)
        
        # compute_spiking_loss is imported from phase1_integration_with_sdt at the top of the file
        total_loss = compute_spiking_loss(action_loss, model_metrics, args.lambda_reg)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        assert total_loss.item() is not None
        assert not torch.isnan(total_loss)
    except Exception as e:
        pytest.fail(f"Training step failed for continuous env: {e}")