echo "--- STEP 2: STARTING TRAINING RUNS (This will take a long time!) ---"

# --- ENVIRONMENT: CartPole-v1 ---
for seed in 42 123 456
do
  # Train DT
  python scripts/train.py --model dt --env CartPole-v1 --seed $seed --save-dir results/CartPole-v1/dt_seed$seed
  # Train SNN-DT
  python scripts/train.py --model snn_dt --env CartPole-v1 --seed $seed --save-dir results/CartPole-v1/snn_dt_seed$seed
  # Train DS-Former
  python scripts/train.py --model dsformer --env CartPole-v1 --seed $seed --save-dir results/CartPole-v1/dsformer_seed$seed
  # Train IQL
  python scripts/train.py --model iql --env CartPole-v1 --seed $seed --save-dir results/CartPole-v1/iql_seed$seed
  # Train CQL
  python scripts/train.py --model cql --env CartPole-v1 --seed $seed --save-dir results/CartPole-v1/cql_seed$seed
done

# --- ENVIRONMENT: Pendulum-v1 ---
for seed in 42 123 456
do
  # Train DT
  python scripts/train.py --model dt --env Pendulum-v1 --seed $seed --save-dir results/Pendulum-v1/dt_seed$seed
  # Train SNN-DT
  python scripts/train.py --model snn_dt --env Pendulum-v1 --seed $seed --save-dir results/Pendulum-v1/snn_dt_seed$seed
  # Train DS-Former
  python scripts/train.py --model dsformer --env Pendulum-v1 --seed $seed --save-dir results/Pendulum-v1/dsformer_seed$seed
  # Train IQL
  python scripts/train.py --model iql --env Pendulum-v1 --seed $seed --save-dir results/Pendulum-v1/iql_seed$seed
  # Train CQL
  python scripts/train.py --model cql --env Pendulum-v1 --seed $seed --save-dir results/Pendulum-v1/cql_seed$seed
done

echo "--- All training runs complete ---"