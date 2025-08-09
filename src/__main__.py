
import argparse
from .run_dsf_baseline import experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse, no-reward-decay
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dc')  
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='gelu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=10) #100
    parser.add_argument('--save_path', type=str, default='save')
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--max_iters_token', type=int, default=5)
    parser.add_argument('--num_steps_per_iter', type=int, default=1000) # 10000
    parser.add_argument('--num_steps_per_iter_token', type=int, default=1000) # 10000
    parser.add_argument('--remove_act_embs', action='store_true')

    parser.add_argument('--load_tokenizer', action='store_true')
    parser.add_argument('--tokenizer_path', type=str) # for loading pretrained tokenizer
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    # bias_window_size for freeFormer
    parser.add_argument('--bias_window_size', type=int, default=8) # bias_window_size, default=8, not use
    # full

    # convolution
    parser.add_argument('--conv_window_size', type=int, default=6)
    # setting name
    parser.add_argument('--setting_name', type=str, default=None)
    # pool size
    parser.add_argument('--pool_size', type=int, default=6)
    # encoder_have_cnn
    parser.add_argument('--encoder_type', type=str, default=None)
    parser.add_argument('--encoder_have_cnn', action='store_true')
    # spike former
    parser.add_argument('--warmup_ratio',  type=float, default=0.10)

    args = parser.parse_args()

    experiment(variant=vars(args))
