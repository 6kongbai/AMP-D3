import argparse
from datetime import datetime


def setting_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_log_dir', default='tf-logs',
                        help='Path to log file.')

    parser.add_argument('--out_checkpoint_dir', default=f'checkpoints/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}')

    parser.add_argument('--save_top_k', default=1,
                        help='save_top_k for train.')

    parser.add_argument('--gpus', default=1,
                        help='gpus for train.')

    parser.add_argument('--max_epochs', default=1000,
                        help='max_epochs for train.')

    parser.add_argument('--batch_size', default=256,
                        help='batch_sizes for train.')

    parser.add_argument('--n_timestep', default=1000,
                        help='n_timestep for diffusion model.')

    parser.add_argument('--max_length', default=50,
                        help='seq max len.')

    parser.add_argument('--beta_schedule', default='linear',
                        help='beta_schedule for diffusion model.')

    parser.add_argument('--beta_start', default=1.e-7,
                        help='beta_start for diffusion model.')

    parser.add_argument('--beta_end', default=2.e-2,
                        help='beta_end for diffusion model.')

    parser.add_argument('--temperature', default=0.1,
                        help='temperature for diffusion model.')

    parser.add_argument('--learning_rate_struct', default=1e-4,
                        help='learning_rate_struct for diffusion model.')

    parser.add_argument('--learning_rate_seq', default=1e-4,
                        help='learning_rate_seq for diffusion model.')

    parser.add_argument('--learning_rate_cont', default=1e-4,
                        help='learning_rate_cont for diffusion model.')

    return parser.parse_args()
