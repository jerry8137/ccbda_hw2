import argparse


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='data')
    parser.add_argument('--lr', type=float,
                        default=0.0003)
    parser.add_argument('--batch_size', type=int,
                        default=2048)
    parser.add_argument('--num_epoch', type=int,
                        default=1000)
    parser.add_argument('--num_workers', type=int,
                        default=8)
    # model param
    parser.add_argument('--num_class', type=int,
                        default=128)
    parser.add_argument('--hidden_size', type=int,
                        default=128)
    parser.add_argument('--weight_decay', type=float,
                        default=0)
    parser.add_argument('--accumulate', type=int,
                        default=1)

    parser.add_argument('--weight', type=str,
                        default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--exp_name', type=str,
                        default='exp')

    return parser.parse_args()
