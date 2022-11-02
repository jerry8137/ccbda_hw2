import argparse


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='data')
    parser.add_argument('--lr', type=float,
                        default=0.01)
    parser.add_argument('--batch_size', type=int,
                        default=8)
    parser.add_argument('--num_epoch', type=int,
                        default=300)
    parser.add_argument('--num_workers', type=int,
                        default=8)
    # model param
    parser.add_argument('--num_class', type=int,
                        default=4)
    parser.add_argument('--hidden_size', type=int,
                        default=512)
    parser.add_argument('--weight_decay', type=float,
                        default=0)

    parser.add_argument('--weight', type=str,
                        default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--exp_name', type=str,
                        default='exp')

    return parser.parse_args()
