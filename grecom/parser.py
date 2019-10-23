import argparse


def common_parser(parser):
    parser.add_argument('--experiments', type=int, default=1,
                        help='Number of experiments to average results on')
    # Learning parameters
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--early-stopping', type=int, default=0,
                        help='Patience -  0 = disable (default: 0)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    # GPU
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu', type=int, default=0, help='id of gpu device')

    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    # Save
    parser.add_argument('--id', type=str, help='identifier of the experiment')
    # Paths
    parser.add_argument('--raw-path', type=str, default='/datasets2/recom_heterograph/raw/')
    parser.add_argument('--data-path', type=str, default='/datasets2/recom_heterograph/data/')
    parser.add_argument('--models-path', type=str, default='/datasets2/recom_heterograph/models/')
    parser.add_argument('--results-path', type=str, default='/datasets2/recom_heterograph/results/')


def parser_recommender():
    parser = argparse.ArgumentParser(description='PyTorch Recommender System')
    parser.add_argument('--dataset', type=str, help='dataset',
                        choices=['movielens100k'])
    parser.add_argument('--model', type=str, help='model',
                        choices=['hetero_gcmc', 'gautorec'])
    common_parser(parser)
    # Parse the arguments
    args = parser.parse_args()
    return args
