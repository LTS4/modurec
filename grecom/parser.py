import argparse


def common_parser(parser):
    parser.add_argument('--isotropic', action='store_true', default=False)
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
    # Print
    parser.add_argument('--log-interval', type=int, default=50,
                        help='how many batches to wait before logging training status')
    # Save
    parser.add_argument('--save-model', action='store_true', default=False)
    parser.add_argument('--save-results', action='store_true', default=False)
    parser.add_argument('--id', type=int, help='identifier of the experiment')
    # Paths
    parser.add_argument('--raw-path', type=str, default='/datasets2/recom_heterograph/raw/')
    parser.add_argument('--data-path', type=str, default='/datasets2/recom_heterograph/data/')
    parser.add_argument('--models-path', type=str, default='/datasets2/recom_heterograph/models/')
    parser.add_argument('--results-path', type=str, default='/datasets2/recom_heterograph/results/')


def parser_recommender():
    parser = argparse.ArgumentParser(description='PyTorch Recommender System')
    parser.add_argument('--dataset', type=str, help='dataset',
                        choices=['movielens100k'])
    parser.add_argument('--layers', type=int, default=1, help='number of hidden layers')
    parser.add_argument('--channels', type=int, help='units in the hidden layer', default=16)
    common_parser(parser)
    # Parse the arguments
    args = parser.parse_args()
    assert args.layers > 0, "The model should have at least 1 hidden layer"
    return args
