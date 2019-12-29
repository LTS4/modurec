from argparse import ArgumentParser


def add_experiment_args(parser):
    """Add parameters related with the experiment

    :param parser: Parser object without the experiment parameters
    :type parser: ArgumentParser
    """
    parser.add_argument(
        '--dataset', required=True, type=str,
        help='Dataset to run the experiment on')
    parser.add_argument(
        '--model', required=True, type=str,
        help='Model to run in the experiment')
    parser.add_argument(
        '--n_runs', type=int, default=1,
        help='Number of runs to average results on')
    parser.add_argument(
        '--preprocess', action='store_true', default=False,
        help='Preprocess data (even if exists)')


def add_splitting_args(parser):
    """Add parameters related with train-test splitting

    :param parser: Parser object without the experiment parameters
    :type parser: ArgumentParser
    """
    parser.add_argument(
        '--split_type', required=True, type=str,
        choices=['predefined', 'random'],
        help='Splitting scenario')
    parser.add_argument(
        '--split_id', type=int, default=1,
        help='Which split to choose (seed used if not predefined scenario)')
    parser.add_argument(
        '--same_split', action='store_true', default=False,
        help='Use the same split across experiments')
    parser.add_argument(
        '--train_prop', type=float, default=0.9,
        help='Proportion used to train. Not valid for predefined scenario.')


def add_hardware_args(parser):
    """Add parameters related with the hardware

    :param parser: Parser object without the hardware parameters
    :type parser: ArgumentParser
    """
    parser.add_argument(
        '--no-cuda', action='store_true', default=False,
        help='Disables CUDA training')
    parser.add_argument(
        '--gpu', type=int, default=0,
        help='Id of gpu device')
    parser.add_argument(
        '--n_cores', type=int, default=-1,
        help='Number of cpu cores')
    parser.add_argument(
        '--torch_seed', type=int, default=0,
        help='Pytorch seed')


def add_hyperparameters(parser):
    """Add parameters related with training

    :param parser: Parser object without the hyperparameters
    :type parser: ArgumentParser
    """
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='Learning rate (default: 0.001)')
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of epochs to train (default: 100)')
    parser.add_argument(
        '--reg', type=float, default=0.001,
        help='General regularization parameter')


def add_path_args(parser):
    """Add parameters related with paths

    :param parser: Parser object without the paths
    :type parser: ArgumentParser
    """
    parser.add_argument(
        '--raw-path', type=str, default='raw/',
        help='Path for the raw data')
    parser.add_argument(
        '--data-path', type=str, default='data/',
        help='Path for the processed data')
    parser.add_argument(
        '--models-path', type=str, default='models/',
        help='Path for the models')
    parser.add_argument(
        '--results-path', type=str, default='results/',
        help='Path for the results')


def parse_inputs():
    """Parse all input arguments from command line

    :return: parsed parameters
    :rtype: dict
    """
    parser = ArgumentParser(description='Autorec++')
    add_experiment_args(parser)
    add_splitting_args(parser)
    add_hardware_args(parser)
    add_hyperparameters(parser)
    add_path_args(parser)
    args = parser.parse_args()
    return args
