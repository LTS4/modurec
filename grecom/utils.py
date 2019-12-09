import os

import torch


def init_pytorch(args):
    """Initialize pytorch seed and device used.

    :param args: Dictionary with execution arguments
    :type args: Namespace
    """
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.torch_seed)
    if use_cuda:
        args.device = torch.device("cuda:" + str(args.gpu))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    else:
        args.device = "cpu"
    print('Device used:', args.device)
    return args


def init_results(args):
    """Create result folders if they don't exist. Add results extra args.

    :param args: Dictionary with execution arguments
    :type args: Namespace
    """
    args.id = f"{args.model}_lr{args.lr}_ep{args.epochs}"
    args.results_path = os.path.join(args.results_path, args.dataset)
    if args.split_type == 'predefined':
        split_folder = args.split_type
    elif args.split_type == 'random':
        split_folder = f"{args.split_type}_t{args.train_prop:.2f}"
    args.split_path = os.path.join(args.results_path, split_folder)
    os.makedirs(args.split_path, exist_ok=True)
    return args
