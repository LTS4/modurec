from grecom import run_experiment
from grecom.parser import parse_inputs
from grecom.data import prepare_data
from grecom.utils import init_pytorch, init_results


if __name__ == "__main__":
    args = parse_inputs()
    args = prepare_data(args)
    args = init_pytorch(args)
    args = init_results(args)
    for i in range(args.n_runs):
        run_experiment(args)
        if not args.same_split:
            args.split_id += 1
