from grecom import run_experiment
from grecom.parser import parse_inputs
from grecom.data import prepare_data
from grecom.utils import init_pytorch, init_results
import pandas as pd


if __name__ == "__main__":
    args = parse_inputs()
    args = prepare_data(args)
    args = init_pytorch(args)
    args = init_results(args)
    all_res = pd.DataFrame()
    for i in range(args.n_runs):
        res = run_experiment(args)
        all_res = all_res.append(res)
        if not args.same_split:
            args.split_id += 1
        if args.eval:
            all_res.to_csv(f"{args.split_path}/erun_{args.model}.csv", sep=';')
            (all_res.groupby(['rating_type'], as_index=False)
                .agg(['mean','std']).to_csv(
                    f"{args.split_path}/eres_{args.model}.csv", sep=';'
            ))
        else:
            all_res.to_csv(f"{args.split_path}/run_{args.model}.csv", sep=';')
            (all_res.groupby(['rating_type'], as_index=False)
                .agg(['mean','std']).to_csv(
                    f"{args.split_path}/res_{args.model}.csv", sep=';'
            ))
