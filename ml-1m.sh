python main.py \
--dataset ml-1m \
--split_type random \
--model Autorec_D \
--epochs 10000 \
--lr 2e-3 \
--weight_decay 1e-4 \
--train_prop 0.9 \
--n_runs 10 \
--results_path results/