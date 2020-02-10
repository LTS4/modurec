python main.py \
--dataset ml-100k \
--split_type random \
--train_prop 0.8 \
--model Autorec \
--epochs 1000 \
--lr 2e-3 \
--reg 1e-4 \
--n_runs 10

python main.py \
--dataset ml-100k \
--split_type random \
--train_prop 0.8 \
--model Autorec_D \
--epochs 1000 \
--lr 2e-3 \
--reg 1e-4 \
--n_runs 10

python main.py \
--dataset ml-100k \
--split_type random \
--train_prop 0.8 \
--model Autorec_DT \
--epochs 1000 \
--lr 2e-3 \
--reg 1e-4 \
--n_runs 10

python main.py \
--dataset ml-100k \
--split_type random \
--train_prop 0.8 \
--model Autorec_DG \
--epochs 1000 \
--lr 2e-3 \
--reg 1e-4 \
--n_runs 10

python main.py \
--dataset ml-100k \
--split_type random \
--train_prop 0.8 \
--model Autorec_DGT \
--epochs 1000 \
--lr 2e-3 \
--reg 1e-4 \
--n_runs 10

python main.py \
--dataset ml-100k \
--split_type random \
--train_prop 0.8 \
--model Autorec_DFT \
--epochs 1000 \
--lr 2e-3 \
--reg 1e-4 \
--n_runs 10

python main.py \
--dataset ml-100k \
--split_type random \
--train_prop 0.8 \
--model Autorec_DFGT \
--epochs 1000 \
--lr 2e-3 \
--reg 1e-4 \
--n_runs 10