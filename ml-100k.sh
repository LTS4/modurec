python main.py \
--dataset ml-100k \
--split_type predefined \
--model Autorec_DT \
--epochs 5000 \
--lr 2e-3 \
--weight_decay 1e-3 \
--same_split \
--n_runs 10 \
--results_path results/

# python main.py --dataset ml-100k --split_type predefined --model Autorec_DFT --epochs 5000 --lr 2e-3 --weight_decay 1e-3 --same_split --n_runs 1 --results_path results/