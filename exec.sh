for i in {1..1}
do
    python train.py \
    --data-path data --raw-path raw \
    --models-path models --results-path results \
    --dataset douban \
    --model gautorec \
    --reg 1e-4 --lr 2e-3 \
    --epochs 2500 \
    --no-time \
    --no-features \
    --no-conv \
    --testing
done