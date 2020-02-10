./libfm-1.40.windows/libfm \
-task r \
-train data/ml-1m/libfm/train \
-test data/ml-1m/libfm/test \
-dim ’1,1,32’ \
-iter 16384 \
-method mcmc \
-init_stdev 0.1 \
--relation data/ml-1m/libfm/rel.user,data/ml-1m/libfm/rel.item,data/ml-1m/libfm/rel.time