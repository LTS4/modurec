./libfm-1.40.windows/libfm \
-task r \
-train data/ml-100k/libfm/train \
-test data/ml-100k/libfm/test \
-dim ’1,1,8’ \
-iter 16384 \
-method mcmc \
-init_stdev 0.1 \
--relation data/ml-100k/libfm/rel.user,data/ml-100k/libfm/rel.item