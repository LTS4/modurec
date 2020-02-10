./libfm-1.40.windows/libfm \
-task r \
-train data/ml-10m/libfm/train \
-test data/ml-10m/libfm/test \
-dim ’1,1,128’ \
-iter 512 \
-method mcmc \
-init_stdev 0.1
#--relation data/ml-10m/libfm/rel.user,data/ml-10m/libfm/rel.item,data/ml-10m/libfm/rel.time