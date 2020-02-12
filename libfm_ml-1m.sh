./libfm-1.40.windows/libfm \
-task r \
-train data/ml-1m/libfm/fb_train \
-test data/ml-1m/libfm/fb_test \
-dim ’1,1,8’ \
-iter 16384 \
-method mcmc \
-init_stdev 0.1 \
-cache_size 500000000